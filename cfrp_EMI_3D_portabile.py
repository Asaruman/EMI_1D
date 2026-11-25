"""
CFRP EMI - MODELLO 3D DELLA PROVA A FINESTRA, VERSIONE PORTABILE
================================================================
Stessa fisica di cfrp_EMI_3D.py (NGSolve), ma scritto con librerie in puro
Python / SciPy che girano su QUALSIASI processore (anche Xeon senza AVX2):
  numpy, scipy, scikit-fem   (+ pypardiso facoltativo: solutore molto piu' veloce)

Prova riprodotta: pannello CFRP 55 x 25 cm nell'apertura di una parete in rame,
collegato con nastro conduttivo; sorgente a 30 cm davanti, ricezione a 30 cm
dietro. SE = |campo con apertura vuota| / |campo con pannello|, come in laboratorio.

Modello:
  - Maxwell armonico 3D (convenzione e^{jwt}), elementi di Nedelec del 1o ordine
  - mesh di tetraedri su griglia graduata: fitta vicino ad apertura e nastro
  - PML (strati assorbenti) su tutti i lati; parete e bordo esterno = rame ideale
  - pannello = superficie con l'impedenza di trasferimento del laminato dal
    modello 1D verificato (con o senza maglia Cu); nastro = fascia lungo il bordo
    con l'impedenza del giunto dal modello 1D
  - sorgente: dipolo (antenna biconica) o spira (bobina)

Uso:
  python cfrp_EMI_3D_portabile.py --prova            mesh grossolana + verifiche (pochi minuti)
  python cfrp_EMI_3D_portabile.py                    64/96/128 MHz, con e senza rame
  python cfrp_EMI_3D_portabile.py --sorgente spira
  python cfrp_EMI_3D_portabile.py --fine             mesh piu' fitta (piu' RAM e tempo)
  python cfrp_EMI_3D_portabile.py --freq 96          solo alcune frequenze [MHz]
Installazione: python -m pip install numpy scipy scikit-fem matplotlib pypardiso
Serve cfrp_EMI_finale.py nella stessa cartella.
Output: EMI3Dp_risultati.csv, EMI3Dp_SE.png
"""
import argparse
import csv
import time
import numpy as np
import scipy.sparse as sp
from skfem import (MeshTet, ElementTetN0, Basis, InteriorFacetBasis, BilinearForm,
                   LinearForm, asm, condense, solve)
from skfem.helpers import dot
import cfrp_EMI_finale as m1d

try:
    import pypardiso
    _solve = pypardiso.spsolve
    SOLVER = "pardiso"
except Exception:
    from scipy.sparse.linalg import spsolve as _solve
    SOLVER = "scipy (SuperLU)"

MU0, EPS0, ETA0 = m1d.MU0, m1d.EPS0, m1d.ETA0

# ======================= GEOMETRIA DELLA PROVA =======================
AP_X, AP_Y = 0.55, 0.25     # m, pannello / apertura (55 cm parallelo a E)
B_NASTRO = m1d.L_NASTRO     # m, fascia del nastro lungo il bordo
D_SORG = m1d.D_SORGENTE     # m, sorgente davanti alla parete
D_RIC = 0.30                # m, ricezione dietro la parete (DA CONFERMARE)
L_DIPOLO = 0.50             # m, dipolo equivalente alla biconica
R_SPIRA = 0.10              # m, raggio della spira
DOM_X, DOM_Y, DOM_Z = 0.60, 0.45, 0.55   # m, mezza larghezza del dominio fisico
PML_T, PML_A = 0.30, 15.0                # spessore e intensita' della PML
FREQS = [64e6, 96e6, 128e6]


def griglia(lo, hi, fini, ancore, hmin, hmax, crescita=1.6):
    """Nodi 1D graduati: passo hmin sulle linee 'fini', cresce fino a hmax;
    le 'ancore' (bordi di dominio e PML) sono incluse senza raffinare."""
    fini = np.array(sorted(set(fini)))
    h = lambda s: min(hmax, hmin + (crescita - 1) * np.min(np.abs(fini - s)))
    pts, s = [lo], lo
    while s < hi:
        s = s + h(s)
        pts.append(min(s, hi))
    obblig = np.unique(np.round(np.concatenate([fini, ancore]), 9))
    obblig = obblig[(obblig >= lo) & (obblig <= hi)]
    pts = np.unique(np.round(np.concatenate([np.array(pts), obblig]), 9))
    keep = []
    for p in pts:   # toglie nodi liberi troppo vicini ai nodi obbligati
        if np.any(np.isclose(p, obblig)) or np.min(np.abs(obblig - p)) > 0.4 * h(p):
            keep.append(p)
    return np.array(keep)


def crea_mesh(hmin, hmax, sorgente):
    P = PML_T
    b = B_NASTRO
    anc = lambda L: [-L - P, -L, L, L + P]
    fx = [-AP_X / 2, AP_X / 2, -AP_X / 2 + b, AP_X / 2 - b]
    fy = [-AP_Y / 2, AP_Y / 2, -AP_Y / 2 + b, AP_Y / 2 - b, -0.01, 0.01]
    fz = [0.0, -D_SORG - 0.01, -D_SORG + 0.01]
    ax_, az_ = [0.0], [D_RIC]
    if sorgente == "dipolo":
        ax_ += [-L_DIPOLO / 2, L_DIPOLO / 2]
    else:
        fx += [-R_SPIRA, R_SPIRA]
        fz += [-D_SORG - R_SPIRA, -D_SORG + R_SPIRA]
    x = griglia(-DOM_X - P, DOM_X + P, fx, anc(DOM_X) + ax_, hmin, hmax)
    y = griglia(-DOM_Y - P, DOM_Y + P, fy, anc(DOM_Y), hmin, hmax)
    z = griglia(-DOM_Z - P, DOM_Z + P, fz, anc(DOM_Z) + az_, hmin, hmax)
    return MeshTet.init_tensor(x, y, z)


def stretching(c, L):
    d = np.clip((np.abs(c) - L) / PML_T, 0, None)
    return 1 - 1j * PML_A * d**2


def zs_pannello(f, con_rame):
    M = m1d.abcd(np.array([f]), m1d.sigma_laminato(), m1d.zs_maglia(np.array([f]), con_rame))
    zt = 1 / M[0, 1, 0]
    return abs(zt) * (1 + 1j) / np.sqrt(2)          # modulo esatto, fase passiva


def zs_nastro(f):
    return complex(m1d.z_giunto_nastro(np.array([f]))[0]) / B_NASTRO


class Modello:
    def __init__(self, hmin, hmax, sorgente):
        t0 = time.time()
        self.sorgente = sorgente
        self.m = crea_mesh(hmin, hmax, sorgente)
        self.e = ElementTetN0()
        self.basis = Basis(self.m, self.e)
        m = self.m
        # facce sul piano z = 0
        fmid = m.p[:, m.facets].mean(axis=1)
        on0 = np.abs(fmid[2]) < 1e-9
        ax, ay = np.abs(fmid[0]), np.abs(fmid[1])
        pann = on0 & (ax < AP_X / 2 - B_NASTRO) & (ay < AP_Y / 2 - B_NASTRO)
        ap = on0 & (ax < AP_X / 2) & (ay < AP_Y / 2)
        self.f_pann = np.nonzero(pann)[0]
        self.f_nastro = np.nonzero(ap & ~pann)[0]
        f_parete = np.nonzero(on0 & ~ap)[0]
        pec = np.concatenate([f_parete, m.boundary_facets()])
        self.D = self.basis.get_dofs(facets=pec).all()
        # elementi della sorgente
        c = m.p[:, m.t].mean(axis=1)
        if sorgente == "dipolo":
            sel = (np.abs(c[0]) < L_DIPOLO / 2) & (np.abs(c[1]) < 0.01) & (np.abs(c[2] + D_SORG) < 0.01)
        else:
            r = np.sqrt(c[0]**2 + (c[2] + D_SORG)**2)
            sel = (np.abs(c[1]) < 0.01) & (np.abs(r - R_SPIRA) < 0.012)
        self.el_ant = np.nonzero(sel)[0]
        print(f"Mesh: {m.t.shape[1]} tetraedri, {self.basis.N} incognite, sorgente {sorgente} "
              f"({len(self.el_ant)} elementi), pannello {len(self.f_pann)} facce, nastro "
              f"{len(self.f_nastro)} facce | solutore {SOLVER} | {time.time()-t0:.0f} s", flush=True)
        self._K = None

    def matrici(self, w):
        """Parte di volume (dipende solo da w)."""
        @BilinearForm(dtype=np.complex128)
        def volume(u, v, wf):
            sx = stretching(wf.x[0], DOM_X); sy = stretching(wf.x[1], DOM_Y); sz = stretching(wf.x[2], DOM_Z)
            L = [sy * sz / sx, sx * sz / sy, sx * sy / sz]
            cc = sum(u.curl[i] * v.curl[i] / L[i] for i in range(3)) / MU0
            mm = sum(L[i] * u.value[i] * v.value[i] for i in range(3)) * EPS0 * w**2
            return cc - mm
        return asm(volume, self.basis)

    def foglio(self, facets):
        fb = InteriorFacetBasis(self.m, self.e, facets=facets, side=0)

        @BilinearForm
        def tang(u, v, wf):
            n = wf.n
            ut = u.value - dot(u.value, n) * n
            vt = v.value - dot(v.value, n) * n
            return dot(ut, vt)
        return asm(tang, fb)

    def rhs(self, w):
        bA = Basis(self.m, self.e, elements=self.el_ant)
        tipo = self.sorgente

        @LinearForm(dtype=np.complex128)
        def J(v, wf):
            if tipo == "dipolo":
                return -1j * w * v.value[0]
            rx, rz = wf.x[0], wf.x[2] + D_SORG
            r = np.sqrt(rx**2 + rz**2)
            return -1j * w * (-rz / r * v.value[0] + rx / r * v.value[2])
        return asm(J, bA)

    def prepara(self):
        if self._K is None:
            self.S_p = self.foglio(self.f_pann)
            self.S_n = self.foglio(self.f_nastro)
            c = self.m.p[:, self.m.t].mean(axis=1)
            sel = (np.abs(c[0]) < 0.075) & (np.abs(c[1]) < 0.075) & (np.abs(c[2] - D_RIC) < 0.075)
            self.b_ric = Basis(self.m, self.e, elements=np.nonzero(sel)[0])
            self._K = True

    def risolvi(self, f, zp=None, zn=None):
        w = 2 * np.pi * f
        self.prepara()
        A = self.matrici(w)
        if zp is not None:
            A = A + 1j * w / zp * self.S_p + 1j * w / zn * self.S_n
        b = self.rhs(w)
        Ac, bc, xc, I = condense(A.tocsr(), b, D=self.D)
        x = np.zeros(self.basis.N, complex)
        if SOLVER == "pardiso":   # pypardiso accetta solo reali: sistema 2N equivalente
            Ar, Ai = Ac.real.tocsr(), Ac.imag.tocsr()
            R = sp.bmat([[Ar, -Ai], [Ai, Ar]], format="csr")
            y = _solve(R, np.concatenate([bc.real, bc.imag]))
            x[I] = y[:len(bc)] + 1j * y[len(bc):]
        else:
            x[I] = _solve(Ac.tocsc(), bc)
        return x

    def ricevuto(self, x, f):
        """|E| e |H| quadratici medi nei tetraedri entro +-7.5 cm dal ricevitore
        (H dal rotore esatto degli elementi di Nedelec)."""
        w = 2 * np.pi * f
        fld = self.b_ric.interpolate(x)
        dV = self.b_ric.dx
        E2 = (np.sum(np.abs(fld.value) ** 2, axis=0) * dV).sum() / dV.sum()
        H2 = (np.sum(np.abs(fld.curl) ** 2, axis=0) * dV).sum() / dV.sum() / (w * MU0) ** 2
        return np.sqrt(E2), np.sqrt(H2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prova", action="store_true")
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--sorgente", choices=["dipolo", "spira"], default="dipolo")
    ap.add_argument("--freq", type=float, nargs="*")
    a = ap.parse_args()
    hmin, hmax = (0.0125, 0.15) if a.prova else (0.008, 0.07) if a.fine else (0.0125, 0.10)
    freqs = [96e6] if a.prova else FREQS
    if a.freq:
        freqs = [v * 1e6 for v in a.freq]
    t0 = time.time()
    M = Modello(hmin, hmax, a.sorgente)
    righe = []
    for f in freqs:
        t1 = time.time()
        e0, h0 = M.ricevuto(M.risolvi(f), f)
        if a.prova:
            for lab, zz in (("foglio trasparente (atteso ~0 dB)", 1e8),
                            ("foglio quasi perfetto (atteso >> 100 dB)", 1e-9)):
                e1, _ = M.ricevuto(M.risolvi(f, zz, zz), f)
                print(f"  verifica - {lab}: SE {20*np.log10(e0/e1):.1f} dB", flush=True)
        for nome, cu in m1d.MATERIALI:
            e1, h1 = M.ricevuto(M.risolvi(f, zs_pannello(f, cu), zs_nastro(f)), f)
            se_e, se_h = 20 * np.log10(e0 / e1), 20 * np.log10(h0 / h1)
            s1d = m1d.se_prova(np.array([f]), cu, "E" if a.sorgente == "dipolo" else "H")[0]
            righe.append((f / 1e6, nome, se_e, se_h, s1d))
            print(f"{f/1e6:5.0f} MHz {nome:10s}: SE_E {se_e:6.1f} dB | SE_H {se_h:6.1f} dB | "
                  f"modello 1D {s1d:6.1f} dB | {time.time()-t1:.0f} s", flush=True)
    with open("EMI3Dp_risultati.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["f_MHz", "materiale", "SE_E_3D_dB", "SE_H_3D_dB", "SE_1D_prova_dB"])
        for r in righe:
            wr.writerow([f"{r[0]:.0f}", r[1]] + [f"{v:.2f}" for v in r[2:]])
    if len(freqs) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for (nome, _), c in zip(m1d.MATERIALI, ("C0", "C1")):
            rr = [r for r in righe if r[1] == nome]
            plt.plot([r[0] for r in rr], [r[2] for r in rr], "o-", color=c, label=f"3D {nome} (E)")
            plt.plot([r[0] for r in rr], [r[3] for r in rr], "s:", color=c, label=f"3D {nome} (H)")
            plt.plot([r[0] for r in rr], [r[4] for r in rr], "--", color=c, alpha=0.6, label=f"1D {nome}")
        plt.axhspan(*m1d.LAB_BAND, color="gray", alpha=0.25, label="misura lab")
        plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]"); plt.grid(True); plt.legend(fontsize=8)
        plt.title(f"Modello 3D portabile ({a.sorgente})"); plt.tight_layout()
        plt.savefig("EMI3Dp_SE.png", dpi=150)
    print(f"Tempo totale {time.time()-t0:.0f} s. Risultati in EMI3Dp_risultati.csv")
