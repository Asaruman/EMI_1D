"""
CFRP EMI - MODELLO 3D DELLA PROVA A FINESTRA (FEM, NGSolve)
===========================================================
Replica in 3D la prova della foto: pannello CFRP 55 x 25 cm nell'apertura di
una parete in rame, collegato con nastro conduttivo; sorgente a 30 cm su un
lato, punto di ricezione sull'altro. La SE e' il rapporto tra il campo
ricevuto con l'apertura VUOTA (riferimento) e con il PANNELLO montato,
esattamente come in laboratorio.

Cosa e' 3D (e il modello 1D non vedeva):
  - dimensioni finite di pannello e apertura, diffrazione dai bordi
  - campo vero della sorgente (dipolo = biconica, oppure spira = bobina)
  - distribuzione reale delle correnti nel pannello e attraverso il nastro
Cosa viene dal modello 1D (cfrp_EMI_finale.py, verificato):
  - il laminato (8 ply, con/senza maglia Cu) e' una superficie con impedenza
    di trasferimento Zt = 1/C della sua matrice ABCD (scarto < 0.01 dB rispetto
    alla soluzione esatta della lastra); modulo esatto, fase resa passiva
  - il nastro e' una fascia larga L_NASTRO lungo il bordo dell'apertura con
    impedenza di foglio Z'/L_NASTRO (Z' = impedenza del giunto per metro)
Metodo: Maxwell armonico, elementi di Nedelec (ordine 1, opzione 2), PML sui bordi,
parete in rame = conduttore perfetto, solutore diretto.

Uso (PC fisso):
  python cfrp_EMI_3D.py --prova            mesh grossolana, ~1-2 min, verifica che tutto giri
  python cfrp_EMI_3D.py                    calcolo standard (3 frequenze, con/senza rame, dipolo)
  python cfrp_EMI_3D.py --sorgente spira   sorgente magnetica
  python cfrp_EMI_3D.py --fine             mesh piu' fitta (piu' RAM e tempo)
  python cfrp_EMI_3D.py --vtk              salva anche i campi per ParaView
Requisiti: pip install ngsolve numpy matplotlib ; cfrp_EMI_finale.py nella stessa cartella.
Output: EMI3D_risultati.csv, EMI3D_SE.png, (EMI3D_campo_*.vtu con --vtk)
"""
import argparse
import csv
import time
import numpy as np
import cfrp_EMI_finale as m1d

from netgen.occ import Box, Cylinder, OCCGeometry, Glue, WorkPlane, Axes, Pnt, Vec, X, Y, Z
from ngsolve import (Mesh, HCurl, BilinearForm, LinearForm, GridFunction, CoefficientFunction,
                     curl, dx, ds, x, y, z, sqrt, IfPos, pml, TaskManager, SetNumThreads,
                     Integrate, VTKOutput, Norm, Preconditioner)
from ngsolve.krylovspace import GMRes

# ======================= GEOMETRIA DELLA PROVA =======================
AP_X, AP_Y = 0.55, 0.25     # m, apertura = pannello (55 cm parallelo a E)
L_NASTRO = m1d.L_NASTRO     # m, fascia del nastro lungo il bordo
D_SORG = m1d.D_SORGENTE     # m, sorgente davanti alla parete
D_RIC = 0.30                # m, punto di ricezione dietro la parete (DA CONFERMARE)
L_DIPOLO = 0.50             # m, lunghezza del dipolo equivalente alla biconica
R_SPIRA = 0.10              # m, raggio della spira (sorgente magnetica)
# dominio (mezza larghezza) e PML
DOM_X, DOM_Y, DOM_Z, PML_T = 0.70, 0.50, 0.65, 0.30
FREQS = [64e6, 96e6, 128e6]
SOLUTORE = "diretto"         # 'diretto' (veloce, molta RAM) | 'iterativo' (poca RAM)


def zs_pannello(f, con_rame):
    """Impedenza di trasferimento del laminato dal modello 1D (convenzione e^{jwt})."""
    M = m1d.abcd(np.array([f]), m1d.sigma_laminato(), m1d.zs_maglia(np.array([f]), con_rame))
    zt = 1 / M[0, 1, 0]
    return abs(zt) * (1 + 1j) / np.sqrt(2)          # modulo esatto, fase passiva


def zs_nastro(f):
    """Impedenza di foglio equivalente della fascia di nastro [ohm/quadro]."""
    return complex(m1d.z_giunto_nastro(np.array([f]))[0]) / L_NASTRO


def geometria(maxh, h_ap, sorgente):
    lo_i = Box(Pnt(-DOM_X, -DOM_Y, -DOM_Z), Pnt(DOM_X, DOM_Y, 0))
    hi_i = Box(Pnt(-DOM_X, -DOM_Y, 0), Pnt(DOM_X, DOM_Y, DOM_Z))
    P = PML_T
    lo_o = Box(Pnt(-DOM_X - P, -DOM_Y - P, -DOM_Z - P), Pnt(DOM_X + P, DOM_Y + P, 0)) - lo_i
    hi_o = Box(Pnt(-DOM_X - P, -DOM_Y - P, 0), Pnt(DOM_X + P, DOM_Y + P, DOM_Z + P)) - hi_i
    lo_o.mat("pml"); hi_o.mat("pml"); hi_i.mat("ric"); lo_i.mat("sorg")
    if sorgente == "dipolo":
        ant = Box(Pnt(-L_DIPOLO / 2, -0.01, -D_SORG - 0.01), Pnt(L_DIPOLO / 2, 0.01, -D_SORG + 0.01))
    else:   # spira con asse lungo y (H tangente al pannello)
        c = Pnt(0, -0.01, -D_SORG)
        ant = Cylinder(c, Y, R_SPIRA + 0.01, 0.02) - Cylinder(c, Y, R_SPIRA - 0.01, 0.02)
    ant.mat("antenna"); ant.maxh = 0.02
    lo_i = lo_i - ant
    wp = WorkPlane(Axes((0, 0, 0), n=Z, h=X))
    ap = wp.MoveTo(-AP_X / 2, -AP_Y / 2).Rectangle(AP_X, AP_Y).Face()
    wp2 = WorkPlane(Axes((0, 0, 0), n=Z, h=X))
    inn = wp2.MoveTo(-AP_X / 2 + L_NASTRO, -AP_Y / 2 + L_NASTRO).Rectangle(
        AP_X - 2 * L_NASTRO, AP_Y - 2 * L_NASTRO).Face()
    shape = Glue([lo_i, hi_i, lo_o, hi_o, ant, ap, inn])
    tol = 1e-6
    for fc in shape.faces:
        pmin, pmax = fc.bounding_box
        if abs(pmin.z) < tol and abs(pmax.z) < tol:            # piano della parete
            sx, sy = pmax.x - pmin.x, pmax.y - pmin.y
            if sx < AP_X - 2 * L_NASTRO + tol and sy < AP_Y - 2 * L_NASTRO + tol:
                fc.name = "pannello"; fc.maxh = h_ap
            elif sx < AP_X + tol and sy < AP_Y + tol:
                fc.name = "nastro"; fc.maxh = L_NASTRO / 2
            else:
                fc.name = "parete"
        elif (abs(pmin.x + DOM_X + P) < tol and abs(pmax.x + DOM_X + P) < tol) or \
             (abs(pmin.x - DOM_X - P) < tol and abs(pmax.x - DOM_X - P) < tol) or \
             (abs(pmin.y + DOM_Y + P) < tol and abs(pmax.y + DOM_Y + P) < tol) or \
             (abs(pmin.y - DOM_Y - P) < tol and abs(pmax.y - DOM_Y - P) < tol) or \
             (abs(pmin.z + DOM_Z + P) < tol and abs(pmax.z + DOM_Z + P) < tol) or \
             (abs(pmin.z - DOM_Z - P) < tol and abs(pmax.z - DOM_Z - P) < tol):
            fc.name = "esterno"
    return OCCGeometry(shape).GenerateMesh(maxh=maxh)


def sorgente_cf(tipo):
    if tipo == "dipolo":
        return CoefficientFunction((1, 0, 0))
    rx, rz = x, z + D_SORG
    r = sqrt(rx * rx + rz * rz)
    return CoefficientFunction((-rz / r, 0, rx / r))


def risolvi(mesh, f, zs_p, zs_n, tipo, ordine, vtk=None):
    """Campo E con pannello (zs_p, zs_n) o apertura vuota (zs_p = None)."""
    w = 2 * np.pi * f
    mu0, eps0 = m1d.MU0, m1d.EPS0
    fes = HCurl(mesh, order=ordine, complex=True, dirichlet="parete|esterno")
    u, v = fes.TnT()
    a = BilinearForm(fes, symmetric=True)
    a += 1 / mu0 * curl(u) * curl(v) * dx - w**2 * eps0 * u * v * dx
    # convenzione NGSolve e^{-iwt}: Z -> conj(Z); foglio: -i w (1/Z) E_t . v_t
    if zs_p is not None:
        a += -1j * w / np.conj(zs_p) * u.Trace() * v.Trace() * ds("pannello")
        a += -1j * w / np.conj(zs_n) * u.Trace() * v.Trace() * ds("nastro")
    pre = Preconditioner(a, "bddc") if SOLUTORE == "iterativo" else None
    J = sorgente_cf(tipo)
    l = LinearForm(fes)
    l += 1j * w * J * v * dx("antenna")
    gfu = GridFunction(fes)
    with TaskManager():
        a.Assemble(); l.Assemble()
        if SOLUTORE == "iterativo":
            gfu.vec.data = GMRes(a.mat, l.vec, pre=pre.mat, tol=1e-12, maxsteps=3000, printrates=False)
        else:
            try:
                inv = a.mat.Inverse(fes.FreeDofs(), inverse="pardiso")
            except Exception:
                inv = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
            gfu.vec.data = inv * l.vec
    if vtk:
        VTKOutput(mesh, coefs=[Norm(gfu)], names=["absE"], filename=vtk).Do()
    return gfu, fes.ndof


def campo_ricevuto(mesh, gfu, f):
    """|E| e |H| mediati su 9 punti attorno al ricevitore (riduce minimi locali)."""
    w = 2 * np.pi * f
    E2, H2 = [], []
    ch = curl(gfu)
    for dxp in (-0.05, 0, 0.05):
        for dyp in (-0.05, 0, 0.05):
            p = mesh(dxp, dyp, D_RIC)
            e = np.array(gfu(p)); c = np.array(ch(p))
            E2.append(np.sum(np.abs(e) ** 2)); H2.append(np.sum(np.abs(c / (w * m1d.MU0)) ** 2))
    return np.sqrt(np.mean(E2)), np.sqrt(np.mean(H2))


def se_1d(f, con_rame, tipo):
    s = "E" if tipo == "dipolo" else "H"
    return m1d.se_prova(np.array([f]), con_rame, s)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prova", action="store_true", help="mesh grossolana, verifica veloce")
    ap.add_argument("--fine", action="store_true", help="mesh piu' fitta")
    ap.add_argument("--sorgente", choices=["dipolo", "spira"], default="dipolo")
    ap.add_argument("--ordine", type=int, default=1, help="1 (default) o 2 (molto piu' RAM)")
    ap.add_argument("--freq", type=float, nargs="*", help="frequenze in MHz (default 64 96 128)")
    ap.add_argument("--thread", type=int, default=0, help="0 = tutti i core")
    ap.add_argument("--vtk", action="store_true")
    ap.add_argument("--iterativo", action="store_true", help="solutore iterativo BDDC+GMRes (meno RAM)")
    a = ap.parse_args()
    if a.iterativo:
        SOLUTORE = "iterativo"
    if a.thread:
        SetNumThreads(a.thread)
    maxh, h_ap = (0.20, 0.06) if a.prova else (0.07, 0.02) if a.fine else (0.12, 0.035)
    freqs = [96e6] if a.prova else FREQS
    if a.freq:
        freqs = [v * 1e6 for v in a.freq]
    t0 = time.time()
    mesh = Mesh(geometria(maxh, h_ap, a.sorgente))
    # PML cartesiana fuori dal dominio interno
    mesh.SetPML(pml.Cartesian((-DOM_X, -DOM_Y, -DOM_Z), (DOM_X, DOM_Y, DOM_Z), 1j), "pml")
    print(f"Mesh: {mesh.ne} tetraedri, {time.time()-t0:.0f} s | sorgente: {a.sorgente} | "
          f"pannello {AP_X*100:.0f}x{AP_Y*100:.0f} cm, nastro {L_NASTRO*1e3:.1f} mm")
    righe = []
    for f in freqs:
        t1 = time.time()
        g0, ndof = risolvi(mesh, f, None, None, a.sorgente, a.ordine,
                           vtk=f"EMI3D_campo_vuoto_{f/1e6:.0f}MHz" if a.vtk else None)
        e0, h0 = campo_ricevuto(mesh, g0, f)
        if a.prova:   # verifiche di coerenza: foglio trasparente e foglio quasi perfetto
            for lab, zz in (("foglio trasparente (atteso ~0 dB)", 1e8), ("foglio quasi perfetto (atteso >> 100 dB)", 1e-9)):
                g, _ = risolvi(mesh, f, zz, zz, a.sorgente, a.ordine)
                e1, _ = campo_ricevuto(mesh, g, f)
                print(f"  verifica - {lab}: SE {20*np.log10(e0/e1):.1f} dB", flush=True)
        for nome, cu in m1d.MATERIALI:
            zp, zn = zs_pannello(f, cu), zs_nastro(f)
            g, _ = risolvi(mesh, f, zp, zn, a.sorgente, a.ordine,
                           vtk=f"EMI3D_campo_{nome.replace(' ', '_')}_{f/1e6:.0f}MHz" if a.vtk else None)
            e1, h1 = campo_ricevuto(mesh, g, f)
            se_e, se_h = 20 * np.log10(e0 / e1), 20 * np.log10(h0 / h1)
            righe.append((f / 1e6, nome, se_e, se_h, se_1d(f, cu, a.sorgente)))
            print(f"{f/1e6:5.0f} MHz {nome:10s}: SE_E {se_e:6.1f} dB | SE_H {se_h:6.1f} dB | "
                  f"modello 1D {righe[-1][4]:6.1f} dB | {ndof} incognite, {time.time()-t1:.0f} s")
    with open("EMI3D_risultati.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["f_MHz", "materiale", "SE_E_3D_dB", "SE_H_3D_dB", "SE_1D_prova_dB"])
        for r in righe:
            wr.writerow([f"{r[0]:.0f}", r[1]] + [f"{v:.2f}" for v in r[2:]])
    if len(freqs) > 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for (nome, _), c in zip(m1d.MATERIALI, ("C0", "C1")):
            rr = [r for r in righe if r[1] == nome]
            plt.plot([r[0] for r in rr], [r[2] for r in rr], "o-", color=c, label=f"3D {nome} (E)")
            plt.plot([r[0] for r in rr], [r[3] for r in rr], "s:", color=c, label=f"3D {nome} (H)")
            plt.plot([r[0] for r in rr], [r[4] for r in rr], "--", color=c, alpha=0.6, label=f"1D {nome}")
        plt.axhspan(*m1d.LAB_BAND, color="gray", alpha=0.25, label="misura lab")
        plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]"); plt.grid(True); plt.legend(fontsize=8)
        plt.title(f"Modello 3D della prova ({a.sorgente})"); plt.tight_layout()
        plt.savefig("EMI3D_SE.png", dpi=150)
    print(f"Tempo totale {time.time()-t0:.0f} s. Risultati in EMI3D_risultati.csv")
