"""
CFRP EMI - SCRIPT FINALE UNICO (autonomo: non importa gli altri file)
=====================================================================
Lastra MTC510-C200-PW-HS-3K (plain weave 3K, Vf 50 %), 8 ply x 0.225 mm,
CON e SENZA maglia di rame al centro. Banda 63-129 MHz (MRI 1.5 T / 3 T).

Per ciascun materiale calcola l'efficacia di schermatura (SE) per:
  - componente ELETTRICA : sorgente elettrica vicina (dipolo, es. antenna biconica)
  - componente MAGNETICA : sorgente magnetica vicina (spira, es. bobina RF)
  - combinazione E+H     : onda piana (campo lontano, Z = 377 ohm), in cui E e H
                           sono legati e la SE e' la stessa per entrambi
e in due situazioni:
  A) MATERIALE   : lastra infinita, solo fisica del laminato (+ maglia)
  B) PROVA LAB   : replica della prova a finestra della foto: pannello nella
                   parete in rame fissato con nastro conduttivo, antenna a
                   distanza D, apertura vuota come riferimento, fughe e rumore.

Metodi (tutti verificati tra loro all'avvio):
  - FEM 1D a elementi quadratici P2 (mesh dalla skin depth, bordi assorbenti esatti)
  - soluzione esatta a matrici ABCD (linee di trasmissione), anche con impedenza
    d'onda complessa (campo vicino, approssimazione di Schelkunoff)
  - maglia Cu = foglio con Zs = Rs*crimp + j w L_griglia
  - giunto pannello-parete = nastro conduttivo: adesivo su rame + capacita'
    attraverso la resina superficiale del pannello (+ contatto fibre se carteggiato)
  - campi E, H e Poynting nello spessore, bilancio R + A + T = 1
  - Monte Carlo opzionale sui parametri incerti

Uso:   python3 cfrp_EMI_finale.py            (mostra i grafici)
       python3 cfrp_EMI_finale.py --no-show  (salva solo i file)
       python3 cfrp_EMI_finale.py --mc 2000  (aggiunge il Monte Carlo)
Requisiti: numpy, matplotlib.
Output: finale_SE_frequenza.png, finale_SE_distanza.png, finale_campi.png,
        finale_montecarlo.png (con --mc), finale_SE.csv
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

# ============================ PARAMETRI ============================
# --- laminato ---
PLY_T = 0.225e-3          # m, spessore di un ply
N_SIDE = 4                # ply per lato della maglia (8 totali)
VF = 0.50                 # frazione volumetrica di fibra
SIGMA_FIBRA = 5.56e4      # S/m (fibra tipo Grafil 34-700)
K_PW = 1.0                # fattore di tessitura plain weave (0.8-1.3)
# --- maglia di rame ---
WIRE_D = 0.15e-3          # m, diametro del filo
MESH_PER_INCH = 50        # fili per pollice  (DA CONFERMARE)
CRIMP = 1.15              # ondulazione del filo
RS_MOLT = 1.0             # moltiplicatore di Rs (giunzioni imperfette tra fili)
# --- sorgenti ---
D_SORGENTE = 0.30         # m, distanza sorgente - pannello (antenna della prova)
DISTANZE = [0.02, 0.05, 0.10, 0.30, 1.0]   # m, per la tabella in campo vicino
# --- prova a finestra (foto) ---
LATO_E = 0.55             # m, lato del pannello parallelo a E   (confermato: 55 x 25 cm)
LATO_H = 0.25             # m, altro lato                        (confermato)
L_NASTRO = 0.0125         # m, nastro incollato sul pannello      (DA CONFERMARE)
RHO_ADESIVO = 3e-6        # ohm*m^2, adesivo conduttivo (scheda tipica)
T_RESINA = 25e-6          # m, resina superficiale sotto il nastro
EPS_R_RESINA = 3.5
RHO_FIBRE = None          # ohm*m^2 se bordo carteggiato, None = non carteggiato
DINAMICA_DB = 100.0       # dB, dinamica ricevitore + antenne
FUGHE_DB = 110.0          # dB, fughe della stanza / cavi
# --- banda e confronto ---
F0, F1, NPTS = 63e6, 129e6, 67
F_TAB = [64e6, 96e6, 128e6]
LAB_BAND = (70.0, 80.0)   # misura di laboratorio [dB]

# ============================ COSTANTI ============================
MU0 = 4e-7 * np.pi
EPS0 = 8.854187817e-12
C0 = 1 / np.sqrt(MU0 * EPS0)
ETA0 = MU0 * C0
RHO_CU = 1.68e-8


def sigma_laminato(sigma_fibra=SIGMA_FIBRA, vf=VF, k=K_PW):
    """Plain weave: meta' delle fibre parallele a E."""
    return k * vf * sigma_fibra / 2


def skin_depth(f, sigma):
    return 1 / np.sqrt(np.pi * f * MU0 * sigma)


# ============================ MAGLIA ============================
def zs_maglia(f, con_rame=True, crimp=CRIMP, rs_molt=RS_MOLT):
    """Impedenza di foglio della maglia [ohm/quadro]; None se senza rame."""
    if not con_rame:
        return None
    pitch = 25.4e-3 / MESH_PER_INCH
    rs0 = RHO_CU * pitch / (np.pi * WIRE_D**2 / 4)
    arg = pitch / (np.pi * WIRE_D)
    lg = MU0 * pitch / (2 * np.pi) * np.log(arg) if arg > 1 else 0.0
    return crimp * rs_molt * rs0 + 1j * 2 * np.pi * np.asarray(f, float) * lg


# ============================ SORGENTI ============================
def impedenza_onda(f, r, sorgente):
    """E/H a distanza r: 'E' dipolo elettrico, 'H' spira, 'EM' onda piana."""
    f = np.asarray(f, float)
    if sorgente == "EM":
        return ETA0 * np.ones_like(f) + 0j
    u = 1 / (1j * 2 * np.pi * f / C0 * r)
    ze = ETA0 * (1 + u + u**2) / (1 + u)
    return ze if sorgente == "E" else ETA0**2 / ze


# ============================ SOLUZIONE ESATTA (ABCD) ============================
def abcd(f, sigma, zs, ply_t=PLY_T):
    """Matrici (N,2,2): N_SIDE ply | foglio zs | N_SIDE ply."""
    f = np.atleast_1d(np.asarray(f, float))
    w = 2 * np.pi * f
    y = sigma + 1j * w * EPS0
    g = np.sqrt(1j * w * MU0 * y)
    zc = np.sqrt(1j * w * MU0 / y)
    gd = g * N_SIDE * ply_t
    half = np.empty((len(f), 2, 2), complex)
    half[:, 0, 0] = half[:, 1, 1] = np.cosh(gd)
    half[:, 0, 1] = zc * np.sinh(gd)
    half[:, 1, 0] = np.sinh(gd) / zc
    if zs is None:
        return half @ half
    sh = np.zeros_like(half)
    sh[:, 0, 0] = sh[:, 1, 1] = 1
    sh[:, 1, 0] = 1 / np.broadcast_to(np.asarray(zs, complex), f.shape)
    return half @ sh @ half


def t_lastra(M, z_ref, z_add=0.0):
    """Trasmissione E_t/E_i (= H_t/H_i) con impedenza z_ref ai due lati.
    z_add [ohm/quadro] = giunto ai bordi in serie al ramo trasversale (rete a T)."""
    A, B, C, D = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    if np.all(np.asarray(z_add) == 0):
        return 2 / (A + B / z_ref + C * z_ref + D)
    za, zb = (A - 1) / C, (D - 1) / C
    zsh = 1 / C + z_add
    A2, B2, C2, D2 = 1 + za / zsh, za + zb + za * zb / zsh, 1 / zsh, 1 + zb / zsh
    return 2 / (A2 + B2 / z_ref + C2 * z_ref + D2)


def db(t):
    return -20 * np.log10(np.abs(t))


# ============================ FEM 1D P2 ============================
_K2 = np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], float) / 3
_M2 = np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], float) / 30


def mesh(f_max, sigma, n_per_delta=4.0, ply_t=PLY_T):
    el = max(1, int(np.ceil(ply_t / (skin_depth(f_max, sigma) / n_per_delta))))
    t = 2 * N_SIDE * ply_t
    return np.linspace(-t / 2, t / 2, 2 * N_SIDE * el + 1)


def fem(f, z, sigma, zs, z_ref=ETA0):
    """E(z) nei nodi P2 con incidente unitario. Ritorna (E, i_maglia)."""
    w = 2 * np.pi * f
    g2 = 1j * w * MU0 * (sigma + 1j * w * EPS0)
    h = np.diff(z)
    ne = len(h)
    Ke = _K2[None] / h[:, None, None] + g2 * h[:, None, None] * _M2[None]
    conn = 2 * np.arange(ne)[:, None] + np.arange(3)[None, :]
    n = 2 * ne + 1
    A = np.zeros((n, n), complex)
    np.add.at(A, (np.repeat(conn[:, :, None], 3, 2), np.repeat(conn[:, None, :], 3, 1)), Ke)
    kb = 1j * w * MU0 / z_ref
    A[0, 0] += kb
    A[-1, -1] += kb
    b = np.zeros(n, complex)
    b[0] = 2 * kb
    i0 = 2 * int(np.argmin(np.abs(z)))
    if zs is not None:
        A[i0, i0] += 1j * w * MU0 / zs
    return np.linalg.solve(A, b), i0


def campi(f, z, E, n_sub=8):
    """E, H (normalizzato ETA0*H) e Poynting S (normalizzato) dentro gli elementi."""
    ne = len(z) - 1
    h = np.diff(z)
    xi = np.linspace(0, 1, n_sub + 1)
    N = np.stack([(1 - xi) * (1 - 2 * xi), 4 * xi * (1 - xi), xi * (2 * xi - 1)], 1)
    dN = np.stack([4 * xi - 3, 4 - 8 * xi, 4 * xi - 1], 1)
    Ee = E[2 * np.arange(ne)[:, None] + np.arange(3)[None, :]]
    Ep = Ee @ N.T
    Hn = -(Ee @ dN.T) / h[:, None] / (1j * 2 * np.pi * f * MU0) * ETA0
    zp = z[:-1, None] + xi[None, :] * h[:, None]
    return zp.ravel(), Ep.ravel(), Hn.ravel(), np.real(Ep * np.conj(Hn)).ravel()


def potenze(z, E, sigma, zs, i0):
    """R, T, potenza assorbita nel laminato e nella maglia (frazioni dell'incidente)."""
    ne = len(z) - 1
    h = np.diff(z)
    xi = np.array([0.5 - np.sqrt(0.6) / 2, 0.5, 0.5 + np.sqrt(0.6) / 2])
    wg = np.array([5, 8, 5]) / 18
    N = np.stack([(1 - xi) * (1 - 2 * xi), 4 * xi * (1 - xi), xi * (2 * xi - 1)], 1)
    Ee = E[2 * np.arange(ne)[:, None] + np.arange(3)[None, :]]
    a_lam = (sigma * (np.abs(Ee @ N.T) ** 2 * wg).sum(1) * h * ETA0).sum()
    a_cu = 0.0 if zs is None else abs(E[i0]) ** 2 * np.real(1 / zs) * ETA0
    return abs(E[0] - 1) ** 2, abs(E[-1]) ** 2, a_lam, a_cu


# ============================ PROVA A FINESTRA ============================
def z_giunto_nastro(f, l_nastro=L_NASTRO, rho_ad=RHO_ADESIVO, t_res=T_RESINA,
                    eps_r=EPS_R_RESINA, rho_fib=RHO_FIBRE):
    """Nastro conduttivo pannello -> parete in rame, per metro di bordo [ohm*m]."""
    w = 2 * np.pi * np.asarray(f, float)
    y = 1j * w * eps_r * EPS0 * l_nastro / t_res
    if rho_fib is not None:
        y = y + l_nastro / (rho_ad + rho_fib)
    return rho_ad / l_nastro + 1 / y


def perdita_apertura_db(f, lato=max(LATO_E, LATO_H)):
    lam = C0 / np.asarray(f, float)
    return np.maximum(0.0, 20 * np.log10(lam / (2 * lato)))


def lettura_db(t, f, fughe=FUGHE_DB, dinamica=DINAMICA_DB, ritardo=5e-9):
    din = dinamica - perdita_apertura_db(f)
    s = t + 10 ** (-fughe / 20) * np.exp(1j * 2 * np.pi * np.asarray(f) * ritardo)
    return -10 * np.log10(np.abs(s) ** 2 + 10 ** (-din / 10))


# ============================ CALCOLO PRINCIPALE ============================
MATERIALI = [("con rame", True), ("senza rame", False)]
SORGENTI = [("E", "elettrica"), ("H", "magnetica"), ("EM", "onda piana E+H")]


def se_materiale(f, con_rame, sorg, d=D_SORGENTE, sigma=None, **kw):
    sigma = sigma_laminato() if sigma is None else sigma
    M = abcd(f, sigma, zs_maglia(f, con_rame, **kw))
    return db(t_lastra(M, impedenza_onda(f, d, sorg)))


def se_prova(f, con_rame, sorg, d=D_SORGENTE, sigma=None, giunto=None, ply_t=PLY_T,
             fughe=FUGHE_DB, dinamica=DINAMICA_DB, mkw=None):
    sigma = sigma_laminato() if sigma is None else sigma
    M = abcd(f, sigma, zs_maglia(f, con_rame, **(mkw or {})), ply_t)
    zg = z_giunto_nastro(f, **(giunto or {}))
    t = t_lastra(M, impedenza_onda(f, d, sorg), 2 * zg / LATO_E)
    return lettura_db(t, f, fughe, dinamica)


def verifiche():
    print("=" * 74 + "\nVERIFICHE (FEM P2 contro soluzione esatta ABCD)\n" + "=" * 74)
    sig = sigma_laminato()
    z = mesh(F1, sig)
    err, bal = 0.0, 0.0
    for nome, cu in MATERIALI:
        for s, _ in SORGENTI:
            for f in (63e6, 129e6):
                zr = impedenza_onda(f, 0.05, s)
                E, i0 = fem(f, z, sig, zs_maglia(f, cu), zr)
                err = max(err, abs(db(E[-1]) - se_materiale(f, cu, s, 0.05)[0]))
        E, i0 = fem(96e6, z, sig, zs_maglia(96e6, cu))
        R, T, al, ac = potenze(z, E, sig, zs_maglia(96e6, cu), i0)
        bal = max(bal, abs(R + T + al + ac - 1))
    ok = err < 1e-2 and bal < 1e-9
    print(f"Scarto max FEM-ABCD (E, H, EM; con/senza rame): {err:.1e} dB")
    print(f"Bilancio energetico max |R+A+T-1|: {bal:.1e}")
    print("-> VERIFICHE SUPERATE\n" if ok else "-> ATTENZIONE: verifiche fallite\n")


def tabella(freqs):
    print("=" * 74)
    print(f"SE [dB] - sorgente a {D_SORGENTE*100:.0f} cm (E, H) e onda piana (EM)")
    print("=" * 74)
    intest = "".join(f"{f/1e6:>9.0f} MHz" for f in F_TAB)
    righe = []
    for blocco, fun in (("A) MATERIALE", se_materiale), ("B) PROVA LAB (finestra + nastro)", se_prova)):
        print(f"\n{blocco}\n{'materiale':12s} {'componente':16s}{intest}")
        for nome, cu in MATERIALI:
            for s, ds in SORGENTI:
                v = fun(np.array(F_TAB), cu, s)
                print(f"{nome:12s} {ds:16s}" + "".join(f"{x:13.1f}" for x in v))
                righe.append((blocco[:1], nome, ds, fun(freqs, cu, s)))
    print(f"\nMisura di laboratorio: {LAB_BAND[0]:.0f}-{LAB_BAND[1]:.0f} dB (con rame)")
    print("Nota: in onda piana (EM) E e H sono legati da 377 ohm e la SE e' unica;")
    print("in campo vicino la sorgente elettrica e' piu' schermata, la magnetica meno.")
    with open("finale_SE.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_MHz"] + [f"{b}_{n.replace(' ', '_')}_{d.replace(' ', '_')}_dB"
                                for b, n, d, _ in righe])
        for i, f in enumerate(freqs):
            w.writerow([f"{f/1e6:.2f}"] + [f"{r[3][i]:.3f}" for r in righe])

    print("\nSE del MATERIALE in campo vicino (media sulla banda):")
    print(f"{'r [cm]':>7s}" + "".join(f"{n+' '+s:>16s}" for n, _ in MATERIALI for s in ("E", "H")))
    for r in DISTANZE:
        print(f"{r*100:7.0f}" + "".join(f"{se_materiale(freqs, cu, s, r).mean():16.1f}"
                                        for _, cu in MATERIALI for s in ("E", "H")))


def grafico_frequenza(freqs):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    col = {"E": "C0", "H": "C3", "EM": "k"}
    for a, (nome, cu) in zip(ax, MATERIALI):
        for s, ds in SORGENTI:
            a.plot(freqs / 1e6, se_materiale(freqs, cu, s), color=col[s], lw=2,
                   label=f"materiale - {ds}")
            a.plot(freqs / 1e6, se_prova(freqs, cu, s), "--", color=col[s], lw=1.5,
                   label=f"prova lab - {ds}")
        a.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab (con rame)")
        a.set_title(f"CFRP {nome}  (sorgenti E/H a {D_SORGENTE*100:.0f} cm)")
        a.set_xlabel("Frequenza [MHz]"); a.grid(True)
    ax[0].set_ylabel("SE [dB]")
    ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig("finale_SE_frequenza.png", dpi=150, bbox_inches="tight")


def grafico_distanza():
    rr = np.logspace(-2, 1, 100)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for a, f in zip(ax, (64e6, 128e6)):
        for (nome, cu), ls in zip(MATERIALI, ("-", "--")):
            for s, c in (("E", "C0"), ("H", "C3")):
                a.semilogx(rr * 100, [se_materiale(f, cu, s, r)[0] for r in rr], ls,
                           color=c, label=f"{nome} - sorgente {s}")
            a.axhline(se_materiale(f, cu, "EM")[0], ls=ls, color="k", lw=1,
                      label=f"{nome} - onda piana")
        a.axvline(C0 / f / (2 * np.pi) * 100, color="gray", lw=1)
        a.set_title(f"{f/1e6:.0f} MHz (linea grigia: lambda/2pi)")
        a.set_xlabel("distanza sorgente - lastra [cm]"); a.grid(True, which="both", alpha=0.4)
    ax[0].set_ylabel("SE materiale [dB]"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("finale_SE_distanza.png", dpi=150)


def grafico_campi():
    f = 96e6
    sig = sigma_laminato()
    z = mesh(F1, sig)
    t = 2 * N_SIDE * PLY_T
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    print(f"\nCampi nello spessore @ {f/1e6:.0f} MHz (onda piana):")
    for (nome, cu), c in zip(MATERIALI, ("C1", "C0")):
        zs = zs_maglia(f, cu)
        E, i0 = fem(f, z, sig, zs)
        zp, Ep, Hn, Sn = campi(f, z, E)
        R, T, al, ac = potenze(z, E, sig, zs, i0)
        ax[0].semilogy(zp * 1e3, np.abs(Ep), color=c, label=nome)
        ax[1].semilogy(zp * 1e3, np.abs(Hn), color=c, label=nome)
        ax[2].semilogy(zp * 1e3, np.maximum(Sn, 1e-30), color=c, label=nome)
        print(f"  {nome:10s}: |E| fronte {abs(E[0]):.2e}, uscita {abs(E[-1]):.2e} | "
              f"|H| fronte {abs(Hn[0]):.2f} | R {R:.6f} A {al+ac:.2e} (maglia {ac:.1e}) "
              f"T {T:.1e} | R+A+T-1 {R+T+al+ac-1:.0e}")
    for a, lab in zip(ax, ["|E| / |E_inc|", "|H| / |H_inc|", "S / S_inc (Poynting)"]):
        for k in range(2 * N_SIDE + 1):
            a.axvline((-t / 2 + k * PLY_T) * 1e3, color="gray", lw=0.4)
        a.axvline(0, color="orange", lw=2)
        a.set_ylabel(lab); a.grid(True, which="both", alpha=0.4)
    ax[0].legend(); ax[0].set_title(f"Campi nello spessore @ {f/1e6:.0f} MHz (arancio: piano della maglia)")
    ax[2].set_xlabel("z [mm]")
    fig.tight_layout(); fig.savefig("finale_campi.png", dpi=150)


def montecarlo(n, seed=1):
    rng = np.random.default_rng(seed)
    f = np.linspace(F0, F1, 23)
    k96 = int(np.argmin(np.abs(f - 96e6)))
    p = dict(sf=rng.normal(SIGMA_FIBRA, 0.08 * SIGMA_FIBRA, n),
             vf=np.clip(rng.normal(VF, 0.02, n), 0.4, 0.6),
             k=rng.triangular(0.8, 1.0, 1.3, n), ply=rng.normal(PLY_T, 1e-5, n),
             crimp=rng.uniform(1.05, 1.4, n), rsm=rng.lognormal(0, np.log(2), n),
             d=rng.uniform(0.2, 0.5, n), tres=T_RESINA * rng.lognormal(0, np.log(2), n),
             lnas=rng.uniform(0.006, 0.025, n), rad=RHO_ADESIVO * rng.lognormal(0, np.log(3), n),
             eps=rng.uniform(3.0, 4.5, n), din=rng.normal(DINAMICA_DB, 3, n),
             fug=rng.uniform(100, 120, n), cal=rng.normal(0, 1.0, n))
    out = {}
    for nome, cu in MATERIALI:
        for s, _ in SORGENTI:
            mat, lab = np.empty((n, len(f))), np.empty((n, len(f)))
            for i in range(n):
                sig = sigma_laminato(p["sf"][i], p["vf"][i], p["k"][i])
                mkw = dict(crimp=p["crimp"][i], rs_molt=p["rsm"][i])
                M = abcd(f, sig, zs_maglia(f, cu, **mkw), p["ply"][i])
                mat[i] = db(t_lastra(M, impedenza_onda(f, p["d"][i], s)))
                g = dict(l_nastro=p["lnas"][i], rho_ad=p["rad"][i], t_res=p["tres"][i], eps_r=p["eps"][i])
                lab[i] = se_prova(f, cu, s, p["d"][i], sig, g, p["ply"][i], p["fug"][i],
                                  p["din"][i], mkw) + p["cal"][i]
            out[(nome, s)] = (np.percentile(mat, [5, 50, 95], 0), np.percentile(lab, [5, 50, 95], 0))
    print("\n" + "=" * 74 + f"\nMONTE CARLO ({n} campioni) - SE @96 MHz: mediana [5 % - 95 %]\n" + "=" * 74)
    print(f"{'materiale':12s} {'comp.':6s} {'MATERIALE':>24s} {'PROVA LAB':>24s}")
    for (nome, s), (qm, ql) in out.items():
        print(f"{nome:12s} {s:6s} {qm[1,k96]:8.1f} [{qm[0,k96]:5.1f}-{qm[2,k96]:5.1f}]"
              f"   {ql[1,k96]:8.1f} [{ql[0,k96]:5.1f}-{ql[2,k96]:5.1f}]")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    col = {"E": "C0", "H": "C3", "EM": "k"}
    for a, (nome, _) in zip(ax, MATERIALI):
        for s, ds in SORGENTI:
            qm, ql = out[(nome, s)]
            a.fill_between(f / 1e6, qm[0], qm[2], color=col[s], alpha=0.15)
            a.plot(f / 1e6, qm[1], color=col[s], lw=2, label=f"materiale - {ds}")
            a.fill_between(f / 1e6, ql[0], ql[2], color=col[s], alpha=0.08, hatch="//")
            a.plot(f / 1e6, ql[1], "--", color=col[s], lw=1.5, label=f"prova lab - {ds}")
        a.axhspan(*LAB_BAND, color="gray", alpha=0.25)
        a.set_title(f"CFRP {nome}: mediana e banda 5-95 %"); a.set_xlabel("Frequenza [MHz]"); a.grid(True)
    ax[0].set_ylabel("SE [dB]")
    ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig("finale_montecarlo.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=0, help="campioni Monte Carlo (0 = salta)")
    ap.add_argument("--no-show", action="store_true", help="non aprire le finestre")
    a = ap.parse_args()
    sig = sigma_laminato()
    print(f"sigma laminato = {sig:.3e} S/m | skin depth {skin_depth(F0, sig)*1e3:.2f}-"
          f"{skin_depth(F1, sig)*1e3:.2f} mm | spessore {2*N_SIDE*PLY_T*1e3:.2f} mm")
    print(f"maglia: filo {WIRE_D*1e3:.2f} mm, {MESH_PER_INCH}/pollice | prova: pannello "
          f"{LATO_E}x{LATO_H} m, sorgente a {D_SORGENTE*100:.0f} cm, nastro {L_NASTRO*1e3:.1f} mm, "
          f"resina {T_RESINA*1e6:.0f} um\n")
    freqs = np.linspace(F0, F1, NPTS)
    verifiche()
    tabella(freqs)
    grafico_frequenza(freqs)
    grafico_distanza()
    grafico_campi()
    if a.mc > 0:
        montecarlo(a.mc)
    if not a.no_show:
        plt.show()
