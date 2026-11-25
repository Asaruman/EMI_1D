"""
MODELLO REALISTICO del campo elettromagnetico nella lastra CFRP
(MTC510 plain weave 3K, Vf 50%, 8 ply da 0.225 mm) con maglia di rame al centro,
63-129 MHz. FEM 1D P2 + soluzione esatta ABCD (vedi cfrp_fem_core.py).

1) CAMPI NELLO SPESSORE: |E|, |H| e vettore di Poynting S(z); potenza per ply e
   nella maglia; bilancio R + A + T = 1 verificato in due modi indipendenti
   (integrale di volume sigma|E|^2 e flusso di Poynting alle facce).
2) SCOMPOSIZIONE SE: 1a interfaccia + resto (assorbimento, maglia, multiple).
3) CAMPO VICINO: SE per sorgente elettrica (dipolo) e magnetica (spira) a
   distanza finita, usando l'impedenza d'onda al posto di 377 ohm
   (approssimazione di Schelkunoff); FEM e ABCD confrontati.
4) CATENA DI MISURA: portacampione coassiale tipo ASTM D4935 con resistenza di
   contatto ai bordi, fughe parassite (cavi/connettori) e dinamica dello
   strumento -> SE che lo strumento legge davvero.
5) MONTE CARLO: incertezza su fibra, Vf, tessitura, spessore, maglia, contatto,
   fughe, dinamica, calibrazione -> bande 5-95 %, probabilita' di cadere nella
   banda di laboratorio, classifica dei parametri piu' influenti.

Uso:  python3 cfrp_campo_elettrico.py [--mc 2000] [--no-show]
Requisiti: numpy, matplotlib.  Tenere cfrp_fem_core.py nella stessa cartella.
Output (nella cartella da cui si lancia): campo_E_profilo.png,
  campo_E_potenza_strati.png, campo_E_SE_RA.png, campo_E_risultati.csv,
  campo_vicino_SE.png, campo_vicino_SE.csv, misura_catena.png,
  montecarlo_SE.png, montecarlo_sensibilita.png, montecarlo_SE.csv
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import cfrp_fem_core as core

# ======================= PARAMETRI NOMINALI =======================
PLY_T, N_SIDE, VF = 0.225e-3, 4, 0.50
SIGMA_FIBRA, K_PW = 5.56e4, 1.0          # S/m; fattore di tessitura plain weave
SIGMA_F = K_PW * VF * SIGMA_FIBRA / 2    # ~1.39e4 S/m (meta' delle fibre lungo E)
WIRE_D, MESH_PER_INCH, CRIMP = 0.15e-3, 50, 1.15   # MESH_PER_INCH: DA CONFERMARE
PITCH = 25.4e-3 / MESH_PER_INCH
MESH_STATE = "reale"                     # 'ideale' | 'reale' | numero | None
F0, F1, NPTS = 63e6, 129e6, 67
F_PROFILI = [63e6, 96e6, 129e6]
T_TOT = 2 * N_SIDE * PLY_T
LAB_BAND = (70.0, 80.0)                  # misura di laboratorio [dB]

# ---- campo vicino ----
DISTANZE = [0.02, 0.05, 0.10, 0.30, 1.0]  # m, distanza sorgente-lastra

# ---- catena di misura (portacampione coassiale tipo ASTM D4935) ----
Z0_CELLA = 50.0          # ohm
R_CONTATTO = 3e-3        # ohm, totale bordo interno + esterno  (DA STIMARE)
L_CONTATTO = 0.0         # H, induttanza di contatto (0 = solo resistiva)
PERDITA_DB = 105.0       # livello delle fughe parassite sotto il riferimento
DINAMICA_DB = 110.0      # dinamica dello strumento (VNA / ricevitore) [dB]

# ---- Monte Carlo ----
N_MC, SEED, NF_MC = 2000, 1, 34


def zs(f, rs=None):
    """Impedenza della maglia; rs = Rs effettiva [ohm/quadro] (None = MESH_STATE)."""
    return core.zs_maglia(f, MESH_STATE if rs is None else rs, WIRE_D, PITCH, CRIMP)


def rs_filo():
    return core.RHO_CU * PITCH / (np.pi * WIRE_D**2 / 4)


def se_db(t):
    return -20 * np.log10(np.abs(t))


# ======================= 1) CAMPI NELLO SPESSORE =======================
def sezione_campi(z):
    print("=" * 70 + "\n1) CAMPI E, H E VETTORE DI POYNTING NELLO SPESSORE\n" + "=" * 70)
    fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for f in F_PROFILI:
        zz, E, i0 = core.solve_E(f, z, SIGMA_F, zs(f))
        zp, Ep, Hn, Sn = core.campi_EH(f, z, E)
        R, T, p_ply, p_cu = core.potenze(f, z, E, SIGMA_F, zs(f), N_SIDE, i0)
        A = p_ply.sum() + p_cu
        lab = f"{f/1e6:.0f} MHz"
        ax[0].semilogy(zp * 1e3, np.abs(Ep), label=lab)
        ax[1].semilogy(zp * 1e3, np.abs(Hn), label=lab)
        ax[2].semilogy(zp * 1e3, np.maximum(Sn, 1e-30), label=lab)
        print(f"{lab}: skin depth {core.skin_depth(f, SIGMA_F)*1e3:.3f} mm | "
              f"|E| fronte {abs(E[0]):.2e}, maglia {abs(E[i0]):.2e}, uscita {abs(E[-1]):.2e}")
        print(f"         |H|/H_inc fronte {abs(Hn[0]):.3f} (conduttore: ~2) | "
              f"uscita {abs(Hn[-1]):.2e} | |E/H| fronte {abs(Ep[0]/Hn[0])*core.ETA0:.3f} ohm")
        # bilancio con due metodi indipendenti
        a_poy = Sn[0] - Sn[-1]
        print(f"         Poynting: S_in = {Sn[0]:.4e} (1-R = {1-R:.4e}) | "
              f"S_out = {Sn[-1]:.3e} (T = {T:.3e})")
        print(f"         assorbita: volume {A:.5e} | flusso {a_poy:.5e} | "
              f"scarto rel. {abs(a_poy-A)/A:.1e} | R+A+T-1 = {R+A+T-1:.1e}")
    titoli = ["|E| / |E_inc|", "|H| / |H_inc|", "S_z / S_inc  (Poynting)"]
    for a, t in zip(ax, titoli):
        for k in range(2 * N_SIDE + 1):
            a.axvline((-T_TOT / 2 + k * PLY_T) * 1e3, color="gray", lw=0.4)
        a.axvline(0, color="orange", lw=2)
        a.set_ylabel(t); a.grid(True, which="both", alpha=0.4)
    ax[0].legend(); ax[0].set_title("Campi attraverso lo spessore (linea arancio = maglia Cu)")
    ax[2].set_xlabel("z [mm]  (0 = piano della maglia)")
    fig.tight_layout(); fig.savefig("campo_E_profilo.png", dpi=150)

    f_c = 96e6
    zz, E, i0 = core.solve_E(f_c, z, SIGMA_F, zs(f_c))
    R, T, p_ply, p_cu = core.potenze(f_c, z, E, SIGMA_F, zs(f_c), N_SIDE, i0)
    A = p_ply.sum() + p_cu
    print(f"\nDissipazione per strato @ {f_c/1e6:.0f} MHz (A = {A:.3e} della potenza incidente):")
    labels = [f"ply {k+1}" for k in range(N_SIDE)] + ["Cu"] + \
             [f"ply {k+1}" for k in range(N_SIDE, 2 * N_SIDE)]
    vals = list(p_ply[:N_SIDE]) + [p_cu] + list(p_ply[N_SIDE:])
    for l, v in zip(labels, vals):
        print(f"  {l:6s}: {v:.3e}  ({100*v/A:.2f} % dell'assorbito)")
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, vals, color=["#4c78a8"] * N_SIDE + ["#f58518"] + ["#4c78a8"] * N_SIDE)
    plt.yscale("log"); plt.ylabel("Potenza dissipata / potenza incidente")
    plt.title(f"Dissipazione per strato @ {f_c/1e6:.0f} MHz")
    plt.grid(True, axis="y", which="both", alpha=0.4); plt.tight_layout()
    plt.savefig("campo_E_potenza_strati.png", dpi=150)


# ======================= 2) SCOMPOSIZIONE SE =======================
def sezione_scomposizione(z, freqs):
    print("\n" + "=" * 70 + "\n2) SCOMPOSIZIONE DELLA SE (onda piana)\n" + "=" * 70)
    rows = []
    for f in freqs:
        zz, E, i0 = core.solve_E(f, z, SIGMA_F, zs(f))
        R, T, p_ply, p_cu = core.potenze(f, z, E, SIGMA_F, zs(f), N_SIDE, i0)
        se_t = -10 * np.log10(T)
        se_r = -10 * np.log10(1 - R)
        rows.append((f / 1e6, se_t, se_r, se_t - se_r, R + T + p_ply.sum() + p_cu))
    rows = np.array(rows)
    print(f"SE totale = {rows[:,1].min():.1f} .. {rows[:,1].max():.1f} dB")
    print(f"  1a interfaccia = {rows[:,2].min():.1f} .. {rows[:,2].max():.1f} dB")
    print(f"  resto (assorb.+interno) = {rows[:,3].min():.1f} .. {rows[:,3].max():.1f} dB")
    print(f"  max |R+A+T-1| = {np.abs(rows[:,4]-1).max():.1e}")
    plt.figure(figsize=(8, 5))
    plt.plot(rows[:, 0], rows[:, 1], lw=2, label="SE totale")
    plt.plot(rows[:, 0], rows[:, 2], "--", label="1a interfaccia (riflessione)")
    plt.plot(rows[:, 0], rows[:, 3], ":", lw=2, label="resto (assorb. + riflessioni interne + maglia)")
    plt.xlabel("Frequenza [MHz]"); plt.ylabel("dB"); plt.title("Scomposizione della SE")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("campo_E_SE_RA.png", dpi=150)
    with open("campo_E_risultati.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_MHz", "SE_tot_dB", "SE_1a_interfaccia_dB", "SE_resto_dB", "R+A+T"])
        for r in rows:
            w.writerow([f"{r[0]:.2f}"] + [f"{v:.6g}" for v in r[1:]])
    return rows


# ======================= 3) CAMPO VICINO =======================
def sezione_campo_vicino(z, freqs):
    print("\n" + "=" * 70 + "\n3) CAMPO VICINO: sorgente elettrica (dipolo) e magnetica (spira)\n" + "=" * 70)
    zsf = zs(freqs)
    M = core.abcd_laminato(freqs, SIGMA_F, zsf, PLY_T, N_SIDE)
    se_pw = se_db(core.t_da_abcd(M))
    # verifica FEM vs ABCD con impedenza di riferimento complessa
    err = 0.0
    for src in ("elettrica", "magnetica"):
        for f in (63e6, 129e6):
            zr = core.impedenza_onda(f, DISTANZE[0], src)
            a = core.se_fem(f, z, SIGMA_F, zs(f), zr)
            b = core.se_esatta(f, SIGMA_F, zs(f), PLY_T, N_SIDE, zr)
            err = max(err, abs(a - b))
    print(f"Verifica FEM vs ABCD con z_ref complessa (r = {DISTANZE[0]*100:.0f} cm): "
          f"scarto max {err:.1e} dB  {'OK' if err < 1e-2 else 'ERRORE'}")
    print(f"Onda piana: SE {se_pw.min():.1f} .. {se_pw.max():.1f} dB")
    print(f"{'r [cm]':>7s} {'|Zw| E @96MHz':>15s} {'SE elettrica':>18s} "
          f"{'|Zw| H @96MHz':>15s} {'SE magnetica':>18s}")
    res = {"onda piana": se_pw}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(freqs / 1e6, se_pw, "k", lw=2.5, label="onda piana")
    cm = plt.get_cmap("viridis")
    for i, r in enumerate(DISTANZE):
        c = cm(i / max(1, len(DISTANZE) - 1))
        line = f"{r*100:7.0f}"
        for src, ls in (("elettrica", "--"), ("magnetica", "-")):
            se = se_db(core.t_da_abcd(M, core.impedenza_onda(freqs, r, src)))
            res[f"{src} r={r*100:g}cm"] = se
            zw = abs(core.impedenza_onda(96e6, r, src))
            line += f" {zw:12.1f} ohm {se.min():8.1f}..{se.max():5.1f} dB"
            ax[0].plot(freqs / 1e6, se, ls, color=c,
                       label=f"{'E' if src == 'elettrica' else 'H'}  r = {r*100:g} cm")
        print(line)
    ax[0].axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    ax[0].set_xlabel("Frequenza [MHz]"); ax[0].set_ylabel("SE [dB]")
    ax[0].set_title("SE in campo vicino (tratteggio: sorgente E; continuo: sorgente H)")
    ax[0].grid(True); ax[0].legend(fontsize=7, ncol=2)

    rr = np.logspace(-2, 1, 120)
    for f, c in ((63e6, "C0"), (128e6, "C3")):
        Mf = core.abcd_laminato(f, SIGMA_F, zs(f), PLY_T, N_SIDE)
        for src, ls in (("elettrica", "--"), ("magnetica", "-")):
            se = [se_db(core.t_da_abcd(Mf, core.impedenza_onda(f, r, src)))[0] for r in rr]
            ax[1].semilogx(rr * 100, se, ls, color=c, label=f"{f/1e6:.0f} MHz, sorgente {src}")
        ax[1].axhline(se_db(core.t_da_abcd(Mf))[0], color=c, lw=0.8, ls=":")
        ax[1].axvline(core.C0 / f / (2 * np.pi) * 100, color=c, lw=0.8, alpha=0.6)
    ax[1].set_xlabel("distanza sorgente - lastra [cm]"); ax[1].set_ylabel("SE [dB]")
    ax[1].set_title("SE vs distanza (punteggiato: onda piana; verticale: lambda/2pi)")
    ax[1].grid(True, which="both", alpha=0.4); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("campo_vicino_SE.png", dpi=150)
    print("Nota: la sorgente magnetica vicina (es. bobina RF) e' il caso peggiore: "
          "l'impedenza d'onda bassa riduce la perdita per riflessione.")
    with open("campo_vicino_SE.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_MHz"] + [k.replace(" ", "_") + "_dB" for k in res])
        for i, f in enumerate(freqs):
            w.writerow([f"{f/1e6:.2f}"] + [f"{res[k][i]:.3f}" for k in res])


# ======================= 4) CATENA DI MISURA =======================
def r_contatto_per(target_db, f=96e6):
    M = core.abcd_laminato(f, SIGMA_F, zs(f), PLY_T, N_SIDE)
    lo, hi = 1e-7, 10.0
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        if se_db(core.t_portacampione(M, f, Z0_CELLA, mid, L_CONTATTO))[0] > target_db:
            lo = mid
        else:
            hi = mid
    return mid


def sezione_misura(freqs):
    print("\n" + "=" * 70 + "\n4) CATENA DI MISURA (portacampione coassiale tipo ASTM D4935)\n" + "=" * 70)
    M = core.abcd_laminato(freqs, SIGMA_F, zs(freqs), PLY_T, N_SIDE)
    se_mat = se_db(core.t_da_abcd(M))
    se0 = se_db(core.t_portacampione(M, freqs, Z0_CELLA, 0.0, 0.0))
    print(f"Materiale (onda piana)            : {se_mat.min():6.1f} .. {se_mat.max():6.1f} dB")
    print(f"Nella cella, contatto perfetto    : {se0.min():6.1f} .. {se0.max():6.1f} dB "
          f"(scarto {np.abs(se0-se_mat).max():.1e}: OK)")
    plt.figure(figsize=(9, 5.5))
    plt.plot(freqs / 1e6, se_mat, "k", lw=2.5, label="materiale (onda piana)")
    for rc in (1e-3, R_CONTATTO, 1e-2, 3e-2):
        se = se_db(core.t_portacampione(M, freqs, Z0_CELLA, rc, L_CONTATTO))
        plt.plot(freqs / 1e6, se, "--", label=f"contatto {rc*1e3:g} mOhm")
        print(f"Contatto {rc*1e3:6.1f} mOhm             : {se.min():6.1f} .. {se.max():6.1f} dB")
    t = core.t_portacampione(M, freqs, Z0_CELLA, R_CONTATTO, L_CONTATTO)
    fase = 2 * np.pi * freqs * 5e-9                    # fuga con ritardo 5 ns
    se_m = core.se_misurata(t, PERDITA_DB, fase, DINAMICA_DB)
    print(f"Catena completa (contatto {R_CONTATTO*1e3:g} mOhm, fughe -{PERDITA_DB:g} dB, "
          f"dinamica {DINAMICA_DB:g} dB): {se_m.min():6.1f} .. {se_m.max():6.1f} dB")
    lim = -10 * np.log10(10 ** (-PERDITA_DB / 10) + 10 ** (-DINAMICA_DB / 10))
    print(f"Massima SE leggibile dalla catena ~ {lim:.0f} dB: oltre, lo strumento "
          f"misura le fughe e il rumore, non il campione.")
    plt.plot(freqs / 1e6, se_m, lw=2, color="C3", label="catena completa (contatto+fughe+rumore)")
    plt.axhline(lim, color="C3", ls=":", lw=1, label="limite della catena")
    plt.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]")
    plt.title("Dal materiale alla lettura dello strumento")
    plt.grid(True); plt.legend(fontsize=8, loc="center right"); plt.tight_layout()
    plt.savefig("misura_catena.png", dpi=150)
    print("Resistenza di contatto che riproduce la misura di laboratorio @96 MHz:")
    for tgt in (70, 75, 80):
        print(f"  SE letta {tgt} dB  <->  R_contatto ~ {r_contatto_per(tgt)*1e3:.2f} mOhm")
    print("  (vernice all'argento / superficie carteggiata: ~0.1-1 mOhm; strato di resina: >10 mOhm)")


# ======================= 5) MONTE CARLO =======================
def campiona(rng, n):
    """Distribuzioni dei parametri incerti (modificare qui)."""
    p = {
        "sigma fibra": rng.normal(SIGMA_FIBRA, 0.08 * SIGMA_FIBRA, n),
        "Vf": np.clip(rng.normal(VF, 0.02, n), 0.4, 0.6),
        "k tessitura": rng.triangular(0.8, 1.0, 1.3, n),
        "spessore ply": rng.normal(PLY_T, 0.010e-3, n),
        "crimp maglia": rng.uniform(1.05, 1.40, n),
        "giunzioni maglia": rng.lognormal(0.0, np.log(2.0), n),      # moltiplica Rs
        "R contatto": R_CONTATTO * rng.lognormal(0.0, np.log(3.0), n),
        "livello fughe": rng.uniform(95, 115, n),
        "ritardo fughe": rng.uniform(0, 20e-9, n),
        "dinamica": rng.normal(DINAMICA_DB, 3.0, n),
        "errore calibr.": rng.normal(0.0, 0.3, n),
    }
    return p


def ranghi(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def sezione_montecarlo(n):
    print("\n" + "=" * 70 + f"\n5) MONTE CARLO ({n} campioni)\n" + "=" * 70)
    rng = np.random.default_rng(SEED)
    freqs = np.linspace(F0, F1, NF_MC)
    p = campiona(rng, n)
    se_mat = np.empty((n, NF_MC)); se_mis = np.empty((n, NF_MC))
    for i in range(n):
        sig = p["k tessitura"][i] * p["Vf"][i] * p["sigma fibra"][i] / 2
        rs = p["crimp maglia"][i] * p["giunzioni maglia"][i] * rs_filo()
        M = core.abcd_laminato(freqs, sig, zs(freqs, rs), p["spessore ply"][i], N_SIDE)
        se_mat[i] = se_db(core.t_da_abcd(M))
        t = core.t_portacampione(M, freqs, Z0_CELLA, p["R contatto"][i], L_CONTATTO)
        fase = 2 * np.pi * freqs * p["ritardo fughe"][i]
        se_mis[i] = core.se_misurata(t, p["livello fughe"][i], fase, p["dinamica"][i]) \
            + p["errore calibr."][i]
    q = lambda a: np.percentile(a, [5, 50, 95], axis=0)
    qm, qs = q(se_mat), q(se_mis)
    k96 = int(np.argmin(np.abs(freqs - 96e6)))
    print(f"SE materiale @96 MHz : mediana {qm[1,k96]:.1f} dB, 90% in [{qm[0,k96]:.1f}, {qm[2,k96]:.1f}]")
    print(f"SE misurata  @96 MHz : mediana {qs[1,k96]:.1f} dB, 90% in [{qs[0,k96]:.1f}, {qs[2,k96]:.1f}]")
    inb = np.mean((se_mis[:, k96] >= LAB_BAND[0]) & (se_mis[:, k96] <= LAB_BAND[1]))
    print(f"Probabilita' che la lettura cada in {LAB_BAND[0]:.0f}-{LAB_BAND[1]:.0f} dB @96 MHz: {100*inb:.0f} %")

    nomi = list(p)
    cor_m = [np.corrcoef(ranghi(p[k]), ranghi(se_mat[:, k96]))[0, 1] for k in nomi]
    cor_s = [np.corrcoef(ranghi(p[k]), ranghi(se_mis[:, k96]))[0, 1] for k in nomi]
    print("\nSensibilita' (correlazione di rango di Spearman con la SE @96 MHz):")
    print(f"  {'parametro':18s} {'materiale':>10s} {'misurata':>10s}")
    for k, a, b in sorted(zip(nomi, cor_m, cor_s), key=lambda x: -abs(x[2])):
        print(f"  {k:18s} {a:10.2f} {b:10.2f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for qq, c, lab in ((qm, "C0", "materiale (onda piana)"), (qs, "C3", "lettura strumento")):
        ax.fill_between(freqs / 1e6, qq[0], qq[2], color=c, alpha=0.25, label=f"{lab}: 5-95 %")
        ax.plot(freqs / 1e6, qq[1], color=c, lw=2, label=f"{lab}: mediana")
    ax.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    ax.set_xlabel("Frequenza [MHz]"); ax.set_ylabel("SE [dB]")
    ax.set_title(f"Incertezza Monte Carlo ({n} campioni)")
    ax.grid(True); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("montecarlo_SE.png", dpi=150)

    o = np.argsort(np.abs(cor_s))
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(nomi))
    ax.barh(y - 0.2, np.array(cor_m)[o], 0.4, label="SE materiale")
    ax.barh(y + 0.2, np.array(cor_s)[o], 0.4, label="SE letta")
    ax.set_yticks(y); ax.set_yticklabels(np.array(nomi)[o])
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("correlazione di rango con SE @96 MHz")
    ax.set_title("Quali parametri contano"); ax.grid(True, axis="x", alpha=0.4)
    ax.legend(); fig.tight_layout(); fig.savefig("montecarlo_sensibilita.png", dpi=150)

    with open("montecarlo_SE.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_MHz", "mat_p5", "mat_p50", "mat_p95", "mis_p5", "mis_p50", "mis_p95"])
        for i, f in enumerate(freqs):
            w.writerow([f"{f/1e6:.2f}"] + [f"{v:.3f}" for v in
                        (qm[0, i], qm[1, i], qm[2, i], qs[0, i], qs[1, i], qs[2, i])])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=N_MC, help="campioni Monte Carlo (0 = salta)")
    ap.add_argument("--no-show", action="store_true", help="non aprire le finestre dei grafici")
    args = ap.parse_args()

    z, nply = core.mesh_automatica(PLY_T, N_SIDE, F1, SIGMA_F, n_per_delta=4)
    print(f"sigma = {SIGMA_F:.2e} S/m | maglia '{MESH_STATE}' | "
          f"mesh P2: {nply} el/ply, {2*(len(z)-1)+1} incognite\n")
    freqs = np.linspace(F0, F1, NPTS)
    sezione_campi(z)
    sezione_scomposizione(z, freqs)
    sezione_campo_vicino(z, freqs)
    sezione_misura(freqs)
    if args.mc > 0:
        sezione_montecarlo(args.mc)
    if not args.no_show:
        plt.show()
