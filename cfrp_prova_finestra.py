"""
REPLICA DELLA PROVA DI LABORATORIO (metodo "a finestra", foto setup_prova.png)
Pannello CFRP (MTC510 PW 3K, 8 ply + maglia Cu) montato con una cornice di legno
in un'apertura della parete schermata; antenna biconica in trasmissione a pochi
decimetri dal pannello; antenna ricevente dall'altro lato. SE = livello con
apertura vuota (riferimento) / livello con pannello.

Catena modellata:
  sorgente biconica vicina -> impedenza d'onda alta (dipolo elettrico, distanza d)
  -> laminato (soluzione esatta ABCD, verificata col FEM)
  -> giunto pannello-parete ai bordi: le correnti indotte devono passare nella
     parete in rame attraverso il nastro conduttivo; sul pannello l'adesivo tocca
     la resina superficiale, non le fibre (giunto capacitivo in serie all'adesivo)
  -> fughe della stanza/cavi + fondo di rumore; la dinamica utile e' ridotta
     dall'attenuazione dell'apertura vuota nella misura di riferimento.

Uso: python3 cfrp_prova_finestra.py [--mc 2000] [--no-show]
Requisiti: numpy, matplotlib; cfrp_fem_core.py e cfrp_campo_elettrico.py nella
stessa cartella. Output: prova_finestra_SE.png, prova_finestra_giunto.png,
prova_finestra_distanza.png, prova_finestra_montecarlo.png, prova_finestra_SE.csv
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import cfrp_fem_core as core
import cfrp_campo_elettrico as ce

# =============== SETUP DELLA PROVA (stimato dalla foto: DA CONFERMARE) ===============
LATO_E = 0.55            # m, lato del pannello parallelo a E (confermato: 55 x 25 cm)
LATO_H = 0.25            # m, altro lato (confermato)
D_ANTENNA = 0.30         # m, distanza centro antenna - pannello
SORGENTE = "elettrica"   # biconica = dipolo elettrico
GIUNTO = "nastro"        # 'nastro' (nastro conduttivo verso la parete in rame: CONFERMATO)
                         # | 'capacitivo' (solo appoggio) | 'resistivo' | 'aperto'
L_NASTRO = 0.0125        # m, larghezza del nastro incollata sul pannello (DA CONFERMARE)
RHO_ADESIVO = 3e-6       # ohm*m^2, contatto dell'adesivo conduttivo (scheda tipica)
RHO_FIBRE = None         # ohm*m^2 se il bordo e' carteggiato (fibre scoperte), None = no
SOVRAPP = 0.03           # m, sovrapposizione pannello-parete per lato
T_RESINA = 25e-6         # m, strato di resina superficiale del laminato
EPS_R = 3.5              # resina epossidica
R_LIN = 0.0              # ohm*m, resistenza di contatto per metro (giunto resistivo)
DINAMICA_DB = 100.0      # dinamica ricevitore + antenne con apertura grande
PERDITA_DB = 110.0       # fughe della stanza / cavi (indipendenti dal pannello)
F0, F1, NPTS = 63e6, 129e6, 67
LAB_BAND = ce.LAB_BAND
N_MC, SEED = 2000, 2


def db(t):
    return -20 * np.log10(np.abs(t))


def materiale(freqs, sigma=ce.SIGMA_F, rs=None, ply=ce.PLY_T):
    return core.abcd_laminato(freqs, sigma, ce.zs(freqs, rs), ply, ce.N_SIDE)


def t_pannello(M, freqs, d=D_ANTENNA, giunto=GIUNTO, t_res=T_RESINA, sovr=SOVRAPP,
               eps=EPS_R, r_lin=R_LIN, l_nastro=L_NASTRO, rho_ad=RHO_ADESIVO,
               rho_fib=RHO_FIBRE):
    zr = core.impedenza_onda(freqs, d, SORGENTE)
    if giunto == "nastro":
        zl = core.z_giunto_nastro(freqs, l_nastro, rho_ad, t_res, eps, rho_fib)
    else:
        zl = core.z_giunto_per_metro(freqs, giunto, r_lin, sovr, t_res, eps)
    return core.t_con_giunto(M, core.z_giunto_equivalente(zl, LATO_E), zr)


def lettura(t, freqs, perdita=PERDITA_DB, dinamica=DINAMICA_DB, ritardo=5e-9):
    din_utile = dinamica - core.perdita_apertura_db(freqs, max(LATO_E, LATO_H))
    return core.se_misurata(t, perdita, 2 * np.pi * freqs * ritardo, din_utile)


def sezione_catena(freqs):
    print("=" * 70 + "\n1) LA PROVA: dal materiale alla lettura\n" + "=" * 70)
    print(f"Pannello {LATO_E:.2f} x {LATO_H:.2f} m, antenna biconica a {D_ANTENNA*100:.0f} cm, "
          f"giunto '{GIUNTO}' (nastro {L_NASTRO*1000:.1f} mm sul pannello, resina "
          f"{T_RESINA*1e6:.0f} um, bordo {'carteggiato' if RHO_FIBRE else 'non carteggiato'})")
    M = materiale(freqs)
    zr = core.impedenza_onda(freqs, D_ANTENNA, SORGENTE)
    righe = {
        "materiale, onda piana": db(core.t_da_abcd(M)),
        f"materiale, bicono a {D_ANTENNA*100:.0f} cm": db(core.t_da_abcd(M, zr)),
        "+ giunto ai bordi (nastro su resina)": db(t_pannello(M, freqs)),
    }
    righe["lettura (fughe + rumore)"] = lettura(t_pannello(M, freqs), freqs)
    for k, v in righe.items():
        print(f"  {k:32s}: {v.min():6.1f} .. {v.max():6.1f} dB")
    ap = db(t_pannello(M, freqs, giunto="aperto"))
    print(f"  {'(pannello isolato, solo legno)':32s}: {ap.min():6.1f} .. {ap.max():6.1f} dB")
    din = DINAMICA_DB - core.perdita_apertura_db(freqs, max(LATO_E, LATO_H))
    print(f"Apertura vuota: attenua {core.perdita_apertura_db(freqs, LATO_E).max():.1f} .. "
          f"{core.perdita_apertura_db(freqs, LATO_E).min():.1f} dB -> dinamica utile "
          f"{din.min():.0f} .. {din.max():.0f} dB")
    plt.figure(figsize=(9, 5.5))
    stili = [("k", "-", 2.5), ("C0", "--", 2), ("C1", "-", 2), ("C3", "-", 2.5)]
    for (k, v), (c, ls, lw) in zip(righe.items(), stili):
        plt.plot(freqs / 1e6, v, ls, color=c, lw=lw, label=k)
    plt.plot(freqs / 1e6, -10 * np.log10(10 ** (-PERDITA_DB / 10) + 10 ** (-din / 10)), ":",
             color="C3", label="limite della catena (fughe + dinamica utile)")
    plt.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]")
    plt.title("Replica della prova a finestra")
    plt.grid(True); plt.legend(fontsize=8, loc="center right"); plt.tight_layout()
    plt.savefig("prova_finestra_SE.png", dpi=150)
    with open("prova_finestra_SE.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_MHz"] + [k.replace(" ", "_") for k in righe])
        for i, f in enumerate(freqs):
            w.writerow([f"{f/1e6:.2f}"] + [f"{righe[k][i]:.3f}" for k in righe])


def sezione_giunto():
    print("\n" + "=" * 70 + "\n2) QUALE GIUNTO CON NASTRO RIPRODUCE LA MISURA (96 MHz)\n" + "=" * 70)
    fr = np.array([63e6, 96e6, 129e6])
    M = materiale(fr)
    tt = np.logspace(-6, -3.3, 80)
    rf = np.logspace(-7, -2, 80)
    se_c = np.array([lettura(t_pannello(M, fr, t_res=t), fr) for t in tt])
    se_r = np.array([lettura(t_pannello(M, fr, rho_fib=r), fr) for r in rf])
    for nome, x, se, u, s in (("resina sotto il nastro", tt, se_c, 1e6, "um"),
                              ("contatto adesivo-fibre", rf, se_r, 1e6, "ohm*mm^2")):
        ok = x[(se[:, 1] >= LAB_BAND[0]) & (se[:, 1] <= LAB_BAND[1])]
        if len(ok):
            print(f"  {nome}: tra {ok.min()*u:.3g} e {ok.max()*u:.3g} {s} -> lettura 70-80 dB")
    print(f"  bordo carteggiato con contatto pari all'adesivo ({RHO_ADESIVO*1e6:.0f} ohm*mm^2): "
          f"{lettura(t_pannello(M, fr, rho_fib=RHO_ADESIVO), fr)[1]:.1f} dB")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, (c, f) in enumerate(zip(("C0", "C2", "C3"), fr)):
        ax[0].semilogx(tt * 1e6, se_c[:, j], color=c, label=f"{f/1e6:.0f} MHz")
        ax[1].semilogx(rf * 1e6, se_r[:, j], color=c, label=f"{f/1e6:.0f} MHz")
    ax[0].axvline(T_RESINA * 1e6, color="k", ls=":", lw=1, label="valore nominale")
    ax[0].set_xlabel("strato di resina sotto il nastro [um]")
    ax[0].set_title(f"Nastro su superficie non carteggiata ({L_NASTRO*1000:.1f} mm sul pannello)")
    ax[1].set_xlabel("resistenza di contatto adesivo - fibre [ohm*mm^2]")
    ax[1].set_title("Nastro su bordo carteggiato (fibre scoperte)")
    for a in ax:
        a.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
        a.set_ylabel("SE letta [dB]"); a.grid(True, which="both", alpha=0.4); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("prova_finestra_giunto.png", dpi=150)


def sezione_distanza():
    print("\n" + "=" * 70 + "\n3) EFFETTO DELLA DISTANZA DELL'ANTENNA\n" + "=" * 70)
    fr = np.array([63e6, 96e6, 129e6])
    M = materiale(fr)
    dd = np.logspace(-1, 0.5, 60)
    se_m = np.array([db(core.t_da_abcd(M, core.impedenza_onda(fr, d, SORGENTE))) for d in dd])
    se_l = np.array([lettura(t_pannello(M, fr, d=d), fr) for d in dd])
    for d in (0.1, 0.3, 1.0, 3.0):
        k = int(np.argmin(np.abs(dd - d)))
        print(f"  d = {dd[k]*100:5.0f} cm: lettura {se_l[k,0]:.1f} / {se_l[k,1]:.1f} / "
              f"{se_l[k,2]:.1f} dB a 63 / 96 / 129 MHz")
    plt.figure(figsize=(9, 5.5))
    for j, (c, f) in enumerate(zip(("C0", "C2", "C3"), fr)):
        plt.semilogx(dd * 100, se_l[:, j], color=c, lw=2, label=f"lettura {f/1e6:.0f} MHz")
        plt.semilogx(dd * 100, se_m[:, j], "--", color=c, lw=1, label=f"solo materiale {f/1e6:.0f} MHz")
    plt.axvline(D_ANTENNA * 100, color="k", ls=":", lw=1, label="distanza della prova")
    plt.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    plt.xlabel("distanza antenna biconica - pannello [cm]"); plt.ylabel("SE [dB]")
    plt.title("Quanto dipende la SE dalla posizione dell'antenna")
    plt.grid(True, which="both", alpha=0.4); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("prova_finestra_distanza.png", dpi=150)


def sezione_montecarlo(n):
    print("\n" + "=" * 70 + f"\n4) MONTE CARLO DELLA PROVA ({n} campioni)\n" + "=" * 70)
    rng = np.random.default_rng(SEED)
    freqs = np.linspace(F0, F1, 34)
    p = ce.campiona(rng, n)
    for k in ("R contatto", "livello fughe", "ritardo fughe", "dinamica", "errore calibr."):
        p.pop(k)
    p["distanza antenna"] = rng.uniform(0.2, 0.5, n)
    p["spessore resina"] = T_RESINA * rng.lognormal(0, np.log(2.0), n)
    p["larghezza nastro"] = rng.uniform(0.006, 0.025, n)
    p["adesivo"] = RHO_ADESIVO * rng.lognormal(0, np.log(3.0), n)
    p["eps_r resina"] = rng.uniform(3.0, 4.5, n)
    p["dinamica"] = rng.normal(DINAMICA_DB, 3, n)
    p["livello fughe"] = rng.uniform(100, 120, n)
    p["errore calibr."] = rng.normal(0, 1.0, n)        # posizionamento antenne
    se = np.empty((n, len(freqs)))
    for i in range(n):
        sig = p["k tessitura"][i] * p["Vf"][i] * p["sigma fibra"][i] / 2
        rs = p["crimp maglia"][i] * p["giunzioni maglia"][i] * ce.rs_filo()
        M = materiale(freqs, sig, rs, p["spessore ply"][i])
        t = t_pannello(M, freqs, d=p["distanza antenna"][i], t_res=p["spessore resina"][i],
                       eps=p["eps_r resina"][i], l_nastro=p["larghezza nastro"][i],
                       rho_ad=p["adesivo"][i])
        se[i] = lettura(t, freqs, p["livello fughe"][i], p["dinamica"][i]) + p["errore calibr."][i]
    q = np.percentile(se, [5, 50, 95], axis=0)
    k = int(np.argmin(np.abs(freqs - 96e6)))
    inb = np.mean((se[:, k] >= LAB_BAND[0]) & (se[:, k] <= LAB_BAND[1]))
    print(f"Lettura @96 MHz: mediana {q[1,k]:.1f} dB, 90% in [{q[0,k]:.1f}, {q[2,k]:.1f}]; "
          f"P(70-80 dB) = {100*inb:.0f} %")
    rk = lambda x: np.argsort(np.argsort(x))
    cor = {nm: np.corrcoef(rk(v), rk(se[:, k]))[0, 1] for nm, v in p.items()}
    print("Parametri piu' influenti (Spearman con la lettura @96 MHz):")
    for nm, c in sorted(cor.items(), key=lambda x: -abs(x[1]))[:6]:
        print(f"  {nm:18s} {c:6.2f}")
    plt.figure(figsize=(9, 5.5))
    plt.fill_between(freqs / 1e6, q[0], q[2], color="C3", alpha=0.25, label="lettura simulata: 5-95 %")
    plt.plot(freqs / 1e6, q[1], color="C3", lw=2, label="lettura simulata: mediana")
    plt.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]")
    plt.title(f"Prova a finestra: incertezza Monte Carlo ({n} campioni)")
    plt.grid(True); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("prova_finestra_montecarlo.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=N_MC)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    freqs = np.linspace(F0, F1, NPTS)
    sezione_catena(freqs)
    sezione_giunto()
    sezione_distanza()
    if a.mc > 0:
        sezione_montecarlo(a.mc)
    if not a.no_show:
        plt.show()
