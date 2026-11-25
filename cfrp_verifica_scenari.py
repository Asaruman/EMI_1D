"""
VERIFICA DEL MODELLO + SCENARI (ottimistico -> molto pessimistico)
CFRP MTC510 plain weave 3K (Vf 50%, ply 0.225 mm), 8 ply, maglia filo 0.15 mm.
Usa il solutore P2 di cfrp_fem_core.py.

PARTE A - verifiche indipendenti: formula chiusa, Schelkunoff, foglio da solo,
          limiti fisici, bilancio energetico, ordine di convergenza.
PARTE B - scenari realistici e matrice (sigma laminato x stato della maglia)
          confrontati con la misura di laboratorio (70-80 dB).
Uso: python cfrp_verifica_scenari.py    Requisiti: numpy, matplotlib
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import cfrp_fem_core as core
from cfrp_fem_core import ETA0, MU0, EPS0

N_SIDE, PLY_T, VF = 4, 0.225e-3, 0.50
WIRE_D, MESH_PER_INCH, CRIMP = 0.15e-3, 50, 1.15
PITCH = 25.4e-3 / MESH_PER_INCH
FREQS = np.linspace(63e6, 129e6, 67)
LAB_BAND = (70.0, 80.0)


def zs(f, stato):
    return core.zs_maglia(f, stato, WIRE_D, PITCH, CRIMP)


def mesh(n=3, ply=PLY_T):
    return core.mesh_uniforme(ply, N_SIDE, n)


def slab_closed_form(f, sigma, d):
    w = 2 * np.pi * f
    y = sigma + 1j * w * EPS0
    g = np.sqrt(1j * w * MU0 * y)
    zm = np.sqrt(1j * w * MU0 / y)
    return 1 / (np.cosh(g * d) + 0.5 * (zm / ETA0 + ETA0 / zm) * np.sinh(g * d))


def schelkunoff(f, sigma, d):
    w = 2 * np.pi * f
    delta = np.sqrt(2 / (w * MU0 * sigma))
    zm = (1 + 1j) / (sigma * delta)
    R = 20 * np.log10(ETA0 / (4 * abs(zm)))
    A = 8.686 * d / delta
    B = 20 * np.log10(abs(1 - np.exp(-2 * d / delta) * np.exp(-2j * d / delta)))
    return R + A + B


def run_checks():
    print("=" * 66 + "\nPARTE A - VERIFICHE INDIPENDENTI (solutore P2)\n" + "=" * 66)
    ok, sigma, d = True, 1.4e4, 2 * N_SIDE * PLY_T

    print("A1) Lastra omogenea 1.8 mm, sigma = 1.4e4 S/m")
    for f in (63e6, 96e6, 129e6):
        fem = core.se_fem(f, mesh(3), sigma, None)
        cf = -20 * np.log10(abs(slab_closed_form(f, sigma, d)))
        sk = schelkunoff(f, sigma, d)
        good = abs(fem - cf) < 0.01
        ok &= good
        print(f"   {f/1e6:5.0f} MHz: FEM {fem:7.3f} | formula chiusa {cf:7.3f} | "
              f"Schelkunoff {sk:7.3f} dB   {'OK' if good else 'ERRORE'}")

    print("A2) Foglio conduttivo da solo: FEM vs 20log|1+eta0/(2Zs)|")
    for z_s in (1.0, 0.1 + 0.05j, 0.01):
        fem = core.se_fem(96e6, mesh(1), 0.0, z_s)
        th = 20 * np.log10(abs(1 + ETA0 / (2 * z_s)))
        good = abs(fem - th) < 0.01
        ok &= good
        print(f"   Zs={z_s}: FEM {fem:7.3f} | teoria {th:7.3f} dB   {'OK' if good else 'ERRORE'}")

    s0 = core.se_fem(96e6, mesh(2), 0.0, None)
    d_thin = 2 * N_SIDE * 1e-9
    s1 = core.se_fem(96e6, mesh(1, 1e-9), sigma, None)
    th = 20 * np.log10(abs(1 + ETA0 * sigma * d_thin / 2))
    good = abs(s0) < 1e-3 and abs(s1 - th) < 0.01
    ok &= good
    print(f"A3) Limiti: sigma=0 -> {s0:.1e} dB ; film sottile -> FEM {s1:.4f} | "
          f"teoria {th:.4f} dB   {'OK' if good else 'ERRORE'}")

    f = 96e6
    z = mesh(3)
    z_s = zs(f, "reale")
    zz, E, i0 = core.solve_E(f, z, sigma, z_s)
    R, T, p_ply, p_cu = core.potenze(f, z, E, sigma, z_s, N_SIDE, i0)
    bal = R + T + p_ply.sum() + p_cu - 1
    good = abs(bal) < 1e-9
    ok &= good
    print(f"A4) Energia: R+A+T-1 = {bal:.2e}   {'OK' if good else 'ERRORE'}")

    ref = -20 * np.log10(abs(slab_closed_form(100e6, sigma, d)))
    errs = [abs(core.se_fem(100e6, mesh(n), sigma, None) - ref) for n in (1, 2, 4, 8)]
    rate = np.log2(errs[1] / errs[2])
    good = errs[-1] < 1e-3 and 3.5 < rate < 4.5
    ok &= good
    print(f"A5) Convergenza P2: errori {['%.1e' % e for e in errs]}, ordine ~{rate:.2f} "
          f"(atteso ~4)   {'OK' if good else 'ERRORE'}")
    print("-> TUTTE LE VERIFICHE SUPERATE" if ok else "-> ATTENZIONE: verifiche fallite")
    print("   (verificano il CODICE, non l'adeguatezza dei dati di ingresso)\n")


def sigma_plain_weave(sigma_fibra, k=1.0, vf=VF):
    """sigma = k * vf * sigma_fibra / 2  (meta' delle fibre lungo E)."""
    return k * vf * sigma_fibra / 2


SCENARI = [
    dict(nome="A Ottimistico", sigma=sigma_plain_weave(6.0e4, 1.3), mesh="ideale",
         nota="fibra 6e4 S/m, k=1.3; maglia ideale"),
    dict(nome="B Nominale", sigma=sigma_plain_weave(5.56e4, 1.0), mesh="reale",
         nota="fibra 5.56e4 S/m; maglia con ondulazione e reattanza"),
    dict(nome="C Pessimistico", sigma=sigma_plain_weave(4.0e4, 0.9), mesh=0.05,
         nota="fibra 4e4 S/m, k=0.9; maglia a contatti parziali (50 mOhm/quadro)"),
    dict(nome="D Molto pess.", sigma=sigma_plain_weave(3.0e4, 0.8), mesh=None,
         nota="fibra 3e4 S/m, k=0.8; maglia isolata"),
]


def sigma_equivalente(target_db, f=96e6):
    lo, hi = 1e2, 1e6
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if core.se_fem(f, mesh(3), mid, None) > target_db:
            hi = mid
        else:
            lo = mid
    return mid


def run_scenari():
    print("=" * 66 + "\nPARTE B - SCENARI (confronto con misura di laboratorio 70-80 dB)\n" + "=" * 66)
    z = mesh(3)
    res = {}
    for s in SCENARI:
        se = np.array([core.se_fem(f, z, s["sigma"], zs(f, s["mesh"])) for f in FREQS])
        res[s["nome"]] = se
        print(f"{s['nome']:15s} sigma={s['sigma']:.2e} -> SE {se.min():6.1f} .. {se.max():6.1f} dB")
        print(f"                {s['nota']}")

    print("\n" + "=" * 66 + "\nMATRICE: sigma laminato x stato maglia (SE @96 MHz)\n" + "=" * 66)
    sig = {"alta": sigma_plain_weave(6.0e4, 1.3), "nominale": sigma_plain_weave(5.56e4, 1.0),
           "bassa": sigma_plain_weave(3.0e4, 0.8)}
    stati = [("maglia ideale", "ideale"), ("maglia reale", "reale"),
             ("contatto parziale", 0.05), ("maglia assente", None)]
    print(f"{'sigma laminato':22s}" + "".join(f"{n:>20s}" for n, _ in stati))
    for k, v in sig.items():
        row = [core.se_fem(96e6, z, v, zs(96e6, m)) for _, m in stati]
        print(f"{k:9s} {v:9.2e} S/m " + "".join(f"{r:17.1f} dB" for r in row))

    ud = VF * 5.56e4
    print(f"\nMisura di laboratorio: {LAB_BAND[0]:.0f}-{LAB_BAND[1]:.0f} dB")
    for tgt in (70, 75, 80):
        se_ = sigma_equivalente(tgt)
        print(f"  {tgt} dB senza maglia -> sigma_eq = {se_:.0f} S/m = {100*se_/ud:.0f}% del "
              f"valore UD ideale ({ud:.0f} S/m); un plain weave 'sano' ne conserva ~50%")

    with open("scenari_SE.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_MHz"] + list(res.keys()))
        for i, f in enumerate(FREQS):
            w.writerow([f"{f/1e6:.2f}"] + [f"{res[k][i]:.3f}" for k in res])
    plt.figure(figsize=(9, 5.5))
    for k, se in res.items():
        plt.plot(FREQS / 1e6, se, lw=2, label=k)
    plt.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab 70-80 dB")
    plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]")
    plt.title("Scenari del modello vs misura di laboratorio")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("scenari_SE.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_checks()
    run_scenari()
