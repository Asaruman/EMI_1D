"""
Efficacia di schermatura (SE) di una lastra CFRP plain weave (8 ply) con maglia
di rame al centro, 63-129 MHz. FEM 1D, onda piana a incidenza normale.
Materiale: prepreg MTC510-C200-PW-HS-3K-42%RW, Vf = 50%, ply 0.225 mm.

Mesh ottimizzata: elementi quadratici, senza aria, passo da skin depth
(vedi cfrp_fem_core.py). Requisiti: numpy, matplotlib. Uso: python cfrp_shield_fem1d.py
Output: se_fem1d.png, se_sensibilita.png, se_convergence.png, se_results.csv
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import cfrp_fem_core as core

# ---------------- PARAMETRI ----------------
PLY_T, N_SIDE, VF = 0.225e-3, 4, 0.50
SIGMA_FIBRA = 5.56e4                         # S/m (fibra tipo Grafil 34-700)
SIGMA_F = VF * SIGMA_FIBRA / 2               # plain weave in piano ~1.39e4 S/m
SIGMA_RANGE = [6.0e3, SIGMA_F, 1.95e4]       # basso / nominale / alto
WIRE_D = 0.15e-3
MESH_PER_INCH = 50                           # DA CONFERMARE
PITCH = 25.4e-3 / MESH_PER_INCH
CRIMP = 1.15
MESH_STATE = "reale"        # 'ideale' | 'reale' | numero (Rs eff. ohm/quadro) | None
LAB_BAND = (70.0, 80.0)     # misura di laboratorio [dB]
F0, F1, NPTS = 63e6, 129e6, 67


def zs(f, stato=None):
    return core.zs_maglia(f, MESH_STATE if stato == "default" else stato, WIRE_D, PITCH, CRIMP)


if __name__ == "__main__":
    freqs = np.linspace(F0, F1, NPTS)
    z, nply = core.mesh_automatica(PLY_T, N_SIDE, F1, max(SIGMA_RANGE), 2.5)
    print(f"sigma laminato = {SIGMA_F:.3e} S/m | skin depth @100 MHz = "
          f"{core.skin_depth(100e6, SIGMA_F)*1e3:.3f} mm")
    print(f"Maglia: filo {WIRE_D*1e3:.2f} mm, {MESH_PER_INCH}/pollice, stato '{MESH_STATE}'")
    print(f"Mesh: {nply} elementi P2 per ply, {2*(len(z)-1)+1} incognite\n")

    se = np.array([core.se_fem(f, z, SIGMA_F, zs(f, "default")) for f in freqs])
    ex = np.array([core.se_esatta(f, SIGMA_F, zs(f, "default"), PLY_T, N_SIDE) for f in freqs])
    se0 = np.array([core.se_fem(f, z, SIGMA_F, None) for f in freqs])
    print(f"SE con maglia   = {se.min():.1f} .. {se.max():.1f} dB")
    print(f"SE senza maglia = {se0.min():.1f} .. {se0.max():.1f} dB")
    print(f"Contributo maglia = {(se-se0).min():.1f} .. {(se-se0).max():.1f} dB")
    print(f"Scarto FEM vs esatta = {np.abs(se-ex).max():.1e} dB")

    with open("se_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_MHz", "SE_FEM_dB", "SE_esatta_dB", "SE_senza_maglia_dB"])
        for r in zip(freqs / 1e6, se, ex, se0):
            w.writerow([f"{r[0]:.2f}"] + [f"{v:.4f}" for v in r[1:]])

    plt.figure(figsize=(8, 5))
    plt.plot(freqs / 1e6, se, lw=2, label=f"FEM con maglia ({MESH_STATE})")
    plt.plot(freqs / 1e6, ex, "k--", label="Esatta (ABCD)")
    plt.plot(freqs / 1e6, se0, ":", label="FEM senza maglia")
    plt.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]")
    plt.title("SE - MTC510 PW 3K, 8 ply + maglia Cu")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("se_fem1d.png", dpi=150)

    plt.figure(figsize=(8, 5))
    for s in SIGMA_RANGE:
        plt.plot(freqs / 1e6, [core.se_fem(f, z, s, zs(f, "default")) for f in freqs],
                 label=f"sigma = {s:.2e} S/m")
    plt.axhspan(*LAB_BAND, color="gray", alpha=0.25, label="misura lab")
    plt.xlabel("Frequenza [MHz]"); plt.ylabel("SE [dB]")
    plt.title("Sensibilita' alla conducibilita' del laminato")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("se_sensibilita.png", dpi=150)
    for s in SIGMA_RANGE:
        v = [core.se_fem(f, z, s, zs(f, "default")) for f in (63e6, 129e6)]
        print(f"  sigma={s:.2e} S/m -> SE {v[0]:.1f} dB @63 MHz, {v[1]:.1f} dB @129 MHz")

    n_list = [1, 2, 3, 4, 6, 8]
    ref = core.se_esatta(100e6, SIGMA_F, zs(100e6, "default"), PLY_T, N_SIDE)
    errs = [abs(core.se_fem(100e6, core.mesh_uniforme(PLY_T, N_SIDE, n), SIGMA_F,
                            zs(100e6, "default")) - ref) for n in n_list]
    plt.figure(figsize=(6, 4.5))
    plt.loglog(n_list, np.maximum(errs, 1e-12), "o-")
    plt.xlabel("Elementi P2 per ply"); plt.ylabel("Errore SE vs esatta [dB]")
    plt.title("Convergenza di mesh @100 MHz"); plt.grid(True, which="both")
    plt.tight_layout(); plt.savefig("se_convergence.png", dpi=150)
    for n, e in zip(n_list, errs):
        print(f"  {n} elem/ply -> errore {e:.1e} dB")
    plt.show()
