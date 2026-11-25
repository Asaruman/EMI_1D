"""
NUCLEO CONDIVISO del modello FEM 1D (lastra CFRP + maglia di rame).
Elementi quadratici (P2), mesh uniforme, senza celle d'aria (bordi assorbenti
esatti per onda piana), passo scelto dalla skin depth.
Importato da: cfrp_shield_fem1d.py, cfrp_campo_elettrico.py, cfrp_verifica_scenari.py
(tenere tutti i file nella stessa cartella).  Richiede solo NumPy.
"""
import numpy as np

MU0 = 4e-7 * np.pi
EPS0 = 8.854187817e-12
C0 = 1 / np.sqrt(MU0 * EPS0)
ETA0 = MU0 * C0
RHO_CU = 1.68e-8


def skin_depth(f, sigma):
    return 1 / np.sqrt(np.pi * f * MU0 * sigma)


# ---------------- maglia di rame: impedenza di foglio ----------------
def lg_griglia(wire_d, pitch):
    """Induttanza di griglia per quadro, E parallelo ai fili (stima, spazio libero)."""
    arg = pitch / (np.pi * wire_d)
    return MU0 * pitch / (2 * np.pi) * np.log(arg) if arg > 1 else 0.0


def zs_maglia(f, stato, wire_d, pitch, crimp=1.15):
    """stato: None (assente) | 'ideale' (solo Rs di filo) | 'reale' (crimp + reattanza)
    | numero (Rs effettiva in ohm/quadro, piu' reattanza)."""
    if stato is None:
        return None
    rs0 = RHO_CU * pitch / (np.pi * wire_d**2 / 4)
    if stato == "ideale":
        return rs0 + 0j
    x = 1j * 2 * np.pi * f * lg_griglia(wire_d, pitch)
    return (crimp * rs0 if stato == "reale" else float(stato)) + x


# ---------------- mesh ----------------
def mesh_uniforme(ply_t, n_side, el_per_ply):
    t = 2 * n_side * ply_t
    return np.linspace(-t / 2, t / 2, 2 * n_side * el_per_ply + 1)


def mesh_automatica(ply_t, n_side, f_max, sigma_max, n_per_delta=2.5):
    """h <= delta_min/n_per_delta. P2: 2.5 -> ~2e-3 dB ; 4 -> ~4e-4 dB."""
    delta_min = skin_depth(f_max, sigma_max)
    el = max(1, int(np.ceil(ply_t / (delta_min / n_per_delta))))
    return mesh_uniforme(ply_t, n_side, el), el


# ---------------- solutore P2 ----------------
_K2 = np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]], float) / 3
_M2 = np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], float) / 30


def solve_E(f, z, sigma, zs, z_ref=ETA0):
    """E(z) con incidente = 1 sulla faccia sinistra. Ritorna (zz, E, i_sheet):
    zz = coordinate di tutti i gradi di liberta' (nodi + punti medi).
    z_ref = impedenza d'onda ai due lati (ETA0 = onda piana; complessa per il
    campo vicino, vedi impedenza_onda)."""
    w = 2 * np.pi * f
    k0 = w / C0
    g2 = 1j * w * MU0 * (sigma + 1j * w * EPS0)
    h = np.diff(z)
    ne = len(h)
    Ke = _K2[None] / h[:, None, None] + g2 * h[:, None, None] * _M2[None]
    conn = 2 * np.arange(ne)[:, None] + np.arange(3)[None, :]
    n = 2 * ne + 1
    A = np.zeros((n, n), dtype=complex)
    np.add.at(A, (np.repeat(conn[:, :, None], 3, 2), np.repeat(conn[:, None, :], 3, 1)), Ke)
    b = np.zeros(n, dtype=complex)
    kb = 1j * w * MU0 / z_ref            # = 1j*k0 per z_ref = ETA0
    A[0, 0] += kb
    A[-1, -1] += kb
    b[0] = 2 * kb
    i_sheet = 2 * int(np.argmin(np.abs(z)))
    if zs is not None:
        A[i_sheet, i_sheet] += 1j * w * MU0 / zs
    E = np.linalg.solve(A, b)
    zz = np.empty(n)
    zz[0::2] = z
    zz[1::2] = 0.5 * (z[1:] + z[:-1])
    return zz, E, i_sheet


def se_fem(f, z, sigma, zs, z_ref=ETA0):
    _, E, _ = solve_E(f, z, sigma, zs, z_ref)
    return -20 * np.log10(abs(E[-1]))


def potenze(f, z, E, sigma, zs, n_side, i_sheet):
    """Frazioni della potenza incidente: R, T, assorbita per ply, nella maglia."""
    ne = len(z) - 1
    h = np.diff(z)
    xi = np.array([0.5 - np.sqrt(0.6) / 2, 0.5, 0.5 + np.sqrt(0.6) / 2])
    wg = np.array([5, 8, 5]) / 18
    N = np.stack([(1 - xi) * (1 - 2 * xi), 4 * xi * (1 - xi), xi * (2 * xi - 1)], axis=1)
    Ee = E[2 * np.arange(ne)[:, None] + np.arange(3)[None, :]]
    int_e2 = (np.abs(Ee @ N.T) ** 2 * wg[None, :]).sum(1) * h      # Gauss 3 punti, esatto
    p_el = sigma * int_e2 * ETA0
    p_ply = p_el.reshape(2 * n_side, ne // (2 * n_side)).sum(1)
    p_cu = 0.0 if zs is None else abs(E[i_sheet]) ** 2 * np.real(1 / zs) * ETA0
    return abs(E[0] - 1) ** 2, abs(E[-1]) ** 2, p_ply, p_cu


# ---------------- soluzione esatta (ABCD), vettoriale in frequenza ----------------
def abcd_laminato(f, sigma, zs, ply_t, n_side):
    """Matrici ABCD (N,2,2) del laminato: n_side ply | foglio zs | n_side ply.
    f scalare o array; zs None, scalare o array come f. Impedenze riferite allo
    spazio libero (onda piana)."""
    f = np.atleast_1d(np.asarray(f, float))
    w = 2 * np.pi * f
    y = sigma + 1j * w * EPS0
    g = np.sqrt(1j * w * MU0 * y)
    zc = np.sqrt(1j * w * MU0 / y)
    gd = g * n_side * ply_t
    half = np.empty((len(f), 2, 2), complex)
    half[:, 0, 0] = half[:, 1, 1] = np.cosh(gd)
    half[:, 0, 1] = zc * np.sinh(gd)
    half[:, 1, 0] = np.sinh(gd) / zc
    if zs is None:
        return half @ half
    sheet = np.zeros_like(half)
    sheet[:, 0, 0] = sheet[:, 1, 1] = 1
    sheet[:, 1, 0] = 1 / np.broadcast_to(np.asarray(zs, complex), f.shape)
    return half @ sheet @ half


def t_da_abcd(M, z_ref=ETA0):
    """Trasmissione di campo E_t/E_i con la stessa impedenza z_ref ai due lati."""
    A, B, C, D = M[..., 0, 0], M[..., 0, 1], M[..., 1, 0], M[..., 1, 1]
    return 2 / (A + B / z_ref + C * z_ref + D)


def se_esatta(f, sigma, zs, ply_t, n_side, z_ref=ETA0):
    t = t_da_abcd(abcd_laminato(f, sigma, zs, ply_t, n_side), z_ref)
    se = -20 * np.log10(np.abs(t))
    return float(se[0]) if np.ndim(f) == 0 else se


# ---------------- campo vicino ----------------
def impedenza_onda(f, r, sorgente="piana"):
    """Impedenza d'onda E/H a distanza r [m] da una sorgente elementare
    (dipolo hertziano 'elettrica' o spira 'magnetica'), direzione trasversale
    (theta = 90 gradi). 'piana' -> ETA0. Usata come z_ref (approssimazione di
    Schelkunoff: lastra infinita illuminata con l'impedenza d'onda locale)."""
    f = np.asarray(f, float)
    if sorgente in (None, "piana"):
        return ETA0 * np.ones_like(f) + 0j
    u = 1 / (1j * 2 * np.pi * f / C0 * r)
    ze = ETA0 * (1 + u + u**2) / (1 + u)
    if sorgente == "elettrica":
        return ze
    if sorgente == "magnetica":
        return ETA0**2 / ze
    raise ValueError(sorgente)


# ---------------- campo magnetico e vettore di Poynting ----------------
def campi_EH(f, z, E, n_sub=8):
    """Valuta E e H dentro ogni elemento P2 (n_sub+1 punti per elemento, i nodi
    condivisi sono ripetuti: cosi' si vede il salto di H sulla maglia).
    Ritorna (zp, Ep, Hn, Sn): Hn = ETA0*H e Sn = Re(E H*)*ETA0, normalizzati
    ai valori dell'onda incidente (incidente: Hn = 1, Sn = 1)."""
    w = 2 * np.pi * f
    ne = len(z) - 1
    h = np.diff(z)
    xi = np.linspace(0, 1, n_sub + 1)
    N = np.stack([(1 - xi) * (1 - 2 * xi), 4 * xi * (1 - xi), xi * (2 * xi - 1)], 1)
    dN = np.stack([4 * xi - 3, 4 - 8 * xi, 4 * xi - 1], 1)
    Ee = E[2 * np.arange(ne)[:, None] + np.arange(3)[None, :]]
    Ep = Ee @ N.T
    dE = (Ee @ dN.T) / h[:, None]
    H = -dE / (1j * w * MU0)                    # dEx/dz = -j w mu Hy
    zp = z[:-1, None] + xi[None, :] * h[:, None]
    Hn = H * ETA0
    Sn = np.real(Ep * np.conj(Hn))
    return zp.ravel(), Ep.ravel(), Hn.ravel(), Sn.ravel()


# ---------------- portacampione e catena di misura ----------------
def t_portacampione(M, f, z0=50.0, r_c=0.0, l_c=0.0):
    """Trasmissione S21 del campione montato in una cella TEM coassiale
    (tipo ASTM D4935) con impedenza di contatto r_c + j w l_c [ohm, totale dei
    due bordi] tra campione e conduttori. La cella e' una linea di impedenza
    z0 = ETA0*g con g = ln(b/a)/(2 pi): le impedenze di foglio si scalano di g.
    Il laminato e' scritto come rete a T; il contatto va in serie al ramo
    trasversale (le correnti del campione si richiudono sui conduttori
    attraverso i contatti). Con r_c = l_c = 0 coincide con l'onda piana."""
    g = z0 / ETA0
    A, C, D = M[..., 0, 0], M[..., 1, 0] / g, M[..., 1, 1]
    za, zb = (A - 1) / C, (D - 1) / C
    zsh = 1 / C + r_c + 1j * 2 * np.pi * np.asarray(f) * l_c
    A2, B2, C2, D2 = 1 + za / zsh, za + zb + za * zb / zsh, 1 / zsh, 1 + zb / zsh
    return 2 / (A2 + B2 / z0 + C2 * z0 + D2)


def se_misurata(t_dut, perdita_db=np.inf, fase_perdita=0.0, dinamica_db=np.inf):
    """SE letta dallo strumento: al campo trasmesso si somma in modo coerente
    un accoppiamento parassita (fughe di cavi/connettori/flange, livello
    -perdita_db, fase fase_perdita) e in potenza il fondo di rumore
    (dinamica_db sotto il riferimento)."""
    t = t_dut + 10 ** (-perdita_db / 20) * np.exp(1j * fase_perdita)
    return -10 * np.log10(np.abs(t) ** 2 + 10 ** (-dinamica_db / 10))


# ---------------- pannello montato in un'apertura (prova a finestra) ----------------
def t_con_giunto(M, z_add, z_ref=ETA0):
    """Trasmissione del laminato (ABCD M, riferito allo spazio libero) quando le
    sue correnti devono passare attraverso un giunto ai bordi. z_add [ohm/quadro]
    e' l'impedenza del giunto riportata a impedenza di foglio equivalente (vedi
    z_giunto_equivalente). Il laminato e' scritto come rete a T e z_add va in
    serie al ramo trasversale. z_ref = impedenza d'onda della sorgente."""
    A, C, D = M[..., 0, 0], M[..., 1, 0], M[..., 1, 1]
    za, zb = (A - 1) / C, (D - 1) / C
    zsh = 1 / C + z_add
    A2, B2, C2, D2 = 1 + za / zsh, za + zb + za * zb / zsh, 1 / zsh, 1 + zb / zsh
    return 2 / (A2 + B2 / z_ref + C2 * z_ref + D2)


def z_giunto_per_metro(f, tipo, r_lin=0.0, sovrapp=0.03, t_resina=50e-6, eps_r=3.5):
    """Impedenza del giunto pannello-parete per metro di bordo [ohm*m].
    'resistivo': r_lin [ohm*m] (contatto metallico diretto).
    'capacitivo': sovrapposizione di larghezza sovrapp [m] separata dalla parete
                  metallica da uno strato di resina t_resina [m] (eps_r), piu' r_lin.
    'aperto': nessun collegamento (pannello isolato, es. solo cornice in legno)."""
    w = 2 * np.pi * np.asarray(f, float)
    if tipo == "resistivo":
        return r_lin + 0j * w
    if tipo == "capacitivo":
        c_lin = eps_r * EPS0 * sovrapp / t_resina          # F/m
        return r_lin + 1 / (1j * w * c_lin)
    if tipo == "aperto":
        return 1e9 + 0j * w
    raise ValueError(tipo)


def z_giunto_equivalente(z_lin, lato_lungo_E):
    """La corrente scorre lungo E per una lunghezza lato_lungo_E [m] ed entra/esce
    dai due bordi perpendicolari: caduta media = 2 z_lin K / l -> ohm/quadro."""
    return 2 * z_lin / lato_lungo_E


def perdita_apertura_db(f, lato_max):
    """Attenuazione dell'apertura vuota (misura di riferimento) per lato < lambda/2:
    20 log(lambda / 2L). Riduce la dinamica utile della prova."""
    lam = C0 / np.asarray(f, float)
    return np.maximum(0.0, 20 * np.log10(lam / (2 * lato_max)))


def z_giunto_nastro(f, larghezza=0.0125, rho_adesivo=3e-6, t_resina=25e-6, eps_r=3.5,
                    rho_fibre=None):
    """Giunto con nastro conduttivo (rame + adesivo conduttivo) tra pannello e
    parete in rame, impedenza per metro di bordo [ohm*m].
    larghezza   : nastro incollato sul pannello [m] (sulla parete si assume uguale)
    rho_adesivo : resistenza di contatto dell'adesivo conduttivo [ohm*m^2]
                  (dato tipico da scheda: ~0.005 ohm su 6.45 cm^2 -> 3e-6)
    t_resina    : strato di resina superficiale del laminato sotto il nastro [m];
                  il nastro vede le fibre solo attraverso di esso (capacita')
    rho_fibre   : se il bordo e' carteggiato, resistenza di contatto adesivo-fibre
                  [ohm*m^2] in parallelo alla capacita' (None = non carteggiato)."""
    w = 2 * np.pi * np.asarray(f, float)
    r_parete = rho_adesivo / larghezza                    # adesivo su rame
    y_pann = 1j * w * eps_r * EPS0 * larghezza / t_resina  # attraverso la resina
    if rho_fibre is not None:
        y_pann = y_pann + larghezza / (rho_adesivo + rho_fibre)
    return r_parete + 1 / y_pann
