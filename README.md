# Modello 3D della prova a finestra (`cfrp_EMI_3D.py`)

## Cosa fa
Simula in 3D la prova di laboratorio. Il pannello CFRP da 55 × 25 cm è montato nell'apertura della parete in rame e collegato con nastro conduttivo. La sorgente sta a 30 cm davanti al pannello, il punto di ricezione a 30 cm dietro.
La SE è calcolata come in laboratorio: campo con apertura vuota / campo con pannello.
Il laminato (con o senza maglia Cu) e il nastro usano i valori del modello 1D già verificato (`cfrp_EMI_finale.py`).

## Installazione (una volta sola)
Serve Python 3.10–3.13 (Windows, Linux o macOS). In un terminale:

```
pip install ngsolve numpy matplotlib
```

Su Windows, se `pip` non viene trovato, usa `py -m pip install ngsolve numpy matplotlib`.
Copia nella stessa cartella `cfrp_EMI_3D.py` e `cfrp_EMI_finale.py`.

## Come si lancia
```
python cfrp_EMI_3D.py --prova                 # 2-4 min: verifica + 96 MHz su mesh grossolana
python cfrp_EMI_3D.py                         # 64, 96, 128 MHz, con e senza rame, sorgente dipolo
python cfrp_EMI_3D.py --sorgente spira        # sorgente magnetica (bobina)
python cfrp_EMI_3D.py --fine                  # mesh più fitta (controllo di convergenza)
python cfrp_EMI_3D.py --freq 64 128           # solo alcune frequenze
python cfrp_EMI_3D.py --vtk                   # salva i campi da aprire con ParaView
python cfrp_EMI_3D.py --iterativo             # meno RAM, più lento
python cfrp_EMI_3D.py --ordine 2              # elementi di ordine 2: più precisi, servono 32+ GB RAM
```

La modalità `--prova` stampa due verifiche, che devono dare:
- foglio trasparente: circa 0 dB
- foglio quasi perfetto: oltre 150 dB

## Risorse (misurate su una macchina Linux a 4 core)
| Modalità | Incognite | RAM | Tempo |
|---|---|---|---|
| `--prova` | ~160 000 | ~3 GB | ~2 min |
| standard (1 frequenza) | ~195 000 | ~4 GB | ~1,5 min |
| standard (3 frequenze) | ~195 000 | ~4 GB | ~5 min |
| `--fine` | ~0,6–1 milione | 16–32 GB | 15–40 min |

## Output
- `EMI3D_risultati.csv`: SE_E e SE_H in 3D, più il valore del modello 1D per confronto
- `EMI3D_SE.png`: grafico 3D contro 1D
- `EMI3D_campo_*.vtu` (con `--vtk`)

## Parametri
I parametri sono in cima a `cfrp_EMI_3D.py`: dimensioni dell'apertura, distanze, lunghezza del dipolo e raggio della spira, dominio e PML.
Materiali e nastro si cambiano in `cfrp_EMI_finale.py`.
