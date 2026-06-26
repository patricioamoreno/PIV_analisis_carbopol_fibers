"""
generar_timeline.py
===================
Crea un JSON con UN vector de tiempo de referencia por carbopol (02 y 05).
Ese vector es el eje x fijo de los espectrogramas y es INDEPENDIENTE de la
zona: se extraen los timestamps de la primera toma de cada carbopol.

esp_plug.py interpola cada toma a esta grilla, de modo que el eje x no
depende de las tomas individuales ni de la zona.

Salida: timeline_ref.json
  {
    "02": {"toma": "...", "t_ref": [0.0, ...], "n": N, "t_final": ...},
    "05": {...}
  }

Uso: python generar_timeline.py
"""
import os, re, glob, json
import numpy as np

CACHE_DIR   = "cache_completo"
OUTPUT_JSON = "time/timeline_ref.json"

REOLOGIAS = ["02", "05"]


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


def reo_de(nombre):
    m = re.search(r'car-(\d+)', nombre)
    return m.group(1) if m else None


def main():
    # Todos los cachés, en orden natural; la PRIMERA toma de cada carbopol
    # (sin importar zona) define su t_ref.
    archivos = sorted(glob.glob(os.path.join(CACHE_DIR, "*_completo.npz")),
                      key=lambda p: natural_sort_key(os.path.basename(p)))

    timeline = {}
    for reo in REOLOGIAS:
        for a in archivos:
            base = os.path.basename(a).replace("_completo.npz", "")
            if reo_de(base) != reo:
                continue
            d = np.load(a)
            t = np.asarray(d['tiempos'], dtype=float)
            t = t - t[0]                       # normalizar a 0
            timeline[reo] = {
                "toma": base,
                "t_ref": t.tolist(),
                "n": int(len(t)),
                "t_final": float(t[-1]),
            }
            print(f"  car-{reo}: ref='{base}'  n={len(t)}  t_final={t[-1]:.2f}s")
            break
        else:
            print(f"  ⚠ No se encontró ninguna toma car-{reo}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False)
    print(f"\n💾 Guardado: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()