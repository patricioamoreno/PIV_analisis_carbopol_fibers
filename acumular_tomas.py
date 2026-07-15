"""
acumular_tomas.py
=================
Modo "acumular varias tomas". Carga varios pares (PIV .npz + fibras CSV),
los combina en una sola tabla por zona y corre el analisis global sobre el
conjunto. Esto multiplica el numero de observaciones: con 1 toma tienes ~7
zonas (7 puntos), con 6 tomas tienes ~40 puntos -> los p-valores empiezan a
ser informativos y el analisis deja de ser solo exploratorio.

Idea clave
----------
Cada (toma, zona) se trata como una observacion independiente. Una misma zona
medida en tomas distintas aporta varios puntos, porque su fluido y la
orientacion/dispersion de sus fibras varian de toma a toma. La columna 'toma'
permite, ademas, control por toma si luego quieres modelos mixtos.

Uso
---
    from acumular_tomas import cargar_varias_tomas, analisis_acumulado

    pares = [
        ("caches_zonas/m74.npz", "fibras/m74-...-ptv.csv"),
        ("caches_zonas/m75.npz", "fibras/m75-...-ptv.csv"),
        ...
    ]
    tabla, c1, c2, c4 = analisis_acumulado(pares, etapas_json=None)

O por descubrimiento automatico (empareja por el codigo mXX del nombre):
    pares = emparejar_por_codigo("caches_zonas", "fibras")
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from win10toast import ToastNotifier

from carga_real import (cargar_todo, PREDICTORES, ETAPAS, _codigo_toma)
from analisis_global import (tabla_por_zona, capa1_global, capa2_global,
                             capa4_global, RESPUESTAS)

DEF_PIV = "cache_zonas"
DEF_FIBRAS = "fibras_ultimo_frame"
# ----------------------------------------------------------------------
# Emparejado automatico PIV <-> fibras por codigo de toma (mXX)
# ----------------------------------------------------------------------


def emparejar_por_codigo(dir_piv, dir_fibras):
    """
    Busca .npz en dir_piv y .csv en dir_fibras y los empareja por su codigo
    mNN. Devuelve lista de tuplas (ruta_npz, ruta_csv) para las que hay match.
    Avisa de los que quedan sin pareja.
    """
    npzs = {(_codigo_toma(f)): f for f in glob.glob(os.path.join(dir_piv, "*.npz"))}
    csvs = {(_codigo_toma(f)): f for f in glob.glob(os.path.join(dir_fibras, "*.csv"))
            if not os.path.basename(f).startswith("_")}
    comunes = sorted(set(npzs) & set(csvs) - {None})
    pares = [(npzs[c], csvs[c]) for c in comunes]

    solo_piv = sorted(set(npzs) - set(csvs) - {None})
    solo_fib = sorted(set(csvs) - set(npzs) - {None})
    if solo_piv:
        print(f"[aviso] PIV sin fibras: {solo_piv}")
    if solo_fib:
        print(f"[aviso] fibras sin PIV: {solo_fib}")
    print(f"Emparejadas {len(pares)} tomas: {comunes}")
    return pares

def _meta_toma(nombre):
    """Extrae (reologia, fibras) del nombre de archivo.
    Ej: 'm74-toma-1-n-0750-car-02-ptv' -> ('car-02', 750).
    Devuelve (None, None) si no encuentra el patron."""
    import re
    car = re.search(r"car-?(\d+)", os.path.basename(str(nombre)).lower())
    fib = re.search(r"n-?(\d+)", os.path.basename(str(nombre)).lower())
    reologia = f"car-{car.group(1)}" if car else None
    fibras = int(fib.group(1)) if fib else None
    return reologia, fibras

# ----------------------------------------------------------------------
# Carga acumulada
# ----------------------------------------------------------------------
def cargar_varias_tomas(pares, etapas_json=None, min_fibras=5, verbose=True):
    """
    pares: lista de (ruta_npz_piv, ruta_csv_fibras).
    Devuelve:
      tabla_acum : una fila por (toma, zona) con fluido + orientacion +
                   dispersion (igual que tabla_por_zona pero con columna 'toma').
      diags      : lista de diccionarios de diagnostico por toma.
    """
    tablas, diags = [], []
    for piv, csv in pares:
        cod = _codigo_toma(piv) or os.path.basename(piv)
        if verbose:
            print(f"\n>>> Toma {cod}")
        try:
            df, diag = cargar_todo(piv, csv, etapas_json, verbose=verbose)
        except Exception as e:
            print(f"[error] toma {cod} omitida: {e}")
            continue
        t = tabla_por_zona(df, min_fibras=min_fibras)
        t.insert(0, "toma", cod)
        reo, fib = _meta_toma(csv)
        t.insert(1, "reologia", reo)
        t.insert(2, "fibras", fib)
        tablas.append(t)
        diag["toma"] = cod
        diags.append(diag)

    if not tablas:
        raise RuntimeError("Ninguna toma se cargo correctamente.")
    tabla_acum = pd.concat(tablas, ignore_index=True)
    if verbose:
        n_obs = tabla_acum["fiable"].sum()
        print(f"\n=== Acumulado: {len(tablas)} tomas, "
              f"{len(tabla_acum)} filas (toma x zona), "
              f"{n_obs} observaciones fiables ===")
    return tabla_acum, diags


# ----------------------------------------------------------------------
# Analisis global sobre el acumulado
# ----------------------------------------------------------------------
def analisis_acumulado(pares, etapas_json=None, min_fibras=5, verbose=True):
    """
    Carga varias tomas y corre Capa 1/2/4 global sobre el conjunto combinado.
    Con N grande (muchas tomas x zonas), estos resultados YA no son solo
    exploratorios: los p-valores son interpretables.

    Devuelve (tabla_acum, c1, c2, c4).
    """
    tabla_acum, diags = cargar_varias_tomas(
        pares, etapas_json, min_fibras, verbose)

    # las capas globales esperan columna 'fiable'; ya viene de tabla_por_zona
    c1 = capa1_global(tabla_acum)
    c2 = capa2_global(tabla_acum)
    c4 = capa4_global(c1)

    if verbose:
        n = tabla_acum["fiable"].sum()
        print(f"\nN efectivo (observaciones fiables) = {n}")
        print("\n--- Capa 1 (top |rho| por respuesta) ---")
        for resp in RESPUESTAS:
            sub = c1[(c1.respuesta == resp) & c1.rho.notna()]
            if not sub.empty:
                b = sub.iloc[0]
                print(f"  {resp:10s}: {b['predictor']} @ {b['etapa']} "
                      f"rho={b['rho']:+.2f} p={b['p_value']:.3f} (n={b['n_zonas']})")
    return tabla_acum, c1, c2, c4


if __name__ == "__main__":
    # Demo: empareja y analiza lo que haya en las carpetas de ejemplo.
    pares = emparejar_por_codigo(DEF_PIV, DEF_FIBRAS)
    if pares:
        tabla, c1, c2, c4 = analisis_acumulado(pares)
        tabla.to_csv("acum_tabla_zona.csv", index=False)
        c1.to_csv("acum_capa1_global.csv", index=False)
        c2.to_csv("acum_capa2_global.csv", index=False)
        c4.to_csv("acum_capa4_global.csv", index=False)
        print("\nGuardados: acum_tabla_zona.csv, acum_capa1_global.csv, "
              "acum_capa2_global.csv, acum_capa4_global.csv")
    else:
        print("No hay tomas emparejadas para analizar.")
    toaster = ToastNotifier()
    toaster.show_toast("VSCode", "¡Tu código de Python terminó exitosamente!", duration=5)