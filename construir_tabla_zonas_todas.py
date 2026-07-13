"""
construir_tabla_zonas_todas.py
==============================
Recorre TODAS las tomas, cruza el fluido PIV (cache_zonas) con la foto final
de fibras (fibras_ultimo_frame) por zona, y arma la tabla acumulada por
(toma, zona) con orientacion (orden_S), dispersion y los predictores de fluido
por etapa. Es el "enfoque de zona final": cada fibra se asocia al fluido de la
zona donde quedo.

Reemplaza el flujo manual de run_real.py (una toma con --flags) por un barrido
automatico de todas las tomas, al estilo de construir_caches_zonas.py:
  - configuracion arriba
  - recorre las carpetas/archivos solos
  - bandera RECALCULO
  - guarda un CSV acumulado (acum_tabla_zona.csv)

Fuentes de datos:
  - PIV  : cache_zonas/<carpeta>_zonas.npz  (los que hace construir_caches_zonas)
  - Fibras: fibras_ultimo_frame/<...>-ptv.csv  (CSV sueltos, foto final)
  - Etapas: etapas_zonas.json  (t_quasi por toma y zona)

Salida:
  acum_tabla_zona.csv  (1 fila por toma x zona: fluido + orden_S + sigma + meta)

Uso:
    python construir_tabla_zonas_todas.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd

from carga_real import cargar_todo, PREDICTORES, ETAPAS
from analisis_global import tabla_por_zona

# ============================================================
# CONFIGURACION — editar aqui
# ============================================================

CACHE_ZONAS = "cache_zonas"                 # .npz de PIV por toma
FIBRAS_DIR = "fibras_ultimo_frame"          # CSV de fibras (sueltos)
ETAPAS_JSON = "etapas_zonas.json"           # cortes t_quasi por toma/zona

SALIDA_CSV = "acum_tabla_zona.csv"

# True -> reconstruye aunque el CSV ya exista
RECALCULO = True

# minimo de fibras para que orden_S de una zona sea fiable
MIN_FIBRAS = 1

# Filtros opcionales (None = todas)
CAR_OBJETIVO = None      # p.ej. "02"
FIB_OBJETIVO = None      # p.ej. "1500"


# ============================================================
# UTILIDADES (estilo del proyecto)
# ============================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


def _codigo_toma(nombre):
    m = re.search(r"m(\d+)", os.path.basename(str(nombre)).lower())
    return f"m{m.group(1)}" if m else None


def _meta_toma(nombre):
    """Extrae (reologia, fibras) del nombre. 'm74-...-n-0750-car-02' -> (car-02, 750)."""
    car = re.search(r"car-?(\d+)", os.path.basename(str(nombre)).lower())
    fib = re.search(r"n-?(\d+)", os.path.basename(str(nombre)).lower())
    return (f"car-{car.group(1)}" if car else None,
            int(fib.group(1)) if fib else None)


def emparejar_piv_fibras():
    """
    Empareja cada .npz de cache_zonas con su CSV de fibras por codigo mNN.
    Devuelve lista de dicts: {cod, carpeta_piv, npz, csv, reologia, fibras}.
    """
    npzs = {}
    for f in glob.glob(os.path.join(CACHE_ZONAS, "*.npz")):
        cod = _codigo_toma(f)
        if cod:
            npzs[cod] = f
    csvs = {}
    for f in glob.glob(os.path.join(FIBRAS_DIR, "*.csv")):
        if os.path.basename(f).startswith("_"):
            continue
        cod = _codigo_toma(f)
        if cod:
            csvs[cod] = f

    comunes = sorted(set(npzs) & set(csvs), key=natural_sort_key)
    pares = []
    for cod in comunes:
        reo, fib = _meta_toma(csvs[cod])
        if CAR_OBJETIVO and reo != f"car-{CAR_OBJETIVO}":
            continue
        if FIB_OBJETIVO and fib != int(FIB_OBJETIVO):
            continue
        pares.append({"cod": cod, "npz": npzs[cod], "csv": csvs[cod],
                      "reologia": reo, "fibras": fib})

    solo_piv = sorted(set(npzs) - set(csvs))
    solo_fib = sorted(set(csvs) - set(npzs))
    if solo_piv:
        print(f"[aviso] PIV sin fibras: {solo_piv}")
    if solo_fib:
        print(f"[aviso] fibras sin PIV: {solo_fib}")
    return pares


# ============================================================
# PROCESO
# ============================================================

def procesar_todas():
    pares = emparejar_piv_fibras()
    print(f"Tomas emparejadas: {len(pares)}\n")
    if not pares:
        print("[ERROR] No se emparejo ninguna toma.")
        print(f"  Revisa que existan .npz en '{CACHE_ZONAS}/' y CSV en "
              f"'{FIBRAS_DIR}/', y que compartan codigo mNN.")
        return None

    tablas = []
    for i, p in enumerate(pares, 1):
        cod = p["cod"]
        print(f"[{i}/{len(pares)}] {cod}  ({p['reologia']}, {p['fibras']} fibras)")
        try:
            # cargar_todo cruza el PIV de esta toma con sus fibras por zona
            df, diag = cargar_todo(p["npz"], p["csv"], ETAPAS_JSON,
                                   verbose=False)
            t = tabla_por_zona(df, min_fibras=MIN_FIBRAS)
            t.insert(0, "toma", cod)
            t.insert(1, "reologia", p["reologia"])
            t.insert(2, "fibras", p["fibras"])
            tablas.append(t)
            n_fiab = int(t["fiable"].sum()) if "fiable" in t else len(t)
            print(f"        {len(t)} zonas ({n_fiab} fiables)")
        except Exception as e:
            print(f"        [error] {cod} omitida: {e}")

    if not tablas:
        print("\n[ERROR] Ninguna toma se proceso correctamente.")
        return None

    acum = pd.concat(tablas, ignore_index=True)
    acum.to_csv(SALIDA_CSV, index=False)
    n_fiab = int(acum["fiable"].sum()) if "fiable" in acum else len(acum)
    print(f"\n{'='*55}")
    print(f"Guardado: {SALIDA_CSV}")
    print(f"  {len(acum)} filas (toma x zona), {n_fiab} observaciones fiables")
    print(f"  {acum['toma'].nunique()} tomas, "
          f"reologias={sorted(acum['reologia'].dropna().unique())}")
    return acum


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if os.path.exists(SALIDA_CSV) and not RECALCULO:
        print(f"✓ Ya existe {SALIDA_CSV} (RECALCULO=False para reconstruir).")
    else:
        procesar_todas()
