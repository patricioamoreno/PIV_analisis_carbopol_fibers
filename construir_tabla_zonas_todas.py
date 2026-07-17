"""
construir_tabla_zonas_todas.py
==============================
UNICO script para el enfoque de "zona final": recorre TODAS las tomas, cruza
el fluido PIV (cache_zonas) con la foto final de fibras (fibras_ultimo_frame)
por zona, y arma la tabla acumulada por (toma, zona) con orientacion (orden_S,
calidad_orientacion), dispersion (sigma_iso, indice_uniformidad) y los
predictores de fluido por etapa.

Fusiona lo que antes eran DOS scripts redundantes (construir_tabla_zonas_todas.py
+ acumular_tomas.py) que hacian exactamente el mismo trabajo -- emparejar
PIV<->fibras, cargar cada toma, armar tabla_por_zona y concatenar -- con solo
tres diferencias reales entre ambos:
  1) acumular_tomas.py ademas guardaba capa1/2/4 en CSV aparte. Eso es
     REDUNDANTE: generar_mapas.py calcula sus propias correlaciones desde la
     tabla directamente (funcion correlaciones()), no lee esos CSV. Se dejan
     disponibles aqui solo como export OPCIONAL (GUARDAR_CAPAS) por si sirven
     para inspeccion manual rapida, no porque algo mas los necesite.
  2) acumular_tomas.py importaba win10toast (notificacion de escritorio),
     dependencia SOLO de Windows que rompe el script en cualquier otro
     sistema. Se elimino.
  3) MIN_FIBRAS estaba en 1 en este script, lo que reintroduce el artefacto de
     "1 fibra = orden_S trivialmente 1" que ya se identifico como problema.
     Se vuelve a 5 (el estandar del proyecto) por defecto.

Fuentes de datos:
  - PIV   : cache_zonas/<carpeta>_zonas.npz  (los que hace construir_caches_zonas)
  - Fibras: fibras_ultimo_frame/<...>-ptv.csv  (CSV sueltos, foto final)
  - Etapas: etapas_zonas.json  (t_quasi por toma y zona)

Salida:
  acum_tabla_zona.csv  (1 fila por toma x zona: fluido + orden_S + calidad +
                        sigma + indice_uniformidad + meta)
  [opcional, GUARDAR_CAPAS=True]:
  acum_capa1_global.csv, acum_capa2_global.csv, acum_capa4_global.csv

Uso:
    python construir_tabla_zonas_todas.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd

from carga_real import cargar_todo, PREDICTORES, ETAPAS
from analisis_global import (tabla_por_zona, capa1_global, capa2_global,
                             capa4_global)
from criterio_exclusion import (cargar_exclusiones, aplicar_exclusiones,
                                reportar_asimetria, UMBRAL_D, DIR_TESTS)

# ============================================================
# CONFIGURACION — editar aqui
# ============================================================

CACHE_ZONAS = "cache_zonas"                 # .npz de PIV por toma
FIBRAS_DIR = "fibras_ultimo_frame"          # CSV de fibras (sueltos)
ETAPAS_JSON = "etapas_zonas.json"           # cortes t_quasi por toma/zona

SALIDA_CSV = "acum_tabla_zona.csv"

# True -> reconstruye aunque el CSV ya exista
RECALCULO = True

# minimo de fibras para que orden_S/calidad_orientacion de una zona sea
# fiable. Con 1 fibra, orden_S es trivialmente 1.0 (artefacto) -- no bajar
# de este valor sin una razon explicita.
MIN_FIBRAS = 5

# Filtros opcionales de carga (None = todas las tomas)
CAR_OBJETIVO = None      # p.ej. "02"
FIB_OBJETIVO = None      # p.ej. "1500"

# Export opcional de las Capas 1/2/4 globales a CSV aparte, para inspeccion
# manual. NO es requerido por generar_mapas.py (que las recalcula solas desde
# acum_tabla_zona.csv), asi que se puede dejar en False sin perder nada.
GUARDAR_CAPAS = True

# Filtro estructural: la orientacion de fibras solo es de interes en la VIGA
# (ver capa1_global/capa2_global, parametro solo_viga). Se mantiene en True
# para que coincida con la instruccion del profesor por defecto.
SOLO_VIGA_EN_CAPAS = True

# Exclusion de celdas (toma, zona, etapa) no comparables con el caso base.
# El criterio y su justificacion estan en criterio_exclusion.py -- en resumen:
# se marca como NaN el predictor de fluido de una celda cuando su d de Cohen
# frente al base es >= UMBRAL_D (0.5), IGNORANDO el p-valor (que con n de
# decenas de miles de puntos PIV declara significativo el 97% de los casos,
# la mitad de ellos con efecto insignificante).
#
# Se marcan celdas, no filas: una toma puede desviarse del base solo en una
# etapa y ser perfectamente usable en la otra (caso m73).
APLICAR_EXCLUSIONES = True

# Guarda TAMBIEN una version sin exclusiones, para el chequeo de sensibilidad
# de la Capa 4. El umbral es una decision metodologica, no un hecho medido:
# la conclusion deberia reportarse con y sin el.
GUARDAR_VERSION_SIN_EXCLUIR = True
SALIDA_CSV_SIN_EXCLUIR = "acum_tabla_zona_sin_excluir.csv"


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
    Devuelve lista de dicts: {cod, npz, csv, reologia, fibras}.
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

    # ── Version SIN exclusiones (chequeo de sensibilidad) ────────────
    # Se guarda antes de tocar nada, para poder correr las Capas con y sin
    # el criterio y comprobar si la conclusion depende del umbral elegido.
    if GUARDAR_VERSION_SIN_EXCLUIR:
        acum.to_csv(SALIDA_CSV_SIN_EXCLUIR, index=False)
        print(f"\n[sensibilidad] Guardado sin exclusiones: "
              f"{SALIDA_CSV_SIN_EXCLUIR}")

    # ── Exclusion de celdas no comparables con el base ───────────────
    if APLICAR_EXCLUSIONES:
        print(f"\n{'-'*55}")
        excl = cargar_exclusiones(DIR_TESTS, umbral_d=UMBRAL_D)
        if excl:
            acum, res_excl = aplicar_exclusiones(acum, excl)
        else:
            print("[criterio_exclusion] sin exclusiones que aplicar "
                  f"(¿corriste analisis.py para generar {DIR_TESTS}/?)")

    acum.to_csv(SALIDA_CSV, index=False)
    n_fiab = int(acum["fiable"].sum()) if "fiable" in acum else len(acum)
    print(f"\n{'='*55}")
    print(f"Guardado: {SALIDA_CSV}")
    print(f"  {len(acum)} filas (toma x zona), {n_fiab} observaciones fiables")
    print(f"  {acum['toma'].nunique()} tomas, "
          f"reologias={sorted(acum['reologia'].dropna().unique())}")

    # Cuantos predictores quedaron sin dato tras la exclusion: si una etapa
    # pierde muchas mas celdas que la otra, la Capa 4 esta comparando
    # conjuntos desiguales y hay que decirlo en el reporte.
    if APLICAR_EXCLUSIONES:
        for e in ETAPAS:
            cols = [f"{p}_{e}" for p in PREDICTORES if f"{p}_{e}" in acum]
            if cols:
                n_nan = int(acum[cols[0]].isna().sum())
                print(f"  {e:12s}: {n_nan}/{len(acum)} filas sin predictor "
                      f"({100*n_nan/len(acum):.0f}%)")

    if GUARDAR_CAPAS:
        c1 = capa1_global(acum, solo_viga=SOLO_VIGA_EN_CAPAS)
        c2 = capa2_global(acum, solo_viga=SOLO_VIGA_EN_CAPAS)
        c4 = capa4_global(c1)
        c1.to_csv("acum_capa1_global.csv", index=False)
        c2.to_csv("acum_capa2_global.csv", index=False)
        c4.to_csv("acum_capa4_global.csv", index=False)
        print(f"  (solo_viga={SOLO_VIGA_EN_CAPAS}) Guardados: "
              f"acum_capa1_global.csv, acum_capa2_global.csv, "
              f"acum_capa4_global.csv")
    return acum


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if os.path.exists(SALIDA_CSV) and not RECALCULO:
        print(f"✓ Ya existe {SALIDA_CSV} (RECALCULO=False para reconstruir).")
    else:
        procesar_todas()
        # La asimetria de exclusiones entre etapas afecta directamente a la
        # Capa 4 (que compara transicion contra cuasi). Se imprime al final
        # para que quede en el log de la corrida, no solo en un script aparte.
        if APLICAR_EXCLUSIONES:
            reportar_asimetria(DIR_TESTS)