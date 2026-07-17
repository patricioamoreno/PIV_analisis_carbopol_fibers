"""
trayectoria_comun.py
=====================
Utilidades compartidas para el sistema de reconstruccion lagrangiana con
validacion cruzada (E1/E2/E3), inspirado en el enfoque de
github.com/LukasWolff2002/Exp_Func_PIV_PTV (carpeta reconstruccion_lagrangiana/).

Reutiliza tu propia funcion asignar_zona (definir_zonas.py) y tu cargar_etapas
(carga_real.py) para que la clasificacion de zonas y los cortes de etapa sean
EXACTAMENTE los mismos que usa el resto de tu pipeline (no una copia paralela).

Fuente de datos PTV completo: ptv_merged.json por ensayo, el mismo archivo que
lee exportar_fibras_ultimo_frame.py -- pero aqui se cargan TODOS los frames,
no solo el ultimo, para poder reconstruir tracks reales (E1) en vez de solo
la foto final.
"""

import os
import re
import json
import numpy as np
import pandas as pd

from definir_zonas import asignar_zona
from carga_real import cargar_etapas


# ----------------------------------------------------------------------
# 1. Carga del PTV completo (todos los frames, no solo el ultimo)
# ----------------------------------------------------------------------
def cargar_ptv_completo(carpeta_ptv):
    """
    Lee ptv_merged.json COMPLETO (todos los frames) de un ensayo.
    Misma correccion de eje que exportar_fibras_ultimo_frame.py (y_mm *= -1)
    -- ESTA es la convencion de TU proyecto (definir_zonas.py, cache_zonas),
    no necesariamente la de otros repos que usen el mismo dato crudo con su
    propio sistema de ejes. Si cruzas resultados con otro pipeline, verifica
    que ambos usen la misma convencion antes de comparar zonas.

    Ademas de x_mm/y_mm/angle_deg, se extraen vx_mm_s/vy_mm_s (velocidad YA
    calculada por el propio algoritmo de PTV frame-a-frame) y length_mm -- mas
    precisas que estimarlas por diferencias finitas sobre el track completo.

    angle_deg se pliega a [0,180) al cargar (simetria de la fibra: 200 grados
    y 20 grados son la misma orientacion fisica). frame_idx se asigna por
    orden de timestamps UNICOS ordenados (robusto a que el JSON no venga
    estrictamente ordenado).

    Devuelve un DataFrame largo: una fila por (track_id, frame) con columnas:
    track_id, frame, t, x_mm, y_mm, angle_deg, vx_mm_s, vy_mm_s, length_mm,
    cam_name.
    """
    path = os.path.join(carpeta_ptv, "ptv_merged.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", []) if isinstance(data, dict) else data
    filas = []
    for fr in frames:
        t = fr.get("timestamp_s", np.nan)
        for fib in fr.get("fibers", []):
            if fib.get("possible_duplicate", False):
                continue
            filas.append({
                "track_id": fib.get("track_id"),
                "t": t,
                "x_mm": fib.get("x_mm", 0.0),
                "y_mm": -fib.get("y_mm", 0.0),   # correccion a sistema fisico
                "angle_deg": fib.get("angle_deg", 0.0),
                "vx_mm_s": fib.get("vx_mm_s", np.nan),
                # vy tambien se invierte: es la derivada de y_mm, que ya se invirtio
                "vy_mm_s": -fib.get("vy_mm_s", np.nan) if fib.get("vy_mm_s") is not None else np.nan,
                "length_mm": fib.get("length_mm", np.nan),
                "cam_name": fib.get("cam_name", "?"),
            })
    df = pd.DataFrame(filas)
    if df.empty:
        return df

    # pliegue de angulo por simetria de la fibra (0 y 180 son la misma orientacion)
    df["angle_deg"] = df["angle_deg"].astype(float) % 180.0

    # frame_idx robusto: por orden de timestamps UNICOS, no por posicion cruda
    # en la lista del JSON (evita problemas si no viene estrictamente ordenado)
    ts_unicos = np.sort(df["t"].unique())
    mapa_idx = {t: i for i, t in enumerate(ts_unicos)}
    df["frame"] = df["t"].map(mapa_idx).astype(int)

    return df


def resumen_fragmentacion(ptv_df):
    """
    Diagnostico rapido de fragmentacion de tracks (cuantos frames tiene cada
    track_id en promedio/mediana). Util para saber si E2 (stitching) es
    necesario o si los tracks ya son suficientemente continuos.
    """
    conteo = ptv_df.groupby("track_id")["frame"].nunique()
    return {"n_tracks": len(conteo), "frames_mediana": float(conteo.median()),
            "frames_media": float(conteo.mean()),
            "frames_min": int(conteo.min()), "frames_max": int(conteo.max())}


# ----------------------------------------------------------------------
# 2. Clasificacion de zona y regimen (etapa) por frame
# ----------------------------------------------------------------------
def clasificar_zona_y_etapa(ptv_df, etapas_json, toma, zonas_todas=None):
    """
    Agrega dos columnas al PTV completo: 'zona' (via asignar_zona, tu propia
    funcion) y 'regimen' (transicion/cuasi, via tu cargar_etapas -- el mismo
    t_quasi por zona que usa el resto del pipeline).

    Devuelve el DataFrame con columnas nuevas 'zona' y 'regimen' agregadas.
    """
    df = ptv_df.copy()
    zonas_arr = asignar_zona(df["x_mm"].to_numpy(), df["y_mm"].to_numpy())
    df["zona"] = zonas_arr

    zonas_presentes = zonas_todas or sorted(
        z for z in df["zona"].unique() if z and z != "fuera")
    cortes = cargar_etapas(etapas_json, df_piv=None, zonas=zonas_presentes,
                           toma=toma)

    corte_map = df["zona"].map(cortes)
    df["regimen"] = np.where(df["t"] <= corte_map, "transicion", "cuasi")
    df.loc[df["zona"].isin([None, "fuera"]), "regimen"] = "n/a"
    return df


# ----------------------------------------------------------------------
# 3. Zona modal (moda) de un track durante un regimen dado
# ----------------------------------------------------------------------
def zona_modal(df_track, regimen):
    """
    Zona mas frecuente (moda) que un track visito DURANTE un regimen
    especifico ('transicion' o 'cuasi'). None si no hay datos en ese regimen.
    Empate: se queda con la primera en orden de aparicion (estable).
    """
    sub = df_track[df_track["regimen"] == regimen]
    sub = sub[~sub["zona"].isin([None, "fuera"])]
    if sub.empty:
        return None
    return sub["zona"].mode(dropna=True).iloc[0]
