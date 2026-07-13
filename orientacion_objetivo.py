"""
orientacion_objetivo.py
=======================
Metricas de orientacion de fibras respecto al ANGULO OBJETIVO de cada zona.

A diferencia del orden-parametro S (que mide si las fibras estan alineadas
ENTRE SI, sin importar la direccion), estas metricas miden si las fibras estan
alineadas EN EL ANGULO QUE LES CORRESPONDE segun la columna de la viga, para
resistir los esfuerzos previstos (flexion y corte).

Convencion de angulo (segun tus datos):
    0 grados  = fibra horizontal (acostada)
    90 grados = fibra vertical

Objetivos por columna (editables abajo):
    Columna 1 -> 135 grados
    Columna 2 -> 0 grados (plana)
    Columna 3 -> 45 grados

Las dos metricas:
  1) desviacion_objetivo(theta, zona): error angular de una fibra respecto a su
     objetivo, en grados [0, 90]. 0 = perfecta, 90 = perpendicular al objetivo.
  2) calidad_orientacion(thetas, zona): puntaje de la zona en [0, 1].
     1 = todas las fibras en su objetivo; 0 = todas perpendiculares al objetivo.
     Es el analogo "correcto" de orden_S, pero midiendo cercania al OBJETIVO.
     Formula: promedio de cos^2(theta - objetivo), que respeta la naturaleza
     circular del angulo (135 y -45 son la misma orientacion de fibra).
"""

import numpy as np
import re

# ============================================================
# CONFIGURACION — editar los angulos objetivo aqui
# ============================================================
# Objetivo por COLUMNA (col 1, 2, 3). Angulos en grados, 0=horizontal.
OBJETIVO_COLUMNA = {
    1: 135.0,   # columna izquierda
    2: 0.0,     # columna central (plana)
    3: 45.0,    # columna derecha
}

# Si una zona no es de viga (p.ej. Z1/Z2/Z3 de la L) no tiene objetivo
# estructural definido -> devuelve None y se excluye de estas metricas.


def columna_de_zona(zona):
    """
    Extrae el numero de columna de una zona de viga tipo 'Vf1c3' -> 3.
    Devuelve None si la zona no es de viga (no tiene 'c<numero>').
    """
    s = str(zona)
    if not s.startswith("V"):
        return None
    m = re.search(r"c(\d+)", s)
    return int(m.group(1)) if m else None


def objetivo_de_zona(zona):
    """Angulo objetivo [grados] de una zona, o None si no aplica."""
    col = columna_de_zona(zona)
    if col is None:
        return None
    return OBJETIVO_COLUMNA.get(col, None)


def desviacion_objetivo(theta_deg, zona):
    """
    Error angular de una fibra respecto al objetivo de su zona, en grados
    [0, 90]. Usa distancia circular con periodo 180 (una fibra no tiene
    sentido, 135 = -45). 0 = alineada al objetivo, 90 = perpendicular.
    Devuelve np.nan si la zona no tiene objetivo.
    """
    obj = objetivo_de_zona(zona)
    if obj is None:
        return np.nan
    d = (float(theta_deg) - obj) % 180.0      # diferencia en [0,180)
    return d if d <= 90.0 else 180.0 - d       # plegar a [0,90]


def calidad_orientacion(thetas_deg, zona):
    """
    Puntaje [0,1] de que tan bien orientadas estan las fibras de una zona
    respecto a SU objetivo. 1 = perfectas, 0 = perpendiculares.
    = promedio de cos^2(theta - objetivo).  np.nan si la zona no tiene objetivo
    o no hay fibras.
    """
    obj = objetivo_de_zona(zona)
    if obj is None:
        return np.nan
    th = np.asarray(thetas_deg, dtype=float)
    th = th[~np.isnan(th)]
    if len(th) == 0:
        return np.nan
    dif = np.deg2rad(th - obj)
    return float(np.mean(np.cos(dif) ** 2))


def calidad_desde_desviacion(desv_grados):
    """
    Convierte una desviacion angular [0,90] al puntaje [0,1] equivalente,
    por si se quiere el puntaje de una sola fibra: cos^2(desv).
    """
    return float(np.cos(np.deg2rad(desv_grados)) ** 2)
