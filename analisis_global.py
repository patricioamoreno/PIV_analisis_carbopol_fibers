"""
analisis_global.py
==================
Analisis GLOBAL entre zonas (el que describio el profesor en la reunion 23-06):
cada zona aporta UN punto = su fluido (mediana por zona/etapa) frente al
RESUMEN de orientacion y dispersion de las fibras que cayeron en ella.

Por que global y no intra-zona
------------------------------
El PIV entrega un escalar por (zona, etapa) -> dentro de una zona el predictor
es constante y no se puede correlacionar con theta fibra a fibra. La pregunta
respondible con estos datos es: "entre zonas, que variable del fluido (y en que
etapa) se asocia a que las fibras queden mas alineadas / mas dispersas".

Unidad de observacion: la ZONA (N = nº de zonas con fibras, aqui hasta 8).
Respuestas por zona (de la foto final):
  - orden_S    : orden-parametro de orientacion en [0,1] (1=fibras alineadas).
  - theta_med  : direccion media circular de la zona [grados].
  - sigma_iso  : dispersion de centroides (RMS) [mm].
Predictores por zona y etapa: V, omega, gamma_dot (mediana robusta).

Metodos
-------
- Capa 1-global: Spearman entre cada predictor (por etapa) y cada respuesta,
  a traves de las zonas. Con N pequeño (<=8) es EXPLORATORIO: detecta señal,
  no la confirma. Se reportan rho y p, y se marca la robustez (N).
- Capa 2-global: regresion estandarizada multivariada por etapa (respuesta ~
  V+omega+gamma_dot a traves de zonas) con beta comparables, R2 y VIF. Solo si
  hay suficientes zonas (N >= nº predictores + 2).
- Capa 4-global: compara la importancia (|rho| y |beta|) entre transicion y
  cuasi para concluir que etapa define cada respuesta.

Random Forest NO se usa aqui: con N<=8 un RF sobreajusta sin remedio. La via
robusta a esa escala es Spearman + regresion lineal con lectura cautelosa.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from carga_real import PREDICTORES, ETAPAS
from nucleo import direccion_media_circular, orden_parametro, \
    dispersion_centroides, calcular_vif
from orientacion_objetivo import (calidad_orientacion, objetivo_de_zona,
                                  desviacion_objetivo)

RESPUESTAS = ["orden_S", "sigma_iso"]   # foco: alineamiento y dispersion

# ----------------------------------------------------------------------
# Construccion de la tabla por zona
# ----------------------------------------------------------------------
def tabla_por_zona(df_largo, min_fibras=5):
    """
    Devuelve un DataFrame de UNA fila por zona con:
      zona, n_fibras, orden_S, theta_med, sigma_iso, fiable,
      V_transicion, omega_transicion, gamma_dot_transicion,
      V_cuasi, omega_cuasi, gamma_dot_cuasi

    'fiable' = n_fibras >= min_fibras. El orden_S con 1-2 fibras es un
    artefacto (1 fibra => S=1 trivialmente), por eso las zonas no fiables se
    marcan y se EXCLUYEN de las correlaciones (capa1/2/4 global filtran por
    'fiable'). Se mantienen en la tabla para transparencia.
    """
    foto = df_largo[df_largo.etapa == "cuasi"].drop_duplicates("fibra_id")

    filas = []
    for zona, g in foto.groupby("zona"):
        S = orden_parametro(g["theta"])
        tm, _ = direccion_media_circular(g["theta"])
        disp = dispersion_centroides(g["x_mm"], g["y_mm"])
        # objetivo_de_zona/calidad_orientacion reciben el NOMBRE de la zona
        # (ej. 'Vf1c1'), no el angulo objetivo. calidad_orientacion ya agrega
        # todas las fibras de la zona; desviacion_objetivo es por-fibra, por
        # eso se promedia con un loop. (Bug corregido: antes se llamaba con
        # 'objetivo' en vez de 'zona', lo que devolvia NaN siempre.)
        objetivo = objetivo_de_zona(zona)
        calidad = calidad_orientacion(g["theta"], zona)
        desv = (np.nanmean([desviacion_objetivo(t, zona) for t in g["theta"]])
                if objetivo is not None else np.nan)
        filas.append({"zona": zona, "n_fibras": len(g),
                      "orden_S": S, "theta_med": tm,
                      "sigma_iso": disp["sigma_iso"],
                      "indice_uniformidad": disp["indice_uniformidad"],
                      "fiable": len(g) >= min_fibras,
                      "objetivo": objetivo,
                      "calidad_orientacion": calidad,
                      "desviacion_media": desv})
    tabla = pd.DataFrame(filas)

    pred = (df_largo.drop_duplicates(["zona", "etapa"])
            [["zona", "etapa"] + PREDICTORES])
    wide = pred.pivot(index="zona", columns="etapa", values=PREDICTORES)
    wide.columns = [f"{p}_{e}" for p, e in wide.columns]
    wide = wide.reset_index()

    return tabla.merge(wide, on="zona", how="left")


# ----------------------------------------------------------------------
# Capa 1 global: Spearman a traves de zonas
# ----------------------------------------------------------------------
def capa1_global(tabla, solo_viga=True):
    """
    solo_viga=True (default): la orientacion de fibras solo se analiza en la
    VIGA (zonas 'V*'), siguiendo instruccion del profesor (reunion 10-07):
    la L es solo conducto de entrada, sin funcion estructural, por lo que su
    orientacion final no es de interes como variable de RESPUESTA. La L sigue
    disponible como predictor de flujo en otros analisis (p.ej. adveccion),
    solo se excluye aqui como zona de respuesta.
    """
    # solo zonas fiables (orden_S robusto). Si no existe 'fiable', usa todas.
    t = tabla[tabla["fiable"]] if "fiable" in tabla else tabla
    if solo_viga and "zona" in t.columns:
        t = t[t["zona"].astype(str).str.startswith("V")]
    filas = []
    for resp in RESPUESTAS:
        for p in PREDICTORES:
            for e in ETAPAS:
                col = f"{p}_{e}"
                if col not in t:
                    continue
                x = t[col].to_numpy(float)
                y = t[resp].to_numpy(float)
                m = ~(np.isnan(x) | np.isnan(y))
                n = int(m.sum())
                if n >= 3 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                    rho, pv = stats.spearmanr(x[m], y[m])
                else:
                    rho, pv = np.nan, np.nan
                filas.append({"respuesta": resp, "predictor": p, "etapa": e,
                              "rho": rho, "p_value": pv, "n_zonas": n,
                              "abs_rho": abs(rho) if rho == rho else np.nan})
    return pd.DataFrame(filas).sort_values(
        ["respuesta", "abs_rho"], ascending=[True, False])


# ----------------------------------------------------------------------
# Capa 2 global: regresion estandarizada por etapa
# ----------------------------------------------------------------------
def capa2_global(tabla, solo_viga=True):
    """solo_viga=True (default): ver docstring de capa1_global."""
    t = tabla[tabla["fiable"]] if "fiable" in tabla else tabla
    if solo_viga and "zona" in t.columns:
        t = t[t["zona"].astype(str).str.startswith("V")]
    filas = []
    for resp in RESPUESTAS:
        for e in ETAPAS:
            cols = [f"{p}_{e}" for p in PREDICTORES if f"{p}_{e}" in t]
            sub = t.dropna(subset=cols + [resp])
            n = len(sub)
            if n < len(cols) + 2 or not cols:
                continue
            sd = sub[cols].std()
            cols_v = [c for c in cols if sd[c] > 0]
            if len(cols_v) < 1:
                continue
            X = pd.DataFrame(StandardScaler().fit_transform(sub[cols_v]),
                             columns=cols_v, index=sub.index)
            y = sub[resp].to_numpy(float)
            r2 = LinearRegression().fit(X, y).score(X, y)
            vif = calcular_vif(sub[cols_v]) if len(cols_v) > 1 else \
                pd.Series({cols_v[0]: 1.0})
            try:
                import statsmodels.api as sm
                mod = sm.OLS(y, sm.add_constant(X)).fit()
                beta = {c: mod.params[c] for c in cols_v}
                pval = {c: mod.pvalues[c] for c in cols_v}
            except Exception:
                reg = LinearRegression().fit(X, y)
                beta = dict(zip(cols_v, reg.coef_))
                pval = {c: np.nan for c in cols_v}
            for c in cols_v:
                p = c.rsplit("_", 1)[0]
                filas.append({"respuesta": resp, "etapa": e, "predictor": p,
                              "beta_std": beta[c], "abs_beta_std": abs(beta[c]),
                              "p_value": pval[c], "r2_modelo": r2,
                              "VIF": float(vif[c]), "n_zonas": n})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# Capa 4 global: comparativa temporal
# ----------------------------------------------------------------------
def capa4_global(c1g):
    """Compara |rho| entre etapas, por respuesta y predictor."""
    piv = c1g.pivot_table(index=["respuesta", "predictor"], columns="etapa",
                          values="rho")
    piv = piv.rename(columns={"transicion": "rho_transicion",
                              "cuasi": "rho_cuasi"}).reset_index()
    for col in ["rho_transicion", "rho_cuasi"]:
        if col not in piv:
            piv[col] = np.nan
    piv["domina_etapa"] = np.where(
        piv["rho_transicion"].abs().fillna(0) >=
        piv["rho_cuasi"].abs().fillna(0), "transicion", "cuasi")
    piv["delta_abs_rho"] = (piv["rho_cuasi"].abs().fillna(0) -
                            piv["rho_transicion"].abs().fillna(0))
    return piv.sort_values(["respuesta", "predictor"]).reset_index(drop=True)