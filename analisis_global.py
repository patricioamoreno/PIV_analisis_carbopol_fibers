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
                                  desviacion_objetivo, columna_de_zona)

RESPUESTAS = ["orden_S", "sigma_iso"]   # foco: alineamiento y dispersion

# ----------------------------------------------------------------------
# Construccion de la tabla por zona
# ----------------------------------------------------------------------
def tabla_por_zona(df_largo, min_fibras=5):
    """
    Devuelve un DataFrame de UNA fila por zona con:
      zona, n_fibras, orden_S, theta_med, sigma_iso, fiable,
      objetivo, calidad_orientacion, cos2delta, columna, desviacion_media,
      V_transicion, omega_transicion, gamma_dot_transicion,
      V_cuasi, omega_cuasi, gamma_dot_cuasi

    cos2delta = cos(2*(theta_med - objetivo)) y columna = columna de viga
    (Vf<fila>c<col> -> col) son el segundo factor de calidad_orientacion,
    expuestos por separado (NaN en zonas sin objetivo, p.ej. Z1/Z2/Z3).

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
        # Segundo factor de calidad_orientacion, expuesto por separado (ver
        # Tarea 6): calidad = 1/2 + (S/2)*cos2delta, con cos2delta constante
        # por columna de viga. Agregar cos2delta entre columnas mezclaria el
        # gradiente cinematico con la asignacion del objetivo, por eso no se
        # suma a RESPUESTAS ni se analiza por capas aqui: solo se deja
        # disponible como columna.
        cos2delta = (np.cos(2 * np.deg2rad(tm - objetivo))
                    if objetivo is not None else np.nan)
        col = columna_de_zona(zona)
        columna = col if col is not None else np.nan
        filas.append({"zona": zona, "n_fibras": len(g),
                      "orden_S": S, "theta_med": tm,
                      "sigma_iso": disp["sigma_iso"],
                      "indice_uniformidad": disp["indice_uniformidad"],
                      "fiable": len(g) >= min_fibras,
                      "objetivo": objetivo,
                      "calidad_orientacion": calidad,
                      "cos2delta": cos2delta,
                      "columna": columna,
                      "desviacion_media": desv})
    tabla = pd.DataFrame(filas)

    # Verificacion de la identidad exacta calidad = 1/2 + (S/2)*cos2delta
    # (ver docstring de calidad_orientacion + Tarea 6 del brief). Si se
    # rompe, algo en el calculo de S/theta_med/objetivo dejo de ser
    # consistente y no hay que seguir adelante en silencio.
    con_objetivo = tabla.dropna(subset=["cos2delta"])
    if len(con_objetivo):
        identidad = (0.5 + 0.5 * con_objetivo["orden_S"] * con_objetivo["cos2delta"]
                    - con_objetivo["calidad_orientacion"]).abs()
        max_diff = float(identidad.max())
        if max_diff >= 1e-12:
            raise RuntimeError(
                "Identidad calidad_orientacion = 1/2 + (S/2)*cos2delta rota: "
                f"max|diff| = {max_diff:.3e} (esperado < 1e-12)")

    pred = (df_largo.drop_duplicates(["zona", "etapa"])
            [["zona", "etapa"] + PREDICTORES])
    wide = pred.pivot(index="zona", columns="etapa", values=PREDICTORES)
    wide.columns = [f"{p}_{e}" for p, e in wide.columns]
    wide = wide.reset_index()

    return tabla.merge(wide, on="zona", how="left")


# ----------------------------------------------------------------------
# Capa 1 global: Spearman a traves de zonas
# ----------------------------------------------------------------------
def spearman_parcial(x, y, z):
    """
    Correlacion parcial de Spearman entre x e y controlando por z.

    Se rankean las tres variables y se calcula la correlacion de Pearson
    entre los residuos de rank(x)~rank(z) y rank(y)~rank(z). Es la version
    no parametrica estandar del control por una covariable.

    POR QUE ES NECESARIA AQUI. n_fibras esta fuertemente asociado tanto a
    las respuestas como a los predictores:

        rho(n_fibras, sigma_iso)     = +0,82 (car-02)  +0,54 (car-05)
        rho(n_fibras, V_transicion)  = +0,76 (car-02)  +0,51 (car-05)

    y ademas sigma_iso es la desviacion MUESTRAL de los centroides en una
    celda de tamaño fijo, de modo que crece con n por puro muestreo (con
    pocas fibras subestima la extension real). Sin controlar por n_fibras,
    parte de la asociacion reportada es un artefacto de cobertura.

    Devuelve (rho_parcial, p_valor, n). El p-valor sigue siendo
    anticonservador por la pseudo-replicacion (zonas de una misma corrida);
    ver la advertencia estadistica del proyecto.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 5 or np.std(x) == 0 or np.std(y) == 0 or np.std(z) == 0:
        return np.nan, np.nan, n
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))

    def _resid(a, b):
        B = np.column_stack([np.ones(len(b)), b])
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef

    ex, ey = _resid(rx, rz), _resid(ry, rz)
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, np.nan, n
    rho, pv = stats.pearsonr(ex, ey)
    return float(rho), float(pv), n


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

                # Control por n_fibras (ver spearman_parcial).
                if "n_fibras" in t:
                    rho_p, pv_p, n_p = spearman_parcial(
                        t[col].to_numpy(float), t[resp].to_numpy(float),
                        t["n_fibras"].to_numpy(float))
                else:
                    rho_p, pv_p, n_p = np.nan, np.nan, np.nan

                filas.append({"respuesta": resp, "predictor": p, "etapa": e,
                              "rho": rho, "p_value": pv, "n_zonas": n,
                              "rho_parcial_nf": rho_p,
                              "p_parcial_nf": pv_p,
                              "n_parcial": n_p,
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
    # Delta segun la definicion de la memoria:
    #     Delta_p = |rho_transicion| - |rho_cuasi|
    # positivo  => la etapa de TRANSICION domina  (coherente con domina_etapa)
    #
    # La version anterior calculaba el signo OPUESTO y lo llamaba
    # 'delta_abs_rho', de modo que la columna contradecia tanto la leyenda de
    # la tabla como la columna domina_etapa de su propia fila.
    piv["delta_abs_rho"] = (piv["rho_transicion"].abs().fillna(0) -
                            piv["rho_cuasi"].abs().fillna(0))
    return piv.sort_values(["respuesta", "predictor"]).reset_index(drop=True)

# ----------------------------------------------------------------------
# Capa 1 y 4 ESTRATIFICADAS por reologia
# ----------------------------------------------------------------------
def capa1_estratificado(tabla, col_estrato="reologia", solo_viga=True):
    """
    Capa 1 corrida DENTRO de cada estrato (por defecto, cada reologia).

    Es la forma correcta segun el propio protocolo del trabajo: Car-0,2 % y
    Car-0,5 % operan en escalas cinematicas no comparables (los tiempos de
    etapa difieren por factores > 2), de modo que agrupar ambas en una sola
    correlacion mezcla dos poblaciones y puede producir un efecto Simpson.
    """
    salidas = []
    for estrato, g in tabla.groupby(col_estrato):
        c1 = capa1_global(g, solo_viga=solo_viga)
        c1.insert(0, col_estrato, estrato)
        salidas.append(c1)
    if not salidas:
        return pd.DataFrame()
    return pd.concat(salidas, ignore_index=True)


def capa4_estratificado(c1_estrat, col_estrato="reologia"):
    """
    Capa 4 DENTRO de cada estrato.

    POR QUE EXISTE. run_real.py y construir_tabla_zonas_todas.py llamaban
    unicamente a capa4_global(c1) sobre la tabla COMPLETA, es decir,
    mezclando ambas reologias en una sola correlacion. El CSV resultante
    (acum_capa4_global.csv) es, por tanto, agrupado (pooled), pese a que la
    tabla correspondiente de la memoria esta estratificada por reologia y a
    que el protocolo del trabajo prohibe explicitamente mezclarlas. No
    existia ninguna version estratificada de la Capa 4 en el codigo.
    """
    salidas = []
    for estrato, g in c1_estrat.groupby(col_estrato):
        c4 = capa4_global(g)
        c4.insert(0, col_estrato, estrato)
        salidas.append(c4)
    if not salidas:
        return pd.DataFrame()
    return pd.concat(salidas, ignore_index=True)
