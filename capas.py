"""
capas.py
========
Las 4 capas de analisis para predecir orientacion (theta) y dispersion de
fibras a partir de los predictores de fluido (V, omega, gamma_dot), por zona
y por etapa.

Capa 1: Correlacion de Spearman robusta (predictor vs theta), por zona/etapa.
Capa 2: Regresion multivariada con coeficientes estandarizados + R2 parcial,
        con diagnostico VIF.
Capa 3: Random Forest + importancia por permutacion (agnostica a colinealidad
        y a no linealidad), por zona/etapa.
Capa 4: Comparativa temporal transicion vs cuasi (unifica pesos de C2 y C3).

Decisiones de modelado
----------------------
- theta es CIRCULAR. Para C1/C2/C3 se usa una respuesta escalar bien definida:
  el alineamiento local de cada fibra respecto a la direccion media de su zona,
  a_i = cos(2*(theta_i - theta_media_zona)) in [-1,1]. Predecir 'a' es
  equivalente a preguntar que variables del fluido hacen que una fibra quede
  mas o menos alineada con el patron dominante de su zona, que es justamente
  la pregunta fisica. (Se expone tambien theta crudo por si se prefiere.)
- Unidad de observacion para theta: fibra-dentro-de-zona (N = nº fibras/zona).
- Dispersion es por zona: se analiza aparte (capa de dispersion global).
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold

from nucleo import (PREDICTORES, ETAPAS, N_ZONAS, etiqueta_zona,
                    direccion_media_circular, calcular_vif,
                    dispersion_centroides, theta_a_rad)

warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MIN_FIBRAS = len(PREDICTORES) + 3   # minimo de fibras por zona para modelar


def _zonas_de(df):
    """
    Lista de (zona_id, nombre) presentes en el df. Si existe columna 'zona'
    (datos reales) usa ese nombre; si no (simulado), usa etiqueta_zona(id).
    """
    ids = sorted(df["zona_id"].unique())
    if "zona" in df.columns:
        nombre = (df.drop_duplicates("zona_id")
                  .set_index("zona_id")["zona"].to_dict())
        return [(z, nombre.get(z, etiqueta_zona(z))) for z in ids]
    return [(z, etiqueta_zona(z)) for z in ids]


# ----------------------------------------------------------------------
# Construccion de la respuesta de alineamiento (circular -> escalar)
# ----------------------------------------------------------------------
def agregar_alineamiento(df):
    """
    Agrega columna 'alineamiento' = cos(2*(theta - theta_media_zona)) usando
    la direccion media circular de cada zona (calculada sobre la etapa cuasi,
    que es la foto final). 1 = fibra alineada con el patron de la zona,
    -1 = perpendicular. Es la respuesta escalar para C1/C2/C3.
    """
    df = df.copy()
    medias = {}
    for z, _ in _zonas_de(df):
        th = df[(df.zona_id == z) & (df.etapa == "cuasi")]["theta"]
        tm, _r = direccion_media_circular(th)
        medias[z] = tm
    tm_arr = df["zona_id"].map(medias).to_numpy()
    dif = theta_a_rad(df["theta"].to_numpy()) - np.deg2rad(tm_arr)
    df["theta_media_zona"] = tm_arr
    df["alineamiento"] = np.cos(2 * dif)
    return df


# ======================================================================
# CAPA 1 — SPEARMAN
# ======================================================================
def capa1_spearman(df, respuesta="alineamiento"):
    filas = []
    zonas = _zonas_de(df)
    for etapa in ETAPAS:
        for z, znom in zonas:
            sub = df[(df.etapa == etapa) & (df.zona_id == z)]
            for p in PREDICTORES:
                x = sub[p].to_numpy(float)
                y = sub[respuesta].to_numpy(float)
                m = ~(np.isnan(x) | np.isnan(y))
                n = int(m.sum())
                if n < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                    rho, pv = np.nan, np.nan
                else:
                    rho, pv = stats.spearmanr(x[m], y[m])
                filas.append({"etapa": etapa, "zona_id": z,
                              "zona": znom, "predictor": p,
                              "rho": rho, "p_value": pv, "n": n})
    res = pd.DataFrame(filas)
    res["abs_rho"] = res["rho"].abs()
    res["signif"] = res["p_value"] < 0.05
    return res


# ======================================================================
# CAPA 2 — REGRESION ESTANDARIZADA
# ======================================================================
def _r2_parcial(X, y):
    full = LinearRegression().fit(X, y).score(X, y)
    out = {}
    for c in X.columns:
        Xr = X.drop(columns=[c])
        if Xr.shape[1] == 0:
            out[c] = full
        else:
            out[c] = full - LinearRegression().fit(Xr, y).score(Xr, y)
    return out, full


def capa2_regresion(df, respuesta="alineamiento"):
    filas = []
    for etapa in ETAPAS:
        for z, znom in _zonas_de(df):
            sub = df[(df.etapa == etapa) & (df.zona_id == z)].dropna(
                subset=PREDICTORES + [respuesta])
            n = len(sub)
            if n < MIN_FIBRAS:
                continue
            sd = sub[PREDICTORES].std()
            preds = [p for p in PREDICTORES if sd[p] > 0]
            if not preds:
                continue
            X = pd.DataFrame(
                StandardScaler().fit_transform(sub[preds]),
                columns=preds, index=sub.index)
            y = sub[respuesta].to_numpy(float)

            vif = calcular_vif(sub[preds]) if len(preds) > 1 else \
                pd.Series({preds[0]: 1.0})
            r2p, r2full = _r2_parcial(X, y)

            try:
                import statsmodels.api as sm
                mod = sm.OLS(y, sm.add_constant(X)).fit()
                beta = {p: mod.params[p] for p in preds}
                pval = {p: mod.pvalues[p] for p in preds}
            except Exception:
                reg = LinearRegression().fit(X, y)
                beta = dict(zip(preds, reg.coef_))
                pval = {p: np.nan for p in preds}

            for p in preds:
                filas.append({"etapa": etapa, "zona_id": z,
                              "zona": znom, "predictor": p,
                              "beta_std": beta[p], "abs_beta_std": abs(beta[p]),
                              "p_value": pval[p], "r2_parcial": r2p[p],
                              "r2_modelo": r2full, "VIF": float(vif[p]),
                              "n": n})
    return pd.DataFrame(filas)


# ======================================================================
# CAPA 3 — RANDOM FOREST + IMPORTANCIA POR PERMUTACION
# ======================================================================
def capa3_random_forest(df, respuesta="alineamiento", n_estimators=300,
                        n_repeats=30, seed=0):
    """
    Por zona y etapa ajusta un Random Forest y mide importancia por
    permutacion. Esta importancia es AGNOSTICA a la colinealidad y captura
    relaciones no lineales e interacciones (justo lo que C2 no puede).

    Reporta tambien R2 por validacion cruzada (5-fold) como medida honesta
    de cuanta señal real hay (R2_cv<=0 => el fluido de esa zona no predice
    la respuesta mejor que la media).
    """
    rng = np.random.default_rng(seed)
    filas = []
    resumen = []
    for etapa in ETAPAS:
        for z, znom in _zonas_de(df):
            sub = df[(df.etapa == etapa) & (df.zona_id == z)].dropna(
                subset=PREDICTORES + [respuesta])
            n = len(sub)
            if n < MIN_FIBRAS:
                continue
            X = sub[PREDICTORES].to_numpy(float)
            y = sub[respuesta].to_numpy(float)

            rf = RandomForestRegressor(
                n_estimators=n_estimators, random_state=seed,
                max_depth=None, min_samples_leaf=3, n_jobs=-1)

            k = min(5, max(2, n // 5))
            try:
                cv = cross_val_score(
                    rf, X, y, cv=KFold(n_splits=k, shuffle=True,
                                       random_state=seed),
                    scoring="r2")
                r2_cv = float(np.mean(cv))
            except Exception:
                r2_cv = np.nan

            rf.fit(X, y)
            r2_train = rf.score(X, y)
            perm = permutation_importance(
                rf, X, y, n_repeats=n_repeats, random_state=seed,
                scoring="r2", n_jobs=-1)

            imps = perm.importances_mean
            tot = np.sum(np.clip(imps, 0, None))
            for j, p in enumerate(PREDICTORES):
                filas.append({
                    "etapa": etapa, "zona_id": z, "zona": znom,
                    "predictor": p,
                    "imp_perm": float(imps[j]),
                    "imp_perm_std": float(perm.importances_std[j]),
                    "imp_rel": float(np.clip(imps[j], 0, None) / tot)
                              if tot > 0 else 0.0,
                    "r2_cv": r2_cv, "r2_train": r2_train, "n": n})
            resumen.append({"etapa": etapa, "zona_id": z,
                            "zona": znom, "r2_cv": r2_cv,
                            "r2_train": r2_train, "n": n})
    return pd.DataFrame(filas), pd.DataFrame(resumen)


# ======================================================================
# CAPA 4 — COMPARATIVA TEMPORAL
# ======================================================================
def capa4_comparativa(c2, c3):
    """
    Unifica los pesos de Capa 2 (|beta_std|) y Capa 3 (imp_rel) y los compara
    entre etapa de transicion y cuasi, por zona y predictor.

    Devuelve un DataFrame ancho con, para cada (zona, predictor):
      beta_transicion, beta_cuasi, imp_transicion, imp_cuasi,
      y 'domina_etapa' = etapa donde el predictor pesa mas (segun RF, que es
      el criterio robusto). Sirve para concluir en que momento del llenado se
      define el comportamiento de cada zona.
    """
    b = (c2.pivot_table(index=["zona", "predictor"], columns="etapa",
                        values="abs_beta_std")
         .rename(columns={"transicion": "beta_transicion",
                          "cuasi": "beta_cuasi"}))
    i = (c3.pivot_table(index=["zona", "predictor"], columns="etapa",
                        values="imp_rel")
         .rename(columns={"transicion": "imp_transicion",
                          "cuasi": "imp_cuasi"}))
    out = b.join(i, how="outer").reset_index()
    for col in ["beta_transicion", "beta_cuasi",
                "imp_transicion", "imp_cuasi"]:
        if col not in out:
            out[col] = np.nan

    def domina(r):
        it, ic = r.get("imp_transicion", np.nan), r.get("imp_cuasi", np.nan)
        if np.isnan(it) and np.isnan(ic):
            return "indeterminado"
        it = 0 if np.isnan(it) else it
        ic = 0 if np.isnan(ic) else ic
        if abs(it - ic) < 0.05:
            return "ambas"
        return "transicion" if it > ic else "cuasi"

    out["domina_etapa"] = out.apply(domina, axis=1)
    out["delta_imp_cuasi_menos_trans"] = (out["imp_cuasi"].fillna(0) -
                                          out["imp_transicion"].fillna(0))
    return out.sort_values(["zona", "predictor"]).reset_index(drop=True)


# ======================================================================
# CAPA DISPERSION  (analisis global de 6 zonas)
# ======================================================================
def analisis_dispersion(df):
    """
    Dispersion es por zona, no por fibra. Se construye una tabla de 6 zonas
    (foto final) con su dispersion de centroides, y se cruza con los
    predictores de fluido por etapa para ver que variable de que etapa se
    asocia mas a la dispersion final. Con solo 6 zonas se usa Spearman
    (no parametrico) como exploracion; los resultados son indicativos, no
    concluyentes (N pequeño).

    Devuelve:
      tabla_zonas : 6 filas con sigma_iso (foto final) y predictores por etapa.
      corr_disp   : correlacion Spearman dispersion vs cada predictor/etapa.
    """
    # dispersion por zona desde la foto final (cuasi, posiciones unicas/fibra)
    foto = df[df.etapa == "cuasi"].drop_duplicates("fibra_id")
    zonas = _zonas_de(df)
    disp_rows = []
    for z, znom in zonas:
        s = foto[foto.zona_id == z]
        d = dispersion_centroides(s["x_mm"], s["y_mm"])
        disp_rows.append({"zona_id": z, "zona": znom,
                          "sigma_iso": d["sigma_iso"],
                          "area_aprox": d["area_aprox"], "n_fibras": d["n"]})
    tabla = pd.DataFrame(disp_rows)

    # predictores medianos por zona y etapa
    med = (df.groupby(["zona_id", "etapa"])[PREDICTORES]
           .median().reset_index())
    wide = med.pivot(index="zona_id", columns="etapa", values=PREDICTORES)
    wide.columns = [f"{p}_{e}" for p, e in wide.columns]
    wide = wide.reset_index()
    tabla = tabla.merge(wide, on="zona_id", how="left")

    # correlaciones dispersion vs predictor/etapa
    corr_rows = []
    for p in PREDICTORES:
        for e in ETAPAS:
            col = f"{p}_{e}"
            if col not in tabla:
                continue
            x = tabla[col].to_numpy(float)
            y = tabla["sigma_iso"].to_numpy(float)
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() >= 3 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                rho, pv = stats.spearmanr(x[m], y[m])
            else:
                rho, pv = np.nan, np.nan
            corr_rows.append({"predictor": p, "etapa": e,
                              "rho_vs_dispersion": rho, "p_value": pv,
                              "n_zonas": int(m.sum())})
    corr = pd.DataFrame(corr_rows).sort_values(
        "rho_vs_dispersion", key=lambda s: s.abs(), ascending=False)
    return tabla, corr