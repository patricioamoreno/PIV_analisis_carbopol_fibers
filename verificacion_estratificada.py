"""
verificacion_estratificada.py
=============================
Análisis CORREGIDO, estratificado por reología.

Reemplaza a verificacion_resultados.py, que agregaba ambas reologías
y por tanto reportaba correlaciones que son artefactos de agregación
(paradoja de Simpson).

Correcciones respecto a la versión anterior:
  1. Todos los Spearman usan manejo explícito de NaN. El criterio de
     exclusión (criterio_exclusion.py) inserta NaN a propósito en celdas no
     comparables con el base; sin ese manejo, scipy propaga el NaN y devuelve
     rho=nan para toda la columna.
  2. Se reportan AUTOMÁTICAMENTE las dos versiones — con y sin exclusiones —
     lado a lado. El umbral d=0.5 es una decisión metodológica, no un hecho
     medido: la conclusión debe presentarse con y sin él (sensibilidad).
  3. Se eliminaron los números hardcodeados de la versión vieja (que ya no
     correspondían a la corrida actual). Todo lo que se imprime se calcula.

Uso:  python verificacion_estratificada.py [ruta_csv] [ruta_csv_sin_excluir]
      (por defecto: acum_tabla_zona.csv y acum_tabla_zona_sin_excluir.csv)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "acum_tabla_zona.csv"
CSV_SIN = (sys.argv[2] if len(sys.argv) > 2
           else "acum_tabla_zona_sin_excluir.csv")

PRED = ["V_cuasi", "V_transicion", "omega_cuasi", "omega_transicion",
        "gamma_dot_cuasi", "gamma_dot_transicion"]

MIN_N = 4  # mínimo de pares para calcular un Spearman con algún sentido


def spearman(x, y):
    """Spearman robusto a NaN. Devuelve (rho, p, n_efectivo)."""
    s = pd.concat([pd.Series(np.asarray(x, float)),
                   pd.Series(np.asarray(y, float))], axis=1).dropna()
    if len(s) < MIN_N:
        return np.nan, np.nan, len(s)
    r, p = stats.spearmanr(s.iloc[:, 0], s.iloc[:, 1])
    return r, p, len(s)


def spearman_parcial(x, y, z):
    """Spearman entre x,y controlando z (residuos de rangos). Robusto a NaN."""
    s = pd.concat([pd.Series(np.asarray(x, float)),
                   pd.Series(np.asarray(y, float)),
                   pd.Series(np.asarray(z, float))], axis=1).dropna()
    if len(s) < MIN_N:
        return np.nan, np.nan, len(s)
    rx = stats.rankdata(s.iloc[:, 0])
    ry = stats.rankdata(s.iloc[:, 1])
    rz = stats.rankdata(s.iloc[:, 2])
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = stats.pearsonr(ex, ey)
    return r, p, len(s)


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def viga_fiable(csv_path):
    d = pd.read_csv(csv_path)
    return d[d.zona.str.startswith("Vf") & d.fiable].copy()


# ----------------------------------------------------------------------
# Carga: versión con exclusiones (principal) y sin (sensibilidad)
# ----------------------------------------------------------------------
v = viga_fiable(CSV)
tiene_sin = os.path.exists(CSV_SIN)
v_sin = viga_fiable(CSV_SIN) if tiene_sin else None

banner("DISEÑO")
d_full = pd.read_csv(CSV)
print(d_full.groupby(["reologia", "fibras"]).size().to_string())
print(f"\nZonas de viga fiables: N = {len(v)}")
print("ADVERTENCIA: cada fila es una celda (toma x zona), NO una réplica "
      "independiente.\nEl n efectivo es el número de TOMAS por reología "
      "(~6), no el de filas.\nLos p-valores están inflados; el peso está en "
      "la consistencia de SIGNO\nentre reologías, no en cruzar p<0.05.")

# ----------------------------------------------------------------------
banner("1. POR QUÉ HAY QUE ESTRATIFICAR")
for reo, s in v.groupby("reologia"):
    print(f"  {reo}: V_tr mediana={s.V_transicion.median():7.2f}  "
          f"orden_S medio={s.orden_S.mean():.3f}")
for col in ["V_transicion", "orden_S"]:
    a = v[v.reologia == "car-02"][col].dropna()
    b = v[v.reologia == "car-05"][col].dropna()
    if len(a) >= MIN_N and len(b) >= MIN_N:
        _, p = stats.mannwhitneyu(a, b)
        print(f"  Mann-Whitney {col:15s} p={p:.6f}  (n={len(a)}/{len(b)})")
    else:
        print(f"  Mann-Whitney {col:15s} n insuficiente")
print("\n  Si las reologías ocupan rangos disjuntos, agregarlas induce Simpson.")

# ----------------------------------------------------------------------
banner("2. SPEARMAN vs orden_S: estratificado y agregado (CON exclusiones)")
print(f"{'predictor':<22}{'car-02':>20}{'car-05':>20}{'AGREGADO':>20}")
for p_ in PRED:
    fila = []
    for reo in ["car-02", "car-05"]:
        s = v[v.reologia == reo]
        r, pv, n = spearman(s[p_], s.orden_S)
        fila.append(f"{r:+.3f}(p{pv:.3f},n{n})" if not np.isnan(r)
                    else f"n={n} insuf")
    r, pv, n = spearman(v[p_], v.orden_S)
    fila.append(f"{r:+.3f}(p{pv:.3f},n{n})" if not np.isnan(r)
                else f"n={n} insuf")
    print(f"{p_:<22}{fila[0]:>20}{fila[1]:>20}{fila[2]:>20}")
print("\n  Un rho AGREGADO grande que se desvanece o cambia de signo dentro de\n"
      "  cada reología es un artefacto de agregación (Simpson): NO reportarlo\n"
      "  como hallazgo.")

# ----------------------------------------------------------------------
banner("3. LO QUE SÍ SOBREVIVE: mismo signo en ambas reologías")
sobreviven = []
for p_ in PRED:
    resultados = {}
    for reo in ["car-02", "car-05"]:
        s = v[v.reologia == reo]
        r, pv, n = spearman(s[p_], s.orden_S)
        rp, pp, npar = spearman_parcial(s[p_], s.orden_S, s.n_fibras)
        resultados[reo] = (r, pv, n, rp, pp)
    r02 = resultados["car-02"][0]
    r05 = resultados["car-05"][0]
    if np.isnan(r02) or np.isnan(r05):
        continue
    if np.sign(r02) != np.sign(r05):
        continue
    sobreviven.append(p_)
    print(f"\n{p_}:  (rho car-02={r02:+.3f}, car-05={r05:+.3f} — mismo signo)")
    ps = []
    for reo in ["car-02", "car-05"]:
        r, pv, n, rp, pp = resultados[reo]
        ps.append(pv)
        print(f"  {reo}: N={n:2d}  rho={r:+.3f} (p={pv:.4f})   "
              f"parcial|n_fibras={rp:+.3f} (p={pp:.3f})")
    ps_val = [p for p in ps if not np.isnan(p)]
    if len(ps_val) == 2:
        _, pc = stats.combine_pvalues(ps_val, method="fisher")
        print(f"  Fisher combinado: p={pc:.4f}"
              f"{'   <-- combinado < 0.05' if pc < 0.05 else ''}")

if not sobreviven:
    print("\n  Ningún predictor conserva el mismo signo en ambas reologías.")

# ----------------------------------------------------------------------
banner("3b. SENSIBILIDAD: mismos Spearman SIN el criterio de exclusión")
if not tiene_sin:
    print(f"  No se encontró {CSV_SIN}. Corre construir_tabla_zonas_todas.py "
          "con\n  GUARDAR_VERSION_SIN_EXCLUIR=True para habilitar este chequeo.")
else:
    print("  El criterio de exclusión (d de Cohen >= 0.5) inserta NaN en "
          "celdas\n  no comparables con el base. Aquí se recalcula todo SIN "
          "ese filtro.\n")
    print(f"{'predictor':<22}{'car-02 con/sin':>26}{'car-05 con/sin':>26}")
    for p_ in PRED:
        celdas = []
        for reo in ["car-02", "car-05"]:
            r_con, _, n_con = spearman(v[v.reologia == reo][p_],
                                       v[v.reologia == reo].orden_S)
            r_sin, _, n_sin = spearman(v_sin[v_sin.reologia == reo][p_],
                                       v_sin[v_sin.reologia == reo].orden_S)
            celdas.append(f"{r_con:+.2f}(n{n_con}) / {r_sin:+.2f}(n{n_sin})")
        print(f"{p_:<22}{celdas[0]:>26}{celdas[1]:>26}")
    print("\n  Lectura: si el signo se mantiene entre 'con' y 'sin', el hallazgo "
          "es\n  robusto a la decisión de exclusión. Si cambia, la conclusión "
          "depende\n  del umbral y hay que decirlo explícitamente en la memoria.")

# ----------------------------------------------------------------------
banner("INTERPRETACIÓN (teoría de Jeffery)")
print("""  Predicción de Jeffery (1922): la parte antisimétrica del gradiente de
  velocidad (vorticidad, omega) hace rotar la fibra en órbitas periódicas,
  lo que impide que se estabilice en una orientación preferente. La parte
  simétrica (tasa de deformación) es la que alinea.

  => Se espera que omega (sobre todo en transición) se asocie NEGATIVAMENTE
     con el orden_S final. Contrastar el signo observado arriba contra esta
     predicción, en ambas reologías, es la evidencia física central.""")

# ----------------------------------------------------------------------
banner("4. LO QUE NO SE SOSTIENE: colinealidad y modelos multivariados")
try:
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    v_cc = v.dropna(subset=PRED)
    n_drop = len(v) - len(v_cc)
    print(f"  Filas con predictores completos: {len(v_cc)}/{len(v)} "
          f"({n_drop} descartadas por NaN de exclusión).")
    if len(v_cc) < len(PRED) + 2:
        print("  Insuficientes filas completas para VIF/RF estables; omitido.")
    else:
        X = StandardScaler().fit_transform(v_cc[PRED])
        print("\n  VIF (colinealidad):")
        for j, p_ in enumerate(PRED):
            otros = [k for k in range(len(PRED)) if k != j]
            r2 = LinearRegression().fit(X[:, otros], X[:, j]).score(
                X[:, otros], X[:, j])
            vif = 1 / (1 - r2) if r2 < 0.999 else np.inf
            flag = "SEVERA" if vif > 10 else ("alta" if vif > 5 else "")
            print(f"    {p_:24s} VIF={vif:8.2f}  {flag}")

        rf = RandomForestRegressor(n_estimators=300, random_state=0)
        cv = cross_val_score(rf, v_cc[PRED].values, v_cc.orden_S.values,
                             cv=KFold(5, shuffle=True, random_state=0),
                             scoring="r2")
        rf.fit(v_cc[PRED].values, v_cc.orden_S.values)
        print(f"\n  Random Forest (AGREGADO): "
              f"R2_train={rf.score(v_cc[PRED].values, v_cc.orden_S.values):+.3f}"
              f"   R2_CV={cv.mean():+.3f}")
        print("    OJO: un R2_CV>0 aquí NO valida el modelo. El RF puede estar")
        print("    aprendiendo a separar reologías (que difieren en todo), no la")
        print("    relación flujo->orientación. Evaluar DENTRO de cada estrato.")

        print("\n  Random Forest (POR REOLOGÍA — el test honesto):")
        for reo in ["car-02", "car-05"]:
            s = v_cc[v_cc.reologia == reo]
            if len(s) < 10:
                print(f"    {reo}: N={len(s)} insuficiente")
                continue
            k = min(5, len(s) // 3)
            cv = cross_val_score(
                RandomForestRegressor(n_estimators=300, random_state=0),
                s[PRED].values, s.orden_S.values,
                cv=KFold(k, shuffle=True, random_state=0), scoring="r2")
            print(f"    {reo}: N={len(s):2d}  R2_CV={cv.mean():+.3f}"
                  f"{'   <-- peor que la media' if cv.mean() <= 0 else ''}")
except ImportError:
    print("  (sklearn no disponible; omitido)")

print("\n  Regla: regresión múltiple descartada si VIF > 10. Random Forest se")
print("  evalúa con R2_CV DENTRO de cada reología, nunca sobre el agregado.\n")
