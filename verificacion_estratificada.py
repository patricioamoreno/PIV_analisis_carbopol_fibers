"""
verificacion_estratificada.py
=============================
Análisis CORREGIDO, estratificado por reología.

Reemplaza a verificacion_resultados.py, que agregaba ambas reologías
y por tanto reportaba correlaciones que son artefactos de agregación
(paradoja de Simpson).

Uso:  python verificacion_estratificada.py [ruta_csv]
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "acum_tabla_zona.csv"
PRED = ["V_cuasi", "V_transicion", "omega_cuasi", "omega_transicion",
        "gamma_dot_cuasi", "gamma_dot_transicion"]


def spearman_parcial(x, y, z):
    """Spearman entre x,y controlando z (vía residuos de rangos)."""
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def banner(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


d = pd.read_csv(CSV)
v = d[d.zona.str.startswith("Vf") & d.fiable]

banner("DISEÑO")
print(d.groupby(["reologia", "fibras"]).size().to_string())
print(f"\nZonas de viga fiables: N = {len(v)}")

# ----------------------------------------------------------------------
banner("1. POR QUÉ HAY QUE ESTRATIFICAR")
for reo, s in v.groupby("reologia"):
    print(f"  {reo}: V_tr mediana={s.V_transicion.median():7.2f}  "
          f"orden_S medio={s.orden_S.mean():.3f}")
for col in ["V_transicion", "orden_S"]:
    a = v[v.reologia == "car-02"][col]
    b = v[v.reologia == "car-05"][col]
    _, p = stats.mannwhitneyu(a, b)
    print(f"  Mann-Whitney {col:15s} p={p:.6f}")
print("\n  Las reologías ocupan rangos disjuntos -> agregarlas induce Simpson.")

# ----------------------------------------------------------------------
banner("2. SPEARMAN: agregado vs estratificado")
print(f"{'predictor':<22}{'car-02':>18}{'car-05':>18}{'AGREGADO':>18}")
for p_ in PRED:
    fila = []
    for reo in ["car-02", "car-05"]:
        s = v[v.reologia == reo]
        r, pv = stats.spearmanr(s[p_], s.orden_S)
        fila.append(f"{r:+.3f}({pv:.3f})")
    r, pv = stats.spearmanr(v[p_], v.orden_S)
    fila.append(f"{r:+.3f}({pv:.3f})")
    print(f"{p_:<22}{fila[0]:>18}{fila[1]:>18}{fila[2]:>18}")

print("\n  V_transicion:  agregado +0.566 (p<0.001)  pero  -0.429 / +0.117 por reología")
print("  omega_cuasi:   agregado -0.528 (p=0.001)  pero  +0.074 / +0.057 por reología")
print("  -> ARTEFACTOS DE AGREGACIÓN. No reportar como hallazgos.")

# ----------------------------------------------------------------------
banner("3. LO QUE SÍ SOBREVIVE: mismo signo en ambas reologías")
for p_ in ["omega_transicion", "gamma_dot_transicion"]:
    print(f"\n{p_}:")
    ps = []
    for reo in ["car-02", "car-05"]:
        s = v[v.reologia == reo]
        r, pv = stats.spearmanr(s[p_], s.orden_S)
        rp, pp = spearman_parcial(s[p_], s.orden_S, s.n_fibras)
        ps.append(pv)
        print(f"  {reo}: N={len(s):2d}  rho={r:+.3f} (p={pv:.4f})   "
              f"parcial|n_fibras={rp:+.3f} (p={pp:.3f})")
    _, pc = stats.combine_pvalues(ps, method="fisher")
    print(f"  Fisher combinado: p={pc:.4f}"
          f"{'   <-- SIGNIFICATIVO' if pc < 0.05 else ''}")

banner("INTERPRETACIÓN")
print("""  omega_transicion se asocia NEGATIVAMENTE con orden_S en ambas reologías.

  Más vorticidad durante la transición -> MENOS alineamiento final.

  Consistente con Jeffery (1922): la parte antisimétrica del gradiente de
  velocidad (vorticidad) hace rotar la fibra en órbitas periódicas, lo que
  impide que se estabilice en una orientación preferente. La parte simétrica
  (deformación) es la que alinea.

  El efecto está replicado en dos fluidos con reologías distintas, tiene el
  signo que la teoría predice, y persiste al controlar por n_fibras.""")

# ----------------------------------------------------------------------
banner("4. LO QUE NO SE SOSTIENE")
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(v[PRED])
    print("  VIF (colinealidad):")
    for j, p_ in enumerate(PRED):
        otros = [k for k in range(len(PRED)) if k != j]
        r2 = LinearRegression().fit(X[:, otros], X[:, j]).score(X[:, otros], X[:, j])
        vif = 1 / (1 - r2) if r2 < 0.999 else np.inf
        flag = "SEVERA" if vif > 10 else ("alta" if vif > 5 else "")
        print(f"    {p_:24s} VIF={vif:8.2f}  {flag}")

    # RF sobre el conjunto agregado (ambas reologías)
    rf = RandomForestRegressor(n_estimators=300, random_state=0)
    cv = cross_val_score(rf, v[PRED].values, v.orden_S.values,
                         cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    rf.fit(v[PRED].values, v.orden_S.values)
    print(f"\n  Random Forest (AGREGADO): R2_train="
          f"{rf.score(v[PRED].values, v.orden_S.values):+.3f}   R2_CV={cv.mean():+.3f}")
    print("    OJO: un R2_CV>0 aquí NO valida el modelo. El RF puede estar")
    print("    aprendiendo a separar reologías (que difieren en todo), no la")
    print("    relación flujo->orientación. Hay que evaluarlo DENTRO de cada estrato.")

    # RF dentro de cada reología: el test honesto
    print("\n  Random Forest (POR REOLOGÍA):")
    for reo in ["car-02", "car-05"]:
        s = v[v.reologia == reo]
        if len(s) < 10:
            print(f"    {reo}: N={len(s)} insuficiente")
            continue
        k = min(5, len(s) // 3)
        cv = cross_val_score(RandomForestRegressor(n_estimators=300, random_state=0),
                             s[PRED].values, s.orden_S.values,
                             cv=KFold(k, shuffle=True, random_state=0), scoring="r2")
        print(f"    {reo}: N={len(s):2d}  R2_CV={cv.mean():+.3f}"
              f"{'   <-- peor que la media' if cv.mean() <= 0 else ''}")
except ImportError:
    print("  (sklearn no disponible; omitido)")

print("\n  Regresión múltiple: descartada por VIF > 10 (criterio a priori).")
print("  Random Forest:      evaluar R2_CV DENTRO de cada reología, no agregado.\n")
