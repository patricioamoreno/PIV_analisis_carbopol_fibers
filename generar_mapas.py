"""
generar_mapas.py
================
Lee acum_tabla_zona.csv (tomas x zonas) y genera TODAS las figuras de la
memoria: mapas fisicos, correlaciones, y analisis por reologia/fibras.

Figuras de zona/correlacion:
  1) fig_panorama_heatmap.png   - heatmap de rho (predictor x etapa) por respuesta
  2) fig_scatter_V_ordenS.png   - scatter V(transicion) vs orden_S + tendencia
  3) fig_barras_temporal.png    - |rho| transicion vs cuasi por predictor
  4) fig_mapa_viga_L.png        - mapa fisico de las zonas coloreadas por orden_S
                                  y por dispersion, con geometria real del montaje
  5) fig_comparacion_L_viga.png - correlaciones calculadas POR SEPARADO (L vs viga)
  6) fig_mapa_por_reologia.png  - mapa fisico de orden_S separado por reologia

Figuras por factor (reologia y concentracion de fibras):
  7) fig_box_reologia.png       - orden_S y sigma por reologia
  8) fig_box_fibras.png         - orden_S y sigma por concentracion de fibras
  9) fig_interaccion.png        - orden_S medio por celda reologia x fibras
 10) fig_corr_por_grupo.png     - rho(V, orden_S) global y por cada factor
     tabla_por_celda.csv        - resumen numerico por celda del diseno

Uso:
    python generar_mapas.py                      # usa acum_tabla_zona.csv
    python generar_mapas.py ruta/a/tu_tabla.csv

IMPORTANTE: las figuras por factor (7-10) y el mapa por reologia (6) requieren
que el CSV tenga las columnas 'reologia' y 'fibras'. Estas las agrega
automaticamente la version nueva de acumular_tomas.py al generar el CSV. Si el
CSV es viejo y no las trae, esas figuras se omiten con un aviso.

No recalcula el pipeline: solo lee el CSV ya generado.
Requiere: pandas, numpy, scipy, matplotlib.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib import cm
from matplotlib.colors import Normalize

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
OUTPUT_FIGS = "figs_memoria"
os.makedirs(OUTPUT_FIGS, exist_ok=True)
PREDICTORES = ["V", "omega", "gamma_dot"]
ETAPAS = ["transicion", "cuasi"]
RESPUESTAS = ["orden_S", "sigma_iso"]
ETIQ_PRED = {"V": r"$V$", "omega": r"$\omega$", "gamma_dot": r"$\dot{\gamma}$"}
MIN_FIB = 5   # zonas con menos fibras no son fiables para orden_S

# Geometria EXACTA de definir_zonas.py (poligonos de 4 vertices).
# Las zonas L son paralelogramos inclinados 30°; la viga, rectangulos.
POLY = {
  "Z1":    [[40.7,111.0],[114.3,68.5],[243.8,292.8],[170.2,335.3]],
  "Z2":    [[23.2,80.7],[109.8,30.7],[127.3,61.0],[40.7,111.0]],
  "Z3":    [[154.0,5.2],[164.0,22.5],[119.8,48.0],[109.8,30.7]],
  "Vf1c1": [[151.0,0.0],[251.0,0.0],[251.0,-37.5],[151.0,-37.5]],
  "Vf1c2": [[251.0,0.0],[351.0,0.0],[351.0,-37.5],[251.0,-37.5]],
  "Vf1c3": [[351.0,0.0],[451.0,0.0],[451.0,-37.5],[351.0,-37.5]],
  "Vf2c1": [[151.0,-37.5],[251.0,-37.5],[251.0,-75.0],[151.0,-75.0]],
  "Vf2c2": [[251.0,-37.5],[351.0,-37.5],[351.0,-75.0],[251.0,-75.0]],
  "Vf2c3": [[351.0,-37.5],[451.0,-37.5],[451.0,-75.0],[351.0,-75.0]],
}
GEO = {z: {"cx": float(np.mean([p[0] for p in v])),
           "cy": float(np.mean([p[1] for p in v])),
           "poly": v} for z, v in POLY.items()}
ZONAS_VIGA = ["Vf1c1", "Vf1c2", "Vf2c1", "Vf2c2", "Vf2c3"]
ZONAS_L = ["Z1", "Z2", "Z3"]


# ----------------------------------------------------------------------
# Correlaciones a partir de la tabla (Spearman entre zonas)
# ----------------------------------------------------------------------
def correlaciones(tabla, zonas=None):
    """rho de Spearman predictor(etapa) vs respuesta, sobre las filas fiables."""
    t = tabla[tabla["fiable"]].copy()
    if zonas is not None:
        t = t[t["zona"].isin(zonas)]
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
                              "rho": rho, "p_value": pv, "n": n})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# FIGURA 1 — heatmap panorama
# ----------------------------------------------------------------------
def fig_heatmap(corr, path=f"{OUTPUT_FIGS}/fig_panorama_heatmap.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, resp in zip(axes, RESPUESTAS):
        sub = corr[corr.respuesta == resp]
        M = (sub.pivot_table(index="predictor", columns="etapa", values="rho")
                .reindex(index=PREDICTORES, columns=ETAPAS).to_numpy())
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ETAPAS)))
        ax.set_xticklabels(["transición", "cuasi"])
        ax.set_yticks(range(len(PREDICTORES)))
        ax.set_yticklabels([ETIQ_PRED[p] for p in PREDICTORES], fontsize=13)
        titulo = "Orientación (orden S)" if resp == "orden_S" else "Dispersión (σ)"
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                            color="white" if abs(M[i, j]) > 0.5 else "black",
                            fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ Spearman")
    fig.suptitle("Correlación flujo–fibras entre zonas (12 tomas)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)


# ----------------------------------------------------------------------
# FIGURA 2 — scatter V vs orden_S (el hallazgo principal)
# ----------------------------------------------------------------------
def fig_scatter(tabla, path=f"{OUTPUT_FIGS}/fig_scatter_V_ordenS.png"):
    t = tabla[tabla["fiable"]].dropna(subset=["V_transicion", "orden_S"])
    x = t["V_transicion"].to_numpy(float)
    y = t["orden_S"].to_numpy(float)
    rho, pv = stats.spearmanr(x, y)

    # color por geometria: viga vs L
    es_l = t["zona"].isin(ZONAS_L).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x[~es_l], y[~es_l], s=70, c="#2c6fbb", edgecolor="white",
               linewidth=0.8, label="Viga", zorder=3)
    ax.scatter(x[es_l], y[es_l], s=70, c="#d98032", edgecolor="white",
               linewidth=0.8, label="L (conducto)", zorder=3)
    # tendencia (ajuste lineal simple, solo visual)
    if len(x) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a + b * xs, "--", color="#444", lw=2,
                label="tendencia", zorder=2)
    ax.set_xlabel("Velocidad en transición  $V$  [mm/s]", fontsize=12)
    ax.set_ylabel("Orden-parámetro de orientación  $S$", fontsize=12)
    ax.set_title(f"Velocidad en transición vs alineamiento de fibras\n"
                 f"ρ = {rho:+.2f}   (p = {pv:.1e},  n = {len(x)})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)


# ----------------------------------------------------------------------
# FIGURA 3 — barras temporales
# ----------------------------------------------------------------------
def fig_barras(corr, path=f"{OUTPUT_FIGS}/fig_barras_temporal.png"):
    sub = corr[corr.respuesta == "orden_S"]
    piv = (sub.pivot_table(index="predictor", columns="etapa", values="rho")
              .reindex(index=PREDICTORES, columns=ETAPAS))
    x = np.arange(len(PREDICTORES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, piv["transicion"].abs(), w, label="transición",
           color="#2c6fbb")
    ax.bar(x + w/2, piv["cuasi"].abs(), w, label="cuasi", color="#d98032")
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQ_PRED[p] for p in PREDICTORES], fontsize=14)
    ax.set_ylabel("|ρ| con el alineamiento (orden S)", fontsize=12)
    ax.set_title("¿En qué etapa se define la orientación?\n"
                 "La asociación es mayor en transición para las 3 variables",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    for i, p in enumerate(PREDICTORES):
        for off, e in [(-w/2, "transicion"), (w/2, "cuasi")]:
            val = abs(piv.loc[p, e])
            if not np.isnan(val):
                ax.text(i + off, val + 0.01, f"{val:.2f}", ha="center",
                        fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)


# ----------------------------------------------------------------------
# FIGURA 4 — mapa fisico de la viga + L
# ----------------------------------------------------------------------
def _resumen_zona(tabla):
    """Promedio por zona (sobre tomas) de orden_S y sigma_iso, solo fiables."""
    t = tabla[tabla["fiable"]]
    return t.groupby("zona").agg(
        orden_S=("orden_S", "mean"),
        sigma_iso=("sigma_iso", "mean"),
        n=("orden_S", "size")).to_dict("index")


def _dibujar_mapa(ax, valores, titulo, cmap, label, vmin, vmax):
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps[cmap]
    for zona, g in GEO.items():
        val = valores.get(zona, {}).get("_v", np.nan)
        color = cmap_obj(norm(val)) if not np.isnan(val) else "#dddddd"
        ax.add_patch(plt.Polygon(g["poly"], closed=True,
                                 facecolor=color, edgecolor="#333",
                                 linewidth=1.5, zorder=2))
        # etiqueta zona + valor
        txt = zona if np.isnan(val) else f"{zona}\n{val:.2f}"
        lum = 0 if np.isnan(val) else norm(val)
        ax.text(g["cx"], g["cy"], txt, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if 0.35 < lum < 0.8 else "black", zorder=3)
    # flecha de flujo L -> viga
    ax.annotate("", xy=(180, -15), xytext=(150, 25),
                arrowprops=dict(arrowstyle="-|>", color="#1a7a3a", lw=2.5),
                zorder=4)
    ax.text(135, 8, "flujo", color="#1a7a3a", fontsize=10,
            style="italic", rotation=-40)
    ax.set_xlim(0, 470)
    ax.set_ylim(-95, 350)
    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.04, label=label)


def fig_mapa(tabla, path=f"{OUTPUT_FIGS}/fig_mapa_viga_L.png"):
    res = _resumen_zona(tabla)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    v1 = {z: {"_v": d["orden_S"]} for z, d in res.items()}
    _dibujar_mapa(ax1, v1, "Orientación: orden-parámetro S por zona\n"
                  "(amarillo = fibras alineadas)", "viridis",
                  "orden S", 0.3, 0.85)

    v2 = {z: {"_v": d["sigma_iso"]} for z, d in res.items()}
    smax = max(d["sigma_iso"] for d in res.values())
    _dibujar_mapa(ax2, v2, "Dispersión: σ de centroides por zona\n"
                  "(rojo = fibras más dispersas)", "YlOrRd",
                  "σ [mm]", 0, smax)

    fig.suptitle("Mapa físico del montaje L → Viga  (promedio de 12 tomas)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)

def fig_mapa_etapas(tabla, predictor="V", path=f"{OUTPUT_FIGS}/fig_mapa_etapas.png"):
    """Dos mapas fisicos lado a lado: el predictor del fluido (por defecto V)
    promedio por zona en TRANSICION vs CUASI. Muestra DONDE y EN QUE ETAPA el
    fluido es mas intenso, para conectar la geografia con la causa temporal."""
    t = tabla[tabla["fiable"]]
    col_t, col_c = f"{predictor}_transicion", f"{predictor}_cuasi"
    if col_t not in t or col_c not in t:
        print(f"  [omitido] fig_mapa_etapas: faltan columnas {col_t}/{col_c}")
        return
    med_t = t.groupby("zona")[col_t].mean().to_dict()
    med_c = t.groupby("zona")[col_c].mean().to_dict()
    # escala comun para que ambos mapas sean comparables
    vals = [v for v in list(med_t.values()) + list(med_c.values())
            if not np.isnan(v)]
    vmax = max(vals) if vals else 1.0
 
    etiq = {"V": "Velocidad V [mm/s]", "omega": "Vorticidad ω [1/s]",
            "gamma_dot": "Tasa γ̇ [1/s]"}.get(predictor, predictor)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _dibujar_mapa(ax1, {z: {"_v": med_t.get(z, np.nan)} for z in GEO},
                  f"{etiq} — TRANSICIÓN", "plasma", etiq, 0, vmax)
    _dibujar_mapa(ax2, {z: {"_v": med_c.get(z, np.nan)} for z in GEO},
                  f"{etiq} — CUASI-ESTACIONARIO", "plasma", etiq, 0, vmax)
    fig.suptitle(f"Campo de fluido por etapa ({etiq})  —  "
                 "misma escala para comparar transición vs cuasi",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)

# ----------------------------------------------------------------------
# FIGURA 5 — comparacion L vs viga (correlaciones por separado)
# ----------------------------------------------------------------------
def fig_comparacion(tabla, path=f"{OUTPUT_FIGS}/fig_comparacion_L_viga.png"):
    corr_viga = correlaciones(tabla, ZONAS_VIGA)
    corr_l = correlaciones(tabla, ZONAS_L)
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (corr, nom, n) in zip(axes, [
            (corr_viga, "VIGA", tabla[tabla.fiable & tabla.zona.isin(ZONAS_VIGA)].shape[0]),
            (corr_l, "L (conducto)", tabla[tabla.fiable & tabla.zona.isin(ZONAS_L)].shape[0])]):
        sub = corr[corr.respuesta == "orden_S"]
        M = (sub.pivot_table(index="predictor", columns="etapa", values="rho")
                .reindex(index=PREDICTORES, columns=ETAPAS).to_numpy())
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(2)); ax.set_xticklabels(["transición", "cuasi"])
        ax.set_yticks(range(3))
        ax.set_yticklabels([ETIQ_PRED[p] for p in PREDICTORES], fontsize=13)
        ax.set_title(f"{nom}  (orden S, n={n} obs.)",
                     fontsize=12, fontweight="bold")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                            color="white" if abs(M[i, j]) > 0.5 else "black",
                            fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ")
    fig.suptitle("¿La relación flujo–orientación es igual en la L que en la viga?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)
    return corr_viga, corr_l


# ----------------------------------------------------------------------
# FIGURA 6 — mapa fisico separado por reologia
# ----------------------------------------------------------------------
def _resumen_zona_filtrado(tabla, mask):
    """orden_S medio por zona, solo filas fiables que cumplen 'mask'."""
    t = tabla[tabla["fiable"] & mask]
    if t.empty:
        return {}
    return t.groupby("zona").agg(orden_S=("orden_S", "mean"),
                                 n=("orden_S", "size")).to_dict("index")


def fig_mapa_por_reologia(tabla, path=f"{OUTPUT_FIGS}/fig_mapa_por_reologia.png"):
    """Un panel de mapa por cada reologia, coloreado por orden_S, para ver si
    el patron espacial de alineamiento cambia entre Carbopol."""
    reologias = sorted(tabla.loc[tabla["fiable"], "reologia"].dropna().unique())
    if not reologias:
        print("  [omitido] fig_mapa_por_reologia: sin columna reologia util")
        return
    n = len(reologias)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
    if n == 1:
        axes = [axes]
    for ax, reo in zip(axes, reologias):
        res = _resumen_zona_filtrado(tabla, tabla["reologia"] == reo)
        vals = {z: {"_v": d["orden_S"]} for z, d in res.items()}
        _dibujar_mapa(ax, vals,
                      f"Orientación (orden S) — {reo}\n(amarillo = alineadas)",
                      "viridis", "orden S", 0.3, 0.85)
    fig.suptitle("Mapa de alineamiento por reología  "
                 "(¿cambia el patrón espacial entre Carbopol?)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)


# ----------------------------------------------------------------------
# FIGURAS 7-10 — analisis por reologia y concentracion de fibras
# ----------------------------------------------------------------------
def _box_por(ax, t, grupo, resp, titulo, ylabel):
    grupos = sorted(t[grupo].dropna().unique())
    datos = [t[t[grupo] == g][resp].dropna().values for g in grupos]
    bp = ax.boxplot(datos, tick_labels=[str(g) for g in grupos],
                    patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    colores = ["#2c6fbb", "#d98032", "#5aa469", "#b5546f"]
    for patch, c in zip(bp["boxes"], colores):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, d in enumerate(datos):
        x = np.random.default_rng(i).normal(i + 1, 0.05, len(d))
        ax.scatter(x, d, s=22, c="#333", alpha=0.6, zorder=3)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel); ax.grid(axis="y", ls=":", alpha=0.5)


def fig_boxplots(tabla, grupo, archivo, nombre_grupo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    tf = tabla[tabla["fiable"]]
    _box_por(ax1, tf, grupo, "orden_S",
             f"Alineamiento (orden S) por {nombre_grupo}", "orden S")
    _box_por(ax2, tf, grupo, "sigma_iso",
             f"Dispersión (σ) por {nombre_grupo}", "σ [mm]")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", archivo)


def fig_interaccion(tabla, archivo=f"{OUTPUT_FIGS}/fig_interaccion.png"):
    tf = tabla[tabla["fiable"]]
    cel = (tf.groupby(["reologia", "fibras"])["orden_S"]
             .agg(["mean", "sem", "size"]).reset_index())
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colores = {"car-02": "#2c6fbb", "car-05": "#d98032"}
    for reo in sorted(cel["reologia"].unique()):
        sub = cel[cel.reologia == reo].sort_values("fibras")
        ax.errorbar(sub["fibras"], sub["mean"], yerr=sub["sem"],
                    marker="o", ms=9, lw=2, capsize=4,
                    color=colores.get(reo, "#555"), label=reo)
        for _, r in sub.iterrows():
            ax.annotate(f"n={int(r['size'])}", (r["fibras"], r["mean"]),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=8, color="#666")
    ax.set_xlabel("Concentración de fibras", fontsize=12)
    ax.set_ylabel("Orden-parámetro de orientación  S  (media)", fontsize=12)
    ax.set_title("Interacción reología × fibras sobre el alineamiento\n"
                 "(líneas separadas = el efecto de las fibras depende del Carbopol)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([750, 1500, 3000])
    ax.legend(title="Reología", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", archivo)
    return cel


def _rho_grupo(t, col_x="V_transicion", col_y="orden_S"):
    x = t[col_x].to_numpy(float); y = t[col_y].to_numpy(float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    rho, pv = stats.spearmanr(x[m], y[m])
    return rho, pv, int(m.sum())


def fig_corr_por_grupo(tabla, archivo=f"{OUTPUT_FIGS}/fig_corr_por_grupo.png"):
    tf = tabla[tabla["fiable"]]
    filas = []
    r, p, n = _rho_grupo(tf)
    filas.append(("TODO", "global", r, p, n))
    for reo in sorted(tf["reologia"].dropna().unique()):
        r, p, n = _rho_grupo(tf[tf.reologia == reo])
        filas.append(("reologia", reo, r, p, n))
    for fib in sorted(tf["fibras"].dropna().unique()):
        r, p, n = _rho_grupo(tf[tf.fibras == fib])
        filas.append(("fibras", str(int(fib)), r, p, n))
    res = pd.DataFrame(filas, columns=["tipo", "grupo", "rho", "p", "n"])

    fig, ax = plt.subplots(figsize=(9, 5))
    etiquetas = [f"{g}\n(n={n})" for g, n in zip(res.grupo, res.n)]
    colores = ["#444" if t == "global" else
               "#2c6fbb" if t == "reologia" else "#d98032"
               for t in res.tipo]
    ax.bar(range(len(res)), res["rho"], color=colores, alpha=0.85)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels(etiquetas, fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("ρ (V transición vs orden_S)", fontsize=11)
    ax.set_title("La relación flujo→alineamiento, desglosada por factor\n"
                 "(* = significativa p<0.05)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    for i, (rho, p) in enumerate(zip(res.rho, res.p)):
        if not np.isnan(rho):
            txt = f"{rho:+.2f}" + ("*" if p < 0.05 else "")
            ax.text(i, rho + (0.02 if rho >= 0 else -0.05), txt,
                    ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", archivo)
    return res


def tiene_factores(tabla):
    """True si el CSV trae las columnas reologia y fibras con datos utiles."""
    return ("reologia" in tabla.columns and "fibras" in tabla.columns
            and tabla["reologia"].notna().any()
            and tabla["fibras"].notna().any())


def analisis_por_factores(tabla):
    """Corre las figuras 6-10 + tabla por celda + resumenes en consola."""
    fig_mapa_por_reologia(tabla)
    fig_boxplots(tabla, "reologia", f"{OUTPUT_FIGS}/fig_box_reologia.png", "reología")
    fig_boxplots(tabla, "fibras", f"{OUTPUT_FIGS}/fig_box_fibras.png",
                 "concentración de fibras")
    fig_interaccion(tabla)
    res_corr = fig_corr_por_grupo(tabla)

    tf = tabla[tabla["fiable"]]
    celda = (tf.groupby(["reologia", "fibras"])
               .agg(n=("orden_S", "size"),
                    orden_S_med=("orden_S", "mean"),
                    orden_S_sd=("orden_S", "std"),
                    sigma_med=("sigma_iso", "mean")).round(3).reset_index())
    celda.to_csv(f"{OUTPUT_FIGS}/tabla_por_celda.csv", index=False)
    print("  guardado: tabla_por_celda.csv")

    print("\n=== Alineamiento medio por REOLOGIA ===")
    print(tf.groupby("reologia")["orden_S"].agg(["mean", "std", "size"])
          .round(3).to_string())
    print("\n=== Alineamiento medio por FIBRAS ===")
    print(tf.groupby("fibras")["orden_S"].agg(["mean", "std", "size"])
          .round(3).to_string())
    print("\n=== rho(V transicion -> orden_S) por grupo ===")
    print(res_corr.to_string(index=False))

    reos = sorted(tf["reologia"].dropna().unique())
    if len(reos) == 2:
        a = tf[tf.reologia == reos[0]]["orden_S"].dropna()
        b = tf[tf.reologia == reos[1]]["orden_S"].dropna()
        if len(a) > 2 and len(b) > 2:
            _, pmw = stats.mannwhitneyu(a, b, alternative="two-sided")
            print(f"\nTest {reos[0]} vs {reos[1]} (orden_S): "
                  f"medias {a.mean():.3f} vs {b.mean():.3f}, p={pmw:.3f}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "acum_tabla_zona.csv"
    tabla = pd.read_csv(csv)
    print(f"Leido {csv}: {len(tabla)} filas, "
          f"{tabla['fiable'].sum()} fiables, "
          f"{tabla['zona'].nunique()} zonas, "
          f"{tabla['toma'].nunique()} tomas")

    corr = correlaciones(tabla)
    print("\nGenerando figuras de zona/correlacion:")
    fig_heatmap(corr)
    fig_scatter(tabla)
    fig_barras(corr)
    fig_mapa(tabla)
    fig_mapa_etapas(tabla, predictor="gamma_dot")
    cv, cl = fig_comparacion(tabla)

    # resumen comparativo en consola
    print("\n--- Comparacion L vs viga (orden_S, V en transicion) ---")
    for nom, c in [("VIGA", cv), ("L", cl)]:
        r = c[(c.respuesta == "orden_S") & (c.predictor == "V") &
              (c.etapa == "transicion")]
        if not r.empty:
            rr = r.iloc[0]
            print(f"  {nom:5s}: rho={rr['rho']:+.2f}  p={rr['p_value']:.3f}  "
                  f"n={rr['n']}")

    # --- analisis por reologia y fibras (requiere columnas en el CSV) ---
    if tiene_factores(tabla):
        print("\nGenerando figuras por reologia y fibras:")
        analisis_por_factores(tabla)
    else:
        print("\n[aviso] El CSV no trae columnas 'reologia'/'fibras' con datos. "
              "Regenera acum_tabla_zona.csv con la version nueva de "
              "acumular_tomas.py para incluir el analisis por factores. "
              "Por ahora se omiten las figuras 6-10.")

    print("\nListo.")


if __name__ == "__main__":
    main()