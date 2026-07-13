"""
analizar_adveccion.py
=====================
Analiza los caches de adveccion (los .npz que produce construir_caches_adveccion)
de TODAS las tomas y responde: que zona influye mas en la orientacion final de
las fibras, segun la velocidad (V), el giro (omega) y gamma_dot que la fibra
experimento EN esa zona a lo largo de su trayectoria reconstruida.

Se ejecuta DESPUES de construir_caches_adveccion.py.

Estilo del proyecto: config arriba, lee de una carpeta de caches, guarda CSV +
figuras. No recalcula trayectorias (eso ya esta cacheado).

Que hace, paso a paso:
  1. Junta la influencia por (fibra, zona) de todas las tomas (loader gemelo).
  2. Anota reologia y fibras desde el codigo de toma.
  3. Resumen por zona: cuanto tiempo pasan las fibras y bajo que flujo.
  4. Correlacion (Spearman) entre el flujo experimentado en cada zona y el
     theta final de la fibra -> que zona/variable esta ligada a la orientacion.
  5. Figuras: exposicion por zona + heatmap de influencia (global y por reologia).

Salidas:
  adv_resumen_zonas.csv          exposicion media de las fibras por zona
  adv_influencia_global.csv      rho(flujo en zona vs theta), todas las tomas
  adv_influencia_por_reologia.csv  lo mismo, separado car-02 / car-05
  fig_adv_exposicion.png         fraccion de tiempo y V medio por zona
  fig_adv_influencia.png         heatmap zona x (V,omega,gamma_dot,tiempo)
  fig_adv_influencia_reologia.png  heatmap separado por reologia

Uso:
    python analizar_adveccion.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from construir_caches_adveccion import cargar_cache_adveccion, CACHE_ADV_DIR

# ============================================================
# CONFIGURACION — editar aqui
# ============================================================

# Carpeta con los caches de adveccion (la que produce el script anterior).
CACHE_ADV = CACHE_ADV_DIR         # normalmente "cache_adveccion"

# Solo considerar la influencia de zonas de la VIGA (el profesor: la orientacion
# solo importa donde afecta la resistencia). Pon False para incluir la L tambien.
SOLO_VIGA = False

# minimo de fibras en una zona para calcular correlacion de esa zona
MIN_FIBRAS_ZONA = 5

FACTORES = [("V", "V_med"), ("omega", "omega_med"),
            ("gamma_dot", "gamma_dot_med"), ("t_en_zona", "frac_tiempo")]
ETIQ = {"V": "V", "omega": "ω", "gamma_dot": "γ̇", "t_en_zona": "tiempo"}


# ============================================================
# UTILIDADES
# ============================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


def _meta_toma(cod_o_nombre):
    car = re.search(r"car-?(\d+)", str(cod_o_nombre).lower())
    fib = re.search(r"n-?(\d+)", str(cod_o_nombre).lower())
    return (f"car-{car.group(1)}" if car else None,
            int(fib.group(1)) if fib else None)


def cargar_todas_las_tomas():
    """
    Lee todos los caches de adveccion de la carpeta y los junta en un solo
    DataFrame con columnas (fibra_id, zona, ..., theta, toma, reologia, fibras).
    La reologia/fibras se leen del METADATO del .npz si esta, o del nombre.
    """
    archivos = sorted(glob.glob(os.path.join(CACHE_ADV, "*__adveccion.npz")),
                      key=natural_sort_key)
    if not archivos:
        return None
    filas = []
    for path in archivos:
        d = np.load(path, allow_pickle=True)
        cod = str(d["toma"]) if "toma" in d.files else \
            os.path.basename(path).split("__")[0]
        df = cargar_cache_adveccion(cod)
        if df is None or len(df) == 0:
            continue
        # reologia/fibras: se guardaron en el metadato del cache de adveccion.
        reo = str(d["reologia"]) if "reologia" in d.files and str(d["reologia"]) else None
        fib = int(d["conc_fibras"]) if "conc_fibras" in d.files else None
        if reo is None:                       # respaldo: del nombre/codigo
            reo, fib = _meta_toma(os.path.basename(path))
        df["toma"] = cod
        df["reologia"] = reo
        df["fibras"] = fib
        # ID global unico (la numeracion de fibra se repite entre tomas)
        df["fibra_uid"] = cod + "_" + df["fibra_id"].astype(str)
        filas.append(df)
        print(f"  {cod}: {len(df)} filas fibra×zona  ({reo}, {fib} fibras)")
    if not filas:
        return None
    out = pd.concat(filas, ignore_index=True)
    if SOLO_VIGA:
        out = out[out["zona"].str.startswith("V")].copy()
        print(f"\n[SOLO_VIGA] quedan {len(out)} filas (zonas de viga)")
    return out


# ============================================================
# ANALISIS
# ============================================================

def resumen_por_zona(df):
    return (df.groupby("zona")
            .agg(tomas=("toma", "nunique"),
                 filas=("fibra_uid", "size"),
                 frac_tiempo_med=("frac_tiempo", "mean"),
                 V_med=("V_med", "mean"),
                 omega_med=("omega_med", "mean"),
                 gamma_dot_med=("gamma_dot_med", "mean"))
            .reset_index()
            .sort_values("frac_tiempo_med", ascending=False))


def correlacion_influencia(df, etiqueta=""):
    """
    Para cada zona y factor, Spearman entre el valor experimentado en esa zona
    y el theta final de la fibra. Rho alto => lo que pasa en esa zona esta
    ligado a como queda orientada la fibra.
    """
    filas = []
    for z in sorted(df["zona"].unique()):
        sub = df[df.zona == z]
        if sub["fibra_uid"].nunique() < MIN_FIBRAS_ZONA:
            continue
        theta = sub["theta"].to_numpy(float)
        for fac, col in FACTORES:
            x = sub[col].to_numpy(float)
            m = ~(np.isnan(x) | np.isnan(theta))
            if m.sum() < MIN_FIBRAS_ZONA or np.std(x[m]) == 0:
                rho, pv = np.nan, np.nan
            else:
                rho, pv = stats.spearmanr(x[m], theta[m])
            filas.append({"grupo": etiqueta or "global", "zona": z,
                          "factor": fac, "rho_vs_theta": rho,
                          "p_value": pv, "n_fibras": int(m.sum())})
    res = pd.DataFrame(filas)
    if len(res):
        res["abs_rho"] = res["rho_vs_theta"].abs()
    return res


# ============================================================
# FIGURAS
# ============================================================

def fig_exposicion(resumen, path="fig_adv_exposicion.png"):
    r = resumen.sort_values("zona")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.bar(r["zona"], r["frac_tiempo_med"], color="#4a7ab5")
    ax1.set_title("Fracción de tiempo que las fibras pasan en cada zona\n"
                  "(en su trayectoria reconstruida)", fontsize=11,
                  fontweight="bold")
    ax1.set_ylabel("fracción de tiempo media")
    ax1.grid(axis="y", ls=":", alpha=0.5)
    ax2.bar(r["zona"], r["V_med"], color="#d4703a")
    ax2.set_title("Velocidad media experimentada por zona", fontsize=11,
                  fontweight="bold")
    ax2.set_ylabel("V media [mm/s]")
    ax2.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)


def _heatmap(ax, corr, titulo):
    piv = corr.pivot_table(index="zona", columns="factor",
                           values="rho_vs_theta")
    orden = ["V", "omega", "gamma_dot", "t_en_zona"]
    piv = piv.reindex(columns=[c for c in orden if c in piv.columns])
    M = piv.to_numpy()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([ETIQ.get(c, c) for c in piv.columns], fontsize=12)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        color="white" if abs(M[i, j]) > 0.5 else "black",
                        fontsize=10, fontweight="bold")
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    return im


def fig_influencia(corr, path="fig_adv_influencia.png"):
    if corr.empty:
        print("  [omitido] fig_adv_influencia: sin datos")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = _heatmap(ax, corr, "Influencia de cada zona en la orientación final\n"
                  "ρ entre el flujo experimentado en la zona y θ (todas las tomas)")
    fig.colorbar(im, ax=ax, label="ρ Spearman")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)


def fig_influencia_reologia(df, path="fig_adv_influencia_reologia.png"):
    reos = sorted([r for r in df["reologia"].dropna().unique()])
    if len(reos) < 2:
        print("  [omitido] fig_adv_influencia_reologia: <2 reologias")
        return None
    fig, axes = plt.subplots(1, len(reos), figsize=(7 * len(reos), 5))
    if len(reos) == 1:
        axes = [axes]
    todas = []
    for ax, reo in zip(axes, reos):
        c = correlacion_influencia(df[df.reologia == reo], etiqueta=reo)
        todas.append(c)
        if c.empty:
            ax.set_title(f"{reo}: sin datos"); continue
        im = _heatmap(ax, c, f"Influencia por zona — {reo}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="ρ")
    fig.suptitle("¿Qué zona orienta las fibras, según reología?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  guardado:", path)
    return pd.concat(todas, ignore_index=True) if todas else None


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Cargando caches de adveccion...")
    df = cargar_todas_las_tomas()
    if df is None:
        print(f"[ERROR] No hay caches en '{CACHE_ADV}/'. "
              f"Corre antes construir_caches_adveccion.py")
        raise SystemExit

    n_fib = df["fibra_uid"].nunique()
    print(f"\nTotal: {len(df)} filas fibra×zona, {n_fib} fibras, "
          f"{df['toma'].nunique()} tomas\n")

    print("Resumen por zona:")
    resumen = resumen_por_zona(df)
    print(resumen.round(3).to_string(index=False))
    resumen.to_csv("adv_resumen_zonas.csv", index=False)

    print("\nCorrelacion influencia (global):")
    corr = correlacion_influencia(df)
    if not corr.empty:
        print(corr.sort_values("abs_rho", ascending=False)
              .head(8)[["zona", "factor", "rho_vs_theta", "p_value", "n_fibras"]]
              .to_string(index=False))
    corr.to_csv("adv_influencia_global.csv", index=False)

    print("\nGenerando figuras...")
    fig_exposicion(resumen)
    fig_influencia(corr)
    corr_reo = fig_influencia_reologia(df)
    if corr_reo is not None:
        corr_reo.to_csv("adv_influencia_por_reologia.csv", index=False)

    print("\nListo. Revisa adv_*.csv y fig_adv_*.png")
