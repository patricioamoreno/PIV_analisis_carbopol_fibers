"""
run_real.py
===========
Orquestador para DATOS REALES (PIV .npy + fibras CSV + etapas JSON).

Uso:
    python run_real.py \
        --piv datos_reales \
        --fibras datos_reales/fibras/m74-toma-1-n-0750-car-02-ptv.csv \
        [--etapas etapas.json]

Si no se pasan flags, usa los valores por defecto de abajo (la muestra).

Salidas:
    real_tabla_zona.csv          (1 fila por zona: fluido + orientacion + dispersion)
    real_capa1_global.csv        (Spearman predictor-respuesta entre zonas)
    real_capa2_global.csv        (regresion estandarizada por etapa)
    real_capa4_global.csv        (comparativa temporal)
    real_fluido_zona_etapa.csv   (medianas robustas del fluido)
    fig_real_orientacion.png     (orden_S y dispersion por zona)
    fig_real_correlaciones.png   (heatmap rho predictor x respuesta x etapa)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from carga_real import cargar_todo, PREDICTORES, ETAPAS
from analisis_global import (tabla_por_zona, capa1_global, capa2_global,
                             capa4_global, RESPUESTAS)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

DEF_PIV = "cache_zonas"
DEF_FIBRAS = "fibras_ultimo_frame"


def banner(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piv", default=DEF_PIV)
    ap.add_argument("--fibras", default=DEF_FIBRAS)
    ap.add_argument("--etapas", default=None)
    args = ap.parse_args()

    banner("1) CARGA DE DATOS REALES")
    df, diag = cargar_todo(args.piv, args.fibras, args.etapas, verbose=True)
    print("\nFibras por zona:")
    for z, n in sorted(diag["fibras_por_zona"].items(),
                       key=lambda kv: -kv[1] if isinstance(kv[1], int) else 0):
        print(f"   {str(z):8s}: {n}")

    banner("2) TABLA POR ZONA (fluido + orientacion + dispersion)")
    tabla = tabla_por_zona(df)
    cols_show = (["zona", "n_fibras", "orden_S", "theta_med", "sigma_iso"] +
                 [f"{p}_{e}" for e in ETAPAS for p in PREDICTORES
                  if f"{p}_{e}" in tabla])
    print(tabla[cols_show].round(2).to_string(index=False))
    print("\nLectura: orden_S alto => fibras alineadas en esa zona; "
          "sigma_iso alto => centroides dispersos.")

    banner("3) CAPA 1 GLOBAL — Spearman predictor vs respuesta (entre zonas)")
    c1 = capa1_global(tabla)
    for resp in RESPUESTAS:
        print(f"\n--- Respuesta: {resp} (ordenado por |rho|) ---")
        sub = c1[c1.respuesta == resp]
        print(sub[["predictor", "etapa", "rho", "p_value", "n_zonas"]]
              .to_string(index=False))
    n_zonas = tabla["zona"].nunique()
    print(f"\nN = {n_zonas} zonas. EXPLORATORIO: con N pequeño, p>0.05 es "
          "normal; mira el SIGNO y la MAGNITUD de rho, no solo el p.")

    banner("4) CAPA 2 GLOBAL — Regresion estandarizada por etapa")
    c2 = capa2_global(tabla)
    if len(c2):
        for resp in RESPUESTAS:
            sub = c2[c2.respuesta == resp]
            if sub.empty:
                continue
            print(f"\n--- Respuesta: {resp} ---")
            print(sub[["etapa", "predictor", "beta_std", "p_value",
                       "r2_modelo", "VIF", "n_zonas"]].to_string(index=False))
    else:
        print("Insuficientes zonas para regresion multivariada estable "
              f"(se necesitan >= {len(PREDICTORES)+2}; hay {n_zonas}). "
              "Usa la Capa 1 global como evidencia principal.")

    banner("5) CAPA 4 GLOBAL — Comparativa temporal (transicion vs cuasi)")
    c4 = capa4_global(c1)
    print(c4[["respuesta", "predictor", "rho_transicion", "rho_cuasi",
              "delta_abs_rho", "domina_etapa"]].to_string(index=False))

    banner("6) CONCLUSION AUTOMATICA")
    for resp in RESPUESTAS:
        sub = c1[(c1.respuesta == resp) & c1.rho.notna()]
        if sub.empty:
            print(f"  {resp}: sin correlaciones calculables.")
            continue
        best = sub.iloc[0]
        signo = "mayor" if best["rho"] > 0 else "menor"
        print(f"  {resp}: el predictor mas asociado es '{best['predictor']}' "
              f"en etapa '{best['etapa']}' (rho={best['rho']:+.2f}). "
              f"A mayor {best['predictor']}, {signo} {resp}.")

    # graficos
    graficar(tabla, c1)

    # guardar
    tabla.to_csv("real_tabla_zona.csv", index=False)
    c1.to_csv("real_capa1_global.csv", index=False)
    c2.to_csv("real_capa2_global.csv", index=False)
    c4.to_csv("real_capa4_global.csv", index=False)
    diag["tabla_fluido"].to_csv("real_fluido_zona_etapa.csv", index=False)

    banner("ARCHIVOS GUARDADOS")
    for f in ["real_tabla_zona.csv", "real_capa1_global.csv",
              "real_capa2_global.csv", "real_capa4_global.csv",
              "real_fluido_zona_etapa.csv", "fig_real_orientacion.png",
              "fig_real_correlaciones.png"]:
        print("  ", f)


def graficar(tabla, c1):
    # --- Fig 1: orden_S y sigma_iso por zona (gris = no fiable) ---
    t = tabla.sort_values("zona")
    colores_S = ["#4a7ab5" if f else "#b0b0b0" for f in t["fiable"]]
    colores_d = ["#d4703a" if f else "#b0b0b0" for f in t["fiable"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(t["zona"], t["orden_S"], color=colores_S)
    ax1.set_title("Orden-parametro de orientacion S por zona\n"
                  "(1 = alineadas, 0 = isotropico; gris = pocas fibras)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("S"); ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", ls=":", alpha=0.5)
    for i, (s, n) in enumerate(zip(t["orden_S"], t["n_fibras"])):
        ax1.text(i, s + 0.02, f"n={n}", ha="center", fontsize=8)

    ax2.bar(t["zona"], t["sigma_iso"], color=colores_d)
    ax2.set_title("Dispersion de centroides (sigma_iso) por zona\n"
                  "(mayor = fibras mas dispersas; gris = pocas fibras)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("sigma_iso [mm]")
    ax2.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("fig_real_orientacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: heatmap rho (predictor x etapa) para cada respuesta ---
    resp_list = c1["respuesta"].unique()
    fig, axes = plt.subplots(1, len(resp_list),
                             figsize=(6 * len(resp_list), 4.5))
    if len(resp_list) == 1:
        axes = [axes]
    for ax, resp in zip(axes, resp_list):
        sub = c1[c1.respuesta == resp]
        piv = sub.pivot_table(index="predictor", columns="etapa", values="rho")
        piv = piv.reindex(index=PREDICTORES, columns=ETAPAS)
        M = piv.to_numpy()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(ETAPAS)))
        ax.set_xticklabels(ETAPAS)
        ax.set_yticks(range(len(PREDICTORES)))
        ax.set_yticklabels([r"$V$", r"$\omega$", r"$\dot{\gamma}$"])
        ax.set_title(f"rho vs {resp}", fontsize=11, fontweight="bold")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                            fontsize=10,
                            color="white" if abs(M[i, j]) > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman rho")
    fig.suptitle("Correlacion fluido-respuesta entre zonas (Capa 1 global)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("fig_real_correlaciones.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()