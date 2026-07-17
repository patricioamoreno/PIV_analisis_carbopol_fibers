"""
run_adveccion.py
================
Ejecuta la reconstruccion de trayectorias de fibras (adveccion hacia atras)
sobre el campo PIV y mide la INFLUENCIA de cada zona sobre la orientacion final
de las fibras, segun la velocidad, el giro (vorticidad) y gamma_dot que la
fibra experimento en cada zona a lo largo de su recorrido.

Uso:
    python run_adveccion.py --piv caches_zonas/m74-....npz \
                            --fibras fibras/m74-....csv \
                            [--pasos 200] [--metodo rk2] [--submuestreo 2]

Salidas:
    adv_trayectorias.csv     - trayectoria reconstruida (fibra x paso)
    adv_influencia_zona.csv   - por (fibra, zona): tiempo y flujo experimentado
    adv_resumen_zonas.csv     - por zona: exposicion media de las fibras
    adv_corr_influencia.csv   - correlacion exposicion-de-zona vs orientacion
    fig_trayectorias.png      - trayectorias sobre el montaje
    fig_influencia_zonas.png  - que zona influye mas en la orientacion

Recomendacion para primera corrida: --submuestreo 3 --pasos 150 para que sea
rapido; luego bajar submuestreo a 1 para precision.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adveccion_fibras import (cargar_campo_piv, retroceder_fibras,
                              influencia_por_zona, resumen_global_zonas,
                              PREDICTORES)


def cargar_fibras_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"angle_deg": "theta"})
    df["fibra_id"] = np.arange(len(df))
    return df[["fibra_id", "x_mm", "y_mm", "theta"]]


def orden_parametro_local(theta_deg):
    """S de un grupo de fibras (0=isotropo, 1=alineadas). theta en grados."""
    t = np.deg2rad(np.asarray(theta_deg, float))
    return float(np.abs(np.nanmean(np.exp(2j * t))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piv", required=True)
    ap.add_argument("--fibras", required=True)
    ap.add_argument("--pasos", type=int, default=None,
                    help="frames a retroceder (por defecto todos)")
    ap.add_argument("--metodo", default="rk2", choices=["euler", "rk2"])
    ap.add_argument("--submuestreo", type=int, default=2,
                    help="usar 1 de cada k frames (k>=1) para acelerar")
    args = ap.parse_args()

    print("1) Cargando campo PIV (con series temporales u,v)...")
    campo = cargar_campo_piv(args.piv)
    print(f"   {len(campo['frames'])} frames, dt={campo['dt']*1000:.1f} ms, "
          f"zonas={sorted(campo['boxes'])}")

    print("2) Cargando fibras (foto final)...")
    fib = cargar_fibras_csv(args.fibras)
    print(f"   {len(fib)} fibras")

    print("3) Reconstruyendo trayectorias (adveccion hacia atras)...")
    traj = retroceder_fibras(
        campo, fib[["x_mm", "y_mm"]].to_numpy(),
        n_pasos=args.pasos, metodo=args.metodo,
        submuestreo_frames=args.submuestreo)
    # adjuntar theta final de cada fibra a su trayectoria
    traj = traj.merge(fib[["fibra_id", "theta"]], on="fibra_id", how="left")
    traj.to_csv("adv_trayectorias.csv", index=False)
    print(f"   {len(traj)} puntos de trayectoria guardados")

    print("4) Midiendo influencia por zona...")
    dt = campo["dt"] * args.submuestreo
    infl = influencia_por_zona(traj, dt)
    infl = infl.merge(fib[["fibra_id", "theta"]], on="fibra_id", how="left")
    infl.to_csv("adv_influencia_zona.csv", index=False)

    resumen = resumen_global_zonas(infl)
    resumen.to_csv("adv_resumen_zonas.csv", index=False)
    print("\n=== Exposicion media de las fibras por zona ===")
    print(resumen.round(3).to_string(index=False))

    print("\n5) Correlacionando exposicion de zona con orientacion final...")
    corr = correlacionar_influencia(infl)
    corr.to_csv("adv_corr_influencia.csv", index=False)
    print("\n=== Que zona/variable se asocia mas a la orientacion final ===")
    print("(rho entre el flujo experimentado en cada zona y el theta final)")
    print(corr.round(3).to_string(index=False))

    print("\n6) Graficando...")
    graficar_trayectorias(campo, traj, fib)
    graficar_influencia(corr)

    print("\nListo. Revisa adv_*.csv y fig_*.png")


def correlacionar_influencia(infl):
    """
    Para cada zona y cada predictor, correlaciona el valor de flujo que la
    fibra experimento EN ESA ZONA con su theta final. Rho alto => lo que pasa
    en esa zona esta ligado a como queda orientada la fibra. Tambien mide si
    el TIEMPO en la zona se asocia a la orientacion.
    Se usa como respuesta el alineamiento local: |cos(2*theta)| no sirve por
    fibra; en su lugar se correlaciona el predictor con theta via Spearman
    sobre el conjunto de fibras (monotona, robusta).
    """
    filas = []
    zonas = sorted(infl["zona"].dropna().unique())
    for z in zonas:
        sub = infl[infl.zona == z]
        n = len(sub)
        if n < 5:
            continue
        theta = sub["theta"].to_numpy(float)
        for p, col in [("V", "V_med"), ("omega", "omega_med"),
                       ("gamma_dot", "gamma_dot_med"),
                       ("t_en_zona", "frac_tiempo")]:
            x = sub[col].to_numpy(float)
            m = ~(np.isnan(x) | np.isnan(theta))
            if m.sum() < 5 or np.std(x[m]) == 0:
                rho, pv = np.nan, np.nan
            else:
                rho, pv = stats.spearmanr(x[m], theta[m])
            filas.append({"zona": z, "factor": p, "rho_vs_theta": rho,
                          "p_value": pv, "n_fibras": int(m.sum())})
    res = pd.DataFrame(filas)
    if len(res):
        res["abs_rho"] = res["rho_vs_theta"].abs()
        res = res.sort_values("abs_rho", ascending=False)
    return res


def graficar_trayectorias(campo, traj, fib, path="fig_trayectorias.png",
                          max_fibras=60):
    """Dibuja las trayectorias reconstruidas sobre las zonas del montaje."""
    fig, ax = plt.subplots(figsize=(11, 8))
    # zonas de fondo
    for z, (x0, x1, y0, y1) in campo["boxes"].items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor="#bbb", lw=1))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, z, color="#bbb",
                fontsize=8, ha="center", va="center")
    # trayectorias (muestra)
    ids = fib["fibra_id"].to_numpy()[:max_fibras]
    cmap = plt.cm.viridis
    for k in ids:
        t = traj[traj.fibra_id == k].sort_values("paso")
        if len(t) < 2:
            continue
        ax.plot(t["x"], t["y"], "-", lw=0.7, alpha=0.6,
                color=cmap(k / max(ids.max(), 1)))
    # punto final (foto PTV) en rojo
    ax.scatter(fib["x_mm"][:max_fibras], fib["y_mm"][:max_fibras],
               s=18, c="red", zorder=5, label="posicion final (PTV)")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    ax.set_title(f"Trayectorias reconstruidas por adveccion hacia atras\n"
                 f"({len(ids)} fibras; rojo = donde terminaron)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   guardado:", path)


def graficar_influencia(corr, path="fig_influencia_zonas.png"):
    """Heatmap: rho(flujo en zona vs theta final) por zona y factor."""
    if corr.empty:
        print("   [omitido] fig_influencia_zonas: sin datos suficientes")
        return
    piv = corr.pivot_table(index="zona", columns="factor",
                           values="rho_vs_theta")
    orden_cols = ["V", "omega", "gamma_dot", "t_en_zona"]
    piv = piv.reindex(columns=[c for c in orden_cols if c in piv.columns])
    fig, ax = plt.subplots(figsize=(8, 5))
    M = piv.to_numpy()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([{"V": "V", "omega": "ω", "gamma_dot": "γ̇",
                         "t_en_zona": "tiempo"}.get(c, c)
                        for c in piv.columns], fontsize=12)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        color="white" if abs(M[i, j]) > 0.5 else "black",
                        fontsize=10, fontweight="bold")
    ax.set_title("Influencia de cada zona en la orientacion final\n"
                 "rho entre el flujo experimentado en la zona y theta",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="ρ Spearman")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("   guardado:", path)


if __name__ == "__main__":
    main()
