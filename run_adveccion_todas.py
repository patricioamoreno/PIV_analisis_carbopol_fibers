"""
run_adveccion_todas.py
=======================
Corre la reconstruccion de trayectorias (adveccion hacia atras) para TODAS las
tomas emparejadas automaticamente (PIV .npz <-> fibras .csv por codigo mNN), 
sin necesidad de mirar la terminal: todo el progreso queda en un LOG en disco,
y los resultados de cada toma se guardan en su propia carpeta. Al final se
genera tambien un resumen acumulado de todas las tomas juntas.

Disenado para dejarlo corriendo (puede tardar bastante: 12 tomas x cientos de
frames x cientos de fibras). Se puede interrumpir y reanudar: las tomas ya
procesadas (con su archivo de salida completo) se saltan automaticamente.

USO
---
    python run_adveccion_todas.py --piv caches_zonas/ --fibras fibras/

Parametros utiles:
    --submuestreo 3      usa 1 de cada 3 frames (mas rapido, menos preciso)
    --pasos 200           limita a 200 pasos hacia atras (resto: todos los frames)
    --metodo rk2          euler (rapido) o rk2 (mas preciso, por defecto)
    --rehacer              fuerza reprocesar tomas ya completadas

SALIDAS (todo queda en disco, nada se imprime solo en pantalla)
-----------------------------------------------------------------
    log_adveccion.txt                      <- progreso completo, revisar aqui
    resultados_adveccion/<toma>/adv_trayectorias.csv
    resultados_adveccion/<toma>/adv_influencia_zona.csv
    resultados_adveccion/<toma>/adv_resumen_zonas.csv
    resultados_adveccion/<toma>/adv_corr_influencia.csv
    resultados_adveccion/<toma>/fig_trayectorias.png
    resultados_adveccion/<toma>/fig_influencia_zonas.png
    resultados_adveccion/RESUMEN_todas_las_tomas.csv   <- influencia por zona, todas las tomas
    resultados_adveccion/CORRELACION_todas_las_tomas.csv <- rho acumulado (mas fibras = mas confiable)
    resultados_adveccion/fig_resumen_influencia.png     <- heatmap final combinando todo

Como revisar el avance mientras corre (desde OTRA terminal):
    tail -f log_adveccion.txt
"""

import os
import re
import glob
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adveccion_fibras import cargar_campo_piv, retroceder_fibras, influencia_por_zona
from run_adveccion import (cargar_fibras_csv, correlacionar_influencia,
                           graficar_trayectorias, graficar_influencia)

OUT_DIR = "resultados_adveccion"
LOG_PATH = "log_adveccion.txt"


def _codigo_toma(nombre):
    """Extrae el codigo mNN del nombre de archivo (ej. 'm74-toma-1...' -> 'm74')."""
    m = re.search(r"m(\d+)", os.path.basename(str(nombre)).lower())
    return f"m{m.group(1)}" if m else None


def emparejar_por_codigo(dir_piv, dir_fibras):
    """
    Busca .npz en dir_piv y .csv en dir_fibras y los empareja por su codigo
    mNN. Devuelve lista de tuplas (ruta_npz, ruta_csv) para las que hay match.
    (Copia liviana de la funcion de acumular_tomas.py, sin sus dependencias
    de analisis estadistico, para que este script sea autonomo.)
    """
    npzs = {_codigo_toma(f): f for f in glob.glob(os.path.join(dir_piv, "*.npz"))}
    csvs = {_codigo_toma(f): f for f in glob.glob(os.path.join(dir_fibras, "*.csv"))
            if not os.path.basename(f).startswith("_")}
    comunes = sorted(set(npzs) & set(csvs) - {None})
    pares = [(npzs[c], csvs[c]) for c in comunes]

    solo_piv = sorted(set(npzs) - set(csvs) - {None})
    solo_fib = sorted(set(csvs) - set(npzs) - {None})
    if solo_piv:
        print(f"[aviso] PIV sin fibras: {solo_piv}")
    if solo_fib:
        print(f"[aviso] fibras sin PIV: {solo_fib}")
    print(f"Emparejadas {len(pares)} tomas: {comunes}")
    return pares


def log(msg):
    """Escribe al log en disco (con timestamp) Y a stdout, sin depender de
    que alguien este mirando la terminal."""
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(linea + "\n")


def toma_completa(carpeta):
    """True si la toma ya tiene todos los archivos de salida esperados."""
    esperados = ["adv_trayectorias.csv", "adv_influencia_zona.csv",
                "adv_resumen_zonas.csv", "adv_corr_influencia.csv"]
    return all(os.path.exists(os.path.join(carpeta, f)) for f in esperados)


def procesar_toma(cod, piv_path, fibras_path, args, carpeta):
    """Corre la adveccion completa para UNA toma y guarda todo en 'carpeta'."""
    os.makedirs(carpeta, exist_ok=True)

    log(f"  [{cod}] cargando campo PIV...")
    campo = cargar_campo_piv(piv_path)
    log(f"  [{cod}] {len(campo['frames'])} frames, "
        f"dt={campo['dt']*1000:.1f} ms, zonas={sorted(campo['boxes'])}")

    log(f"  [{cod}] cargando fibras...")
    fib = cargar_fibras_csv(fibras_path)
    log(f"  [{cod}] {len(fib)} fibras a rastrear")

    log(f"  [{cod}] reconstruyendo trayectorias "
        f"(esto puede tardar varios minutos)...")
    t0 = time.time()
    traj = retroceder_fibras(
        campo, fib[["x_mm", "y_mm"]].to_numpy(),
        n_pasos=args.pasos, metodo=args.metodo,
        submuestreo_frames=args.submuestreo, verbose=False)
    traj = traj.merge(fib[["fibra_id", "theta"]], on="fibra_id", how="left")
    traj.to_csv(os.path.join(carpeta, "adv_trayectorias.csv"), index=False)
    log(f"  [{cod}] trayectorias listas en {time.time()-t0:.0f}s "
        f"({len(traj)} puntos)")

    dt = campo["dt"] * args.submuestreo
    infl = influencia_por_zona(traj, dt)
    infl = infl.merge(fib[["fibra_id", "theta"]], on="fibra_id", how="left")
    infl["toma"] = cod
    infl.to_csv(os.path.join(carpeta, "adv_influencia_zona.csv"), index=False)

    resumen = infl.groupby("zona").agg(
        fibras=("fibra_id", "nunique"),
        frac_tiempo_med=("frac_tiempo", "mean"),
        V_med=("V_med", "mean"), omega_med=("omega_med", "mean"),
        gamma_dot_med=("gamma_dot_med", "mean")).reset_index()
    resumen["toma"] = cod
    resumen.to_csv(os.path.join(carpeta, "adv_resumen_zonas.csv"), index=False)

    corr = correlacionar_influencia(infl)
    corr["toma"] = cod
    corr.to_csv(os.path.join(carpeta, "adv_corr_influencia.csv"), index=False)

    try:
        graficar_trayectorias(campo, traj, fib,
                              path=os.path.join(carpeta, "fig_trayectorias.png"))
        graficar_influencia(corr,
                            path=os.path.join(carpeta, "fig_influencia_zonas.png"))
    except Exception as e:
        log(f"  [{cod}] [aviso] fallo al graficar (no crítico): {e}")

    log(f"  [{cod}] OK — guardado en {carpeta}/")
    return infl, corr


def combinar_resultados(carpetas_ok):
    """Junta la influencia y correlacion de todas las tomas procesadas."""
    infl_all, corr_all = [], []
    for cod, carpeta in carpetas_ok:
        fi = os.path.join(carpeta, "adv_influencia_zona.csv")
        fc = os.path.join(carpeta, "adv_corr_influencia.csv")
        if os.path.exists(fi):
            infl_all.append(pd.read_csv(fi))
        if os.path.exists(fc):
            corr_all.append(pd.read_csv(fc))
    if not infl_all:
        return None, None
    infl = pd.concat(infl_all, ignore_index=True)
    corr = pd.concat(corr_all, ignore_index=True) if corr_all else pd.DataFrame()

    resumen = (infl.groupby("zona").agg(
        tomas=("toma", "nunique"), fibras=("fibra_id", "size"),
        frac_tiempo_med=("frac_tiempo", "mean"),
        V_med=("V_med", "mean"), omega_med=("omega_med", "mean"),
        gamma_dot_med=("gamma_dot_med", "mean"))
        .reset_index().sort_values("frac_tiempo_med", ascending=False))
    resumen.to_csv(os.path.join(OUT_DIR, "RESUMEN_todas_las_tomas.csv"),
                   index=False)

    if not corr.empty:
        # promedio ponderado simple del rho entre tomas (mismo zona/factor)
        corr_acc = (corr.groupby(["zona", "factor"])
                    .agg(rho_medio=("rho_vs_theta", "mean"),
                         n_total=("n_fibras", "sum"),
                         n_tomas=("toma", "nunique")).reset_index())
        corr_acc.to_csv(
            os.path.join(OUT_DIR, "CORRELACION_todas_las_tomas.csv"),
            index=False)
        graficar_resumen_final(corr_acc)

    return resumen, corr


def graficar_resumen_final(corr_acc, path=None):
    path = path or os.path.join(OUT_DIR, "fig_resumen_influencia.png")
    piv = corr_acc.pivot_table(index="zona", columns="factor",
                               values="rho_medio")
    orden = ["V", "omega", "gamma_dot", "t_en_zona"]
    piv = piv.reindex(columns=[c for c in orden if c in piv.columns])
    fig, ax = plt.subplots(figsize=(8, 5.5))
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
    ax.set_title("Influencia de cada zona en la orientación final\n"
                 "(ρ promedio entre TODAS las tomas)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="ρ Spearman medio")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piv", required=True, help="carpeta con .npz de PIV")
    ap.add_argument("--fibras", required=True, help="carpeta con .csv de fibras")
    ap.add_argument("--pasos", type=int, default=None)
    ap.add_argument("--metodo", default="rk2", choices=["euler", "rk2"])
    ap.add_argument("--submuestreo", type=int, default=2)
    ap.add_argument("--rehacer", action="store_true",
                    help="reprocesa tomas aunque ya esten completas")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    log("=" * 70)
    log("INICIO procesamiento por lotes de adveccion (todas las tomas)")
    log(f"PIV dir: {args.piv} | Fibras dir: {args.fibras}")
    log(f"submuestreo={args.submuestreo} pasos={args.pasos} "
        f"metodo={args.metodo}")
    log("=" * 70)

    pares = emparejar_por_codigo(args.piv, args.fibras)
    if not pares:
        log("[ERROR] No se emparejo ninguna toma. Revisa las carpetas.")
        return

    carpetas_ok = []
    for i, (piv_path, fib_path) in enumerate(pares, 1):
        cod = _codigo_toma(piv_path)
        carpeta = os.path.join(OUT_DIR, cod)

        if not args.rehacer and toma_completa(carpeta):
            log(f"[{i}/{len(pares)}] {cod}: ya procesada, se omite "
                f"(usa --rehacer para forzar)")
            carpetas_ok.append((cod, carpeta))
            continue

        log(f"[{i}/{len(pares)}] {cod}: procesando...")
        t0 = time.time()
        try:
            procesar_toma(cod, piv_path, fib_path, args, carpeta)
            carpetas_ok.append((cod, carpeta))
            log(f"[{i}/{len(pares)}] {cod}: TERMINADA en "
                f"{(time.time()-t0)/60:.1f} min")
        except Exception as e:
            log(f"[{i}/{len(pares)}] {cod}: [ERROR] {e}")
            log(traceback.format_exc())
            log(f"[{i}/{len(pares)}] {cod}: se continua con la siguiente toma")

    log("-" * 70)
    log(f"Tomas procesadas correctamente: {len(carpetas_ok)}/{len(pares)}")
    log("Combinando resultados de todas las tomas...")
    resumen, corr = combinar_resultados(carpetas_ok)
    if resumen is not None:
        log("\n" + resumen.round(3).to_string(index=False))
        log(f"\nGuardado: {OUT_DIR}/RESUMEN_todas_las_tomas.csv")
        log(f"Guardado: {OUT_DIR}/CORRELACION_todas_las_tomas.csv")
        log(f"Guardado: {OUT_DIR}/fig_resumen_influencia.png")
    log("=" * 70)
    log("PROCESO COMPLETO. Revisa la carpeta 'resultados_adveccion/'.")
    log("=" * 70)


if __name__ == "__main__":
    main()
