"""
analisis_por_zona.py
=====================
Compara tomas de una misma mezcla (misma reología y concentración),
POR ZONA en vez de por polilínea de corte.

Motivación: una polilínea resuelve solo un corte 1D del dominio; una zona
agrega sobre una región de área positiva y es más representativa del
conjunto del flujo en esa región. Es el análogo directo de analisis.py,
pero usa cache_zonas/ + etapas_zonas.json en vez de cache_completo/ +
etapas_polilinea.json.

Para cada grupo (reología, concentración):
  - Fig 1: diagnóstico del criterio V3 por zona (velocidad media de la
           zona por frame, igual serie que usa calcular_etapas_zonas.py
           para fijar el corte).
  - Fig 2: boxplots por zona y etapa (transición y cuasi-estacionario),
           con cada toma del grupo y una caja BASE (tomas n-0000 de la
           misma reología) para comparar.
  - CSV: tests estadísticos (entre tomas, y cada toma vs base) con p-valor,
         d de Cohen y diferencia de medianas, por (zona, etapa) -- para
         decidir qué CELDAS excluir por no ser comparables con el caso
         base, sin tener que descartar una toma completa.

Uso:
    python analisis_por_zona.py
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from construir_caches_zonas import cargar_cache_zonas
from calcular_etapas_zonas import serie_temporal_zona
from detectar_etapas import natural_sort_key, VENTANA_SUAVIZADO, EPS_CUASI

# ============================================================
# CONFIGURACIÓN
# ============================================================

OUTPUT_PATH        = "Analisis_COMPARATIVA_zonas"
ETAPAS_ZONAS_JSON  = "etapas_zonas.json"
CACHE_DIR          = "cache_zonas"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# UTILIDADES
# ============================================================

def cargar_etapas_zonas():
    with open(ETAPAS_ZONAS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def zonas_presentes(etapas_zonas):
    """Lista de zonas distintas en las claves del JSON, en orden L -> viga.

    Las claves tienen forma '{carpeta}_{zona}', y ningún nombre de zona
    (Z1, Z2, Z3, Vf1c1, ...) ni de carpeta contiene guion bajo, así que el
    último segmento tras el último '_' es siempre la zona.
    """
    zonas = {clave.rsplit("_", 1)[1] for clave in etapas_zonas.keys()}
    orden_l = ["Z1", "Z2", "Z3"]
    orden_viga = sorted(z for z in zonas if z.startswith("Vf"))
    return [z for z in orden_l if z in zonas] + orden_viga


def _carpeta_de_clave(clave, zona):
    """Recupera el nombre de carpeta a partir de una clave '{carpeta}_{zona}'."""
    sufijo = f"_{zona}"
    assert clave.endswith(sufijo), f"clave {clave} no termina en {sufijo}"
    return clave[: -len(sufijo)]


def agrupar_carpetas(etapas_zonas):
    """Agrupa carpetas por (reo, conc), a partir de las claves del JSON."""
    grupos = {}
    for clave in etapas_zonas.keys():
        zona = clave.rsplit("_", 1)[1]
        carpeta = _carpeta_de_clave(clave, zona)
        m_reo  = re.search(r"car-(\d+)", carpeta)
        m_conc = re.search(r"n-(\d+)",   carpeta)
        if not m_reo or not m_conc:
            continue
        key = (m_reo.group(1), m_conc.group(1))
        grupos.setdefault(key, set()).add(carpeta)
    return {k: sorted(v, key=natural_sort_key) for k, v in grupos.items()}


# ============================================================
# FUNCIONES DE DATOS
# ============================================================

def recolectar_toma_zona(carpeta, zona, etapa, etapas_zonas, cache_dir=CACHE_DIR):
    """v_mag de una sola toma, para una zona y etapa (rango de frame_idx
    según etapas_zonas.json)."""
    clave = f"{carpeta}_{zona}"
    if clave not in etapas_zonas:
        return None
    i_ini, i_fin = etapas_zonas[clave]["etapas"][etapa]
    cache = cargar_cache_zonas(carpeta, cache_dir)
    if cache is None:
        print(f"  ⚠ Sin caché: {carpeta} / {zona}")
        return None
    m = ((cache["zona"] == zona) &
         (cache["frame_idx"] >= i_ini) & (cache["frame_idx"] < i_fin))
    v = cache["v_mag"][m]
    return v[~np.isnan(v)]


def recolectar_base_zona(etapas_zonas, reo, zona, etapa, cache_dir=CACHE_DIR):
    """Junta valores de todas las tomas n-0000 de una reología, como base
    de referencia (recorte a percentil 5-95 por toma antes de concatenar,
    mismo criterio que analisis.py)."""
    todos = []
    for clave in etapas_zonas.keys():
        z = clave.rsplit("_", 1)[1]
        if z != zona:
            continue
        carpeta = _carpeta_de_clave(clave, zona)
        if f"car-{reo}" not in carpeta or "n-0000" not in carpeta:
            continue
        v = recolectar_toma_zona(carpeta, zona, etapa, etapas_zonas, cache_dir)
        if v is not None and len(v) > 0:
            p5, p95 = np.percentile(v, [5, 95])
            todos.append(v[(v >= p5) & (v <= p95)])
    return np.concatenate(todos) if todos else None


def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return np.nan
    return abs((np.mean(a) - np.mean(b)) / s_pooled)


def interpretar_d(d):
    if np.isnan(d):  return "n/a"
    if d < 0.2:      return "insignificante"
    if d < 0.5:      return "débil"
    if d < 0.8:      return "medio"
    return "fuerte"


# ============================================================
# DIAGNÓSTICO CRITERIO V3, POR ZONA
# ============================================================

def graficar_diagnostico_criterio_zona(etapas_zonas, carpetas, zona,
                                       output_path, ids, car):
    """
    Análogo de graficar_diagnostico_criterio en analisis.py, pero usando
    serie_temporal_zona() -- la MISMA serie que calcular_etapas_zonas.py
    usa para fijar el corte V3 -- en vez de la matriz de la polilínea.
    """
    series_t, series_v, series_tasa = [], [], []

    for carpeta in carpetas:
        clave = f"{carpeta}_{zona}"
        if clave not in etapas_zonas:
            continue
        cache = cargar_cache_zonas(carpeta, CACHE_DIR)
        if cache is None:
            continue
        t_full, v_media = serie_temporal_zona(cache, zona)

        v_s = pd.Series(v_media).rolling(window=VENTANA_SUAVIZADO,
                                         center=True).mean()
        v_s = v_s.fillna(pd.Series(v_media)).values
        dt     = np.diff(t_full)
        dv     = np.diff(v_s) / dt
        dv     = np.append(dv, dv[-1])
        v_peak = np.nanmax(v_s)
        tasa   = np.abs(dv) / v_peak if v_peak > 0 else np.zeros_like(dv)

        series_t.append(t_full)
        series_v.append(v_s)
        series_tasa.append(tasa)

    if not series_t:
        return

    t_min = max(t[0]  for t in series_t)
    t_max = min(t[-1] for t in series_t)
    t_com = np.linspace(t_min, t_max, 500)

    v_interp    = np.array([np.interp(t_com, st, sv) for st, sv in zip(series_t, series_v)])
    tasa_interp = np.array([np.interp(t_com, st, sr) for st, sr in zip(series_t, series_tasa)])
    v_prom    = np.nanmean(v_interp,    axis=0)
    tasa_prom = np.nanmean(tasa_interp, axis=0)

    t_peaks_grupo, t_quasis_grupo = [], []
    for carpeta in carpetas:
        clave = f"{carpeta}_{zona}"
        if clave in etapas_zonas:
            t_peaks_grupo.append(etapas_zonas[clave]["t_peak"])
            t_quasis_grupo.append(etapas_zonas[clave]["t_quasi"])
    t_peak_med  = np.mean(t_peaks_grupo)  if t_peaks_grupo  else None
    t_quasi_med = np.mean(t_quasis_grupo) if t_quasis_grupo else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    if t_peak_med and t_quasi_med:
        for ax in (ax1, ax2):
            ax.axvspan(t_com[0],   t_peak_med,  alpha=0.07, color="royalblue")
            ax.axvspan(t_peak_med, t_quasi_med, alpha=0.07, color="darkorange")
            ax.axvspan(t_quasi_med, t_com[-1],  alpha=0.07, color="forestgreen")
        ax1.axvline(t_peak_med,  color="royalblue",   lw=1.4, ls="--",
                    label=f"Peak  t={t_peak_med:.2f}s")
        ax1.axvline(t_quasi_med, color="forestgreen", lw=1.4, ls="--",
                    label=f"Cuasi t={t_quasi_med:.2f}s")

    for sv, sr in zip(v_interp, tasa_interp):
        ax1.plot(t_com, sv, color="gray", lw=0.7, alpha=0.4)
        ax2.plot(t_com, sr, color="gray", lw=0.7, alpha=0.4)

    ax1.plot(t_com, v_prom,    color="steelblue", lw=2.0, label="Promedio tomas")
    ax2.plot(t_com, tasa_prom, color="darkorange", lw=2.0, label="Promedio tomas")
    ax2.axhline(EPS_CUASI, color="crimson", lw=1.2, ls="--",
               label=f"ε = {EPS_CUASI} s⁻¹")

    ax1.set_ylabel("V media (mm/s)")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, ncol=3)
    ax2.set_ylabel("|dV/dt| / V_peak  (s⁻¹)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylim(0, 0.15)
    ax2.legend(fontsize=8)

    titulo = f"Diagnóstico criterio V3 — m{ids} Car-{car} — Zona {zona}"
    fig.suptitle(titulo, fontsize=11, fontweight="bold")

    fname = os.path.join(output_path, f"m{ids}_car{car}_diag_{zona}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"  ✅ Diagnóstico guardado: {fname}")


# ============================================================
# ANÁLISIS POR GRUPO
# ============================================================

def analizar_grupo(etapas_zonas, carpetas, zonas, output_path):
    car  = re.search(r"car-(\w+)", carpetas[0]).group(1)
    conc = re.search(r"n-(\d+)",   carpetas[0]).group(1)
    labels_tomas = ["m" + re.search(r"m(\d+)", c).group(1) + "-toma-" +
                    re.search(r"-toma-(\d+)", c).group(1) for c in carpetas]
    ids   = "+".join(sorted(set(re.search(r"m(\d+)", c).group(1) for c in carpetas)))
    label = f"m{ids} Car-{car} n-{conc}"

    print(f"\n{'='*55}\n  {label}  ({len(carpetas)} tomas, {len(zonas)} zonas)")

    # ── Figuras de diagnóstico criterio V3 (una por zona) ────
    for zona in zonas:
        graficar_diagnostico_criterio_zona(
            etapas_zonas, carpetas, zona, output_path, ids, car
        )

    # ── Boxplots + tests, por zona ────────────────────────────
    filas_csv = []

    for zona in zonas:
        vals_base = {
            "transicion": recolectar_base_zona(etapas_zonas, car, zona, "transicion"),
            "cuasi":      recolectar_base_zona(etapas_zonas, car, zona, "cuasi"),
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 subplot_kw=dict(box_aspect=None))
        fig.subplots_adjust(bottom=0.1)

        for ax, etapa in zip(axes, ["transicion", "cuasi"]):
            datos, lbs = [], []
            for i, carpeta in enumerate(carpetas):
                v = recolectar_toma_zona(carpeta, zona, etapa, etapas_zonas)
                if v is not None and len(v) > 0:
                    p5, p95 = np.percentile(v, [5, 95])
                    datos.append(v[(v >= p5) & (v <= p95)])
                    lbs.append(labels_tomas[i])

            if not datos:
                ax.set_visible(False)
                continue

            vb = vals_base[etapa]
            all_datos = datos + ([vb] if vb is not None else [])
            all_lbs   = lbs   + (["BASE"] if vb is not None else [])

            bp = ax.boxplot(all_datos, tick_labels=all_lbs,
                            showfliers=False, patch_artist=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor("lightcoral" if i == len(datos) else "steelblue")
                patch.set_alpha(0.5)

            # ── Test entre tomas del grupo ────────────────────
            if len(datos) >= 2:
                if len(datos) == 2:
                    stat, p_entre = stats.mannwhitneyu(datos[0], datos[1],
                                                       alternative="two-sided")
                    metodo_entre  = "Mann-Whitney"
                    d_entre       = cohen_d(datos[0], datos[1])
                    delta_med     = abs(np.median(datos[0]) - np.median(datos[1]))
                else:
                    stat, p_entre = stats.kruskal(*datos)
                    metodo_entre  = "Kruskal-Wallis"
                    d_entre       = np.nan
                    delta_med     = np.nan
                res_entre = (f"{metodo_entre} p={p_entre:.4f} | "
                            f"d={d_entre:.2f} ({interpretar_d(d_entre)}) | "
                            f"Δmed={delta_med:.3f} mm/s")
                filas_csv.append({
                    "zona": zona, "etapa": etapa,
                    "comparacion": "entre_tomas", "toma": "todas",
                    "metodo":      metodo_entre,
                    "p_value":     round(p_entre, 6),
                    "resultado":   "SÍ difieren" if p_entre < 0.05 else "NO difieren",
                    "cohen_d":     round(d_entre, 4) if not np.isnan(d_entre) else "",
                    "efecto":      interpretar_d(d_entre),
                    "delta_mediana": round(delta_med, 4) if not np.isnan(delta_med) else "",
                    "mediana_toma": "", "mediana_base": "",
                })
            else:
                res_entre = "solo 1 toma"

            ax.set_title(f"{etapa} — {zona}\n{res_entre}", fontsize=9)

            # ── Test cada toma vs base ────────────────────────
            if vb is not None:
                for j, (v, lbl) in enumerate(zip(datos, lbs)):
                    stat, p_vb = stats.mannwhitneyu(v, vb, alternative="two-sided")
                    d_vb       = cohen_d(v, vb)
                    delta_vb   = np.median(v) - np.median(vb)
                    resultado  = "SÍ difiere" if p_vb < 0.05 else "NO difiere"
                    ax.text(j + 1, -0.07,
                        f"p={p_vb:.3f}\nd={d_vb:.2f} ({interpretar_d(d_vb)})\nΔ={delta_vb:+.3f}",
                        ha="center", va="top", fontsize=7, color="navy",
                        transform=ax.get_xaxis_transform())
                    filas_csv.append({
                        "zona": zona, "etapa": etapa,
                        "comparacion":   "vs_base", "toma": lbl,
                        "metodo":        "Mann-Whitney",
                        "p_value":       round(p_vb, 6),
                        "resultado":     resultado,
                        "cohen_d":       round(d_vb, 4),
                        "efecto":        interpretar_d(d_vb),
                        "delta_mediana": round(float(delta_vb), 4),
                        "mediana_toma":  round(float(np.median(v)),  4),
                        "mediana_base":  round(float(np.median(vb)), 4),
                    })

            ax.set_ylabel("Velocidad (mm/s)")
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"{label} — zona {zona} — {len(carpetas)} tomas\n"
                    "(whiskers = percentil 5-95)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"m{ids}_car{car}_bp_{zona}.png"),
                    dpi=200)
        plt.close()

    # ── CSV con todos los tests ───────────────────────────────
    if filas_csv:
        df = pd.DataFrame(filas_csv)
        csv_path = os.path.join(output_path, f"m{ids}_car{car}_tests.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  ✅ Tests guardados: {csv_path}")

    print(f"  ✅ Figuras guardadas.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    etapas_zonas = cargar_etapas_zonas()
    zonas        = zonas_presentes(etapas_zonas)
    grupos       = agrupar_carpetas(etapas_zonas)

    print(f"Zonas detectadas: {zonas} ")

    for (reo, conc), carpetas in sorted(grupos.items()):
        output_path = os.path.join(OUTPUT_PATH, f"reo{reo}_conc{conc}")
        os.makedirs(output_path, exist_ok=True)
        print(f"\nReología {reo} | Concentración {conc} | {len(carpetas)} carpetas")
        if len(carpetas) < 1:
            print("  Menos de 1 toma, se omite.")
            continue
        analizar_grupo(etapas_zonas, carpetas, zonas, output_path)