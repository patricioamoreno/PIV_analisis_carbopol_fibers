"""
box_act.py
==========
Estadística refinada de velocidades por polilínea.

Para cada combinación (reología + concentración de fibras):
  - Reune v_mag de TODOS los frames de la etapa indicada
    de TODAS las carpetas de esa combinación (todas las mezclas juntas)

Genera:
  Fig 1 — Car-02: 4 boxplots (0, 750, 1500, 3000 fibras)
  Fig 2 — Car-05: 4 boxplots (0, 750, 1500, 3000 fibras)
  Fig 3 — Comparativa Car-02 vs Car-05
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from utils_etapas import natural_sort_key, cargar_etapas
from construir_caches import cargar_cache_completo, nombre_base_carpeta

# ============================================================
# CONFIGURACIÓN
# ============================================================

CACHE_DIR             = "cache_completo"
ETAPAS_POLILINEA_JSON = "etapas_polilinea.json"
OUT_BASE = "Boxplots/Boxplots_por_etapa"

CONCENTRACIONES = ['0000', '0750', '1500', '3000']
LABELS_CONC     = ['0 fibras', '750 fibras', '1500 fibras', '3000 fibras']
COLORES_CAR     = {'02': 'steelblue', '05': 'darkorange'}

ZONAS = {
    'L':       '',
    'viga175': 'viga175',
    'viga250': 'viga250',
}


# ============================================================
# UTILIDADES
# ============================================================

def agrupar_carpetas(etapas_poli):
    """
    Agrupa carpetas por (reo, conc) a partir del JSON de etapas_polilinea.
    Usa nombre_base_carpeta para eliminar sufijos de zona (_L, _viga175, etc.)
    """
    grupos = {}
    for clave in etapas_poli.keys():
        carpeta = nombre_base_carpeta(clave)
        m_reo   = re.search(r'car-(\d+)', carpeta)
        m_conc  = re.search(r'n-(\d+)',   carpeta)
        if not m_reo or not m_conc:
            continue
        key = (m_reo.group(1), m_conc.group(1))
        grupos.setdefault(key, set()).add(carpeta)
    return {k: sorted(v, key=natural_sort_key) for k, v in grupos.items()}


# ============================================================
# RECOLECCIÓN DE DATOS
# ============================================================

def recolectar_desde_cache(reo, conc, zona, etapa, variable='vel', p=True, usar_magnitud=False):
    """
    Reúne todos los valores de la variable indicada para una combinación
    (reo, conc, zona, etapa) de todas las tomas disponibles.
    
    usar_magnitud=True → usa cachés _mag (magnitud √u²+v²), solo disponible para vigas.
    """
    etapas_poli = cargar_etapas(ETAPAS_POLILINEA_JSON)
    grupos      = agrupar_carpetas(etapas_poli)
    key         = (reo, conc)
    if key not in grupos:
        return None

    prefijo = ZONAS[zona]
    todos   = []

    for carpeta in grupos[key]:
        clave_zona = f"{carpeta}_{zona}"
        if clave_zona not in etapas_poli:
            continue

        i_ini, i_fin = etapas_poli[clave_zona]['etapas'][etapa]

        mat_full, _ = cargar_cache_completo(carpeta, prefijo, CACHE_DIR, usar_magnitud=usar_magnitud)
        if mat_full is None:
            continue

        i_fin     = min(i_fin, mat_full.shape[0])
        mat_etapa = mat_full[i_ini:i_fin]

        if variable == 'vel':
            vals = mat_etapa[~np.isnan(mat_etapa)].flatten()
        else:
            N_t, N_s   = mat_etapa.shape
            ss         = np.linspace(0, 20, N_s) if zona == 'L' else np.linspace(0, 100, N_s)
            gamma_vals = []
            for i in range(N_t):
                v = mat_etapa[i, :]
                if np.all(np.isnan(v)):
                    continue
                nan_mask = np.isnan(v)
                if nan_mask.any():
                    idx = np.arange(len(v))
                    v   = v.copy()
                    v[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], v[~nan_mask])
                v_s = uniform_filter1d(v, size=5, mode='nearest')
                gd  = np.abs(np.gradient(v_s, ss))
                gd[nan_mask] = np.nan
                gamma_vals.append(gd[~np.isnan(gd)])
            vals = np.concatenate(gamma_vals) if gamma_vals else np.array([])

        todos.append(vals)

    if not todos:
        return None
    resultado = np.concatenate(todos)
    if p:
        print(f"  Car-{reo} / {conc} / {zona} / {etapa} → {len(resultado):,} valores")
    return resultado


# ============================================================
# ESTADÍSTICAS DE RESUMEN
# ============================================================

def imprimir_estadisticas(zona, etapa, variable, f_txt):
    ylabel = "Velocidad (mm/s)" if variable == 'vel' else "γ̇ (s⁻¹)"

    lines = [
        f"{'='*85}",
        f"  Variable: {ylabel}  |  Etapa: {etapa}  |  Zona: {zona}",
        f"  {'Caso':<30} {'Mediana':>10} {'IQR':>10} {'N pts':>10}   {'Δ vs base':>20}",
        f"  {'-'*80}",
    ]

    for reo in ['02', '05']:
        vals_base = recolectar_desde_cache(reo, '0000', zona, etapa=etapa, variable=variable, p=False)
        med_base  = np.median(vals_base) if vals_base is not None and len(vals_base) > 0 else None

        for conc, label_conc in zip(CONCENTRACIONES, LABELS_CONC):
            vals = recolectar_desde_cache(reo, conc, zona, etapa=etapa, variable=variable, p=False)
            if vals is not None and len(vals) > 0:
                q25, q75 = np.percentile(vals, [25, 75])
                mediana  = np.median(vals)
                iqr      = q75 - q25
                label    = f"Car-{reo} / {conc} fibras"
                if med_base is not None and conc != '0000':
                    diff     = mediana - med_base
                    diff_pct = (diff / med_base) * 100
                    diff_str = f"{diff:+.3f}  ({diff_pct:+.1f}%)"
                else:
                    diff_str = "base"
                lines.append(f"  {label:<30} {mediana:>10.3f} {iqr:>10.3f} {len(vals):>12,}   {diff_str}")
            else:
                lines.append(f"  Car-{reo} / {conc} fibras{'Sin datos':>40}")

    lines.append(f"{'='*85}\n")
    f_txt.write('\n'.join(lines) + '\n')


# ============================================================
# GRAFICADO
# ============================================================

def graficar_una_reologia(reo, ax, zona, etapa, variable='vel', mostrar_n=True, usar_magnitud=False):
    datos, labels = [], []
    color = COLORES_CAR[reo]

    for conc, label in zip(CONCENTRACIONES, LABELS_CONC):
        vals = recolectar_desde_cache(reo, conc, zona, etapa=etapa, variable=variable, usar_magnitud=usar_magnitud)
        if vals is not None and len(vals) > 0:
            p5, p95 = np.percentile(vals, [5, 95])
            vals    = vals[(vals >= p5) & (vals <= p95)]
            datos.append(vals)
            mediana = f"\nmed={np.median(vals):.2f} mm/s" if variable == 'vel' else f"\nmed={np.median(vals):.2f} s⁻¹"
            n_str = f"\n(n={len(vals)/1e3:.1f}K)" if mostrar_n else ""
            labels.append(label + mediana + n_str)
        else:
            datos.append(np.array([np.nan]))
            labels.append(label)

    ax.boxplot(
        datos,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color='black', lw=2),
        whiskerprops=dict(color=color, lw=1.2),
        capprops=dict(color=color, lw=1.5),
        boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color),
    )
    reo_label = reo.replace('02', '0.2%').replace('05', '0.5%')
    ylabel    = "Velocidad (mm/s)" if variable == 'vel' else "Tasa de deformación γ̇ [s⁻¹]"
    mag_str   = " [magnitud |V|]" if usar_magnitud else ""
    ax.set_ylabel(ylabel)
    ax.set_title(f"Carbopol {reo_label} — {etapa} - {zona}{mag_str}\n(whiskers = percentil 5-95)")
    ax.grid(True, alpha=0.3, axis='y')


def graficar_comparativa(ax, zona, etapa, variable='vel', usar_magnitud=False):
    n_conc = len(CONCENTRACIONES)
    pos_02 = np.arange(1, n_conc * 3, 3)
    pos_05 = pos_02 + 1

    for posiciones, reo in [(pos_02, '02'), (pos_05, '05')]:
        color = COLORES_CAR[reo]
        datos, posiciones_validas = [], []
        for conc, pos in zip(CONCENTRACIONES, posiciones):
            vals = recolectar_desde_cache(reo, conc, zona, etapa=etapa, variable=variable, usar_magnitud=usar_magnitud)
            if vals is not None and len(vals) > 0:
                p5, p95 = np.percentile(vals, [5, 95])
                datos.append(vals[(vals >= p5) & (vals <= p95)])
                posiciones_validas.append(pos)

        reo_label = reo.replace('02', '0.2%').replace('05', '0.5%')
        ax.boxplot(
            datos,
            positions=posiciones_validas,
            widths=0.7,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color='black', lw=2),
            whiskerprops=dict(color=color, lw=1.2),
            capprops=dict(color=color, lw=1.5),
            boxprops=dict(facecolor=color, alpha=0.4, edgecolor=color),
            label=f"Car-{reo_label}"
        )

    ax.set_xticks(pos_02 + 0.5)
    ax.set_xticklabels(LABELS_CONC)
    ylabel = "Velocidad (mm/s)" if variable == 'vel' else "Tasa de deformación γ̇ [s⁻¹]"
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_BASE, exist_ok=True)
    txt_path = os.path.join(OUT_BASE, "estadisticas_todas.txt")

    with open(txt_path, 'w', encoding='utf-8') as f_txt:
        # ── Boxplots originales (proyección) ──────────────────────────────
        for variable in ['vel', 'gamma_dot']:
            for etapa in ['transicion', 'cuasi']:
                OUTPUT_PATH = f"{OUT_BASE}/boxplots_{variable}_{etapa}"
                os.makedirs(OUTPUT_PATH, exist_ok=True)

                print(f"\n{'='*58}")
                print(f"  Boxplots — variable: {variable} | etapa: {etapa}")
                print(f"  Output: {OUTPUT_PATH}")
                print(f"{'='*58}")

                for zona in ['L', 'viga175', 'viga250']:
                    imprimir_estadisticas(zona, etapa, variable, f_txt)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    graficar_una_reologia('02', ax, zona, etapa, variable=variable)
                    plt.tight_layout()
                    ruta = os.path.join(OUTPUT_PATH, f"boxplot_car02_{zona}.png")
                    plt.savefig(ruta, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ {ruta}")

                    fig, ax = plt.subplots(figsize=(10, 5))
                    graficar_una_reologia('05', ax, zona, etapa, variable=variable)
                    plt.tight_layout()
                    ruta = os.path.join(OUTPUT_PATH, f"boxplot_car05_{zona}.png")
                    plt.savefig(ruta, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ {ruta}")

                    fig, ax = plt.subplots(figsize=(12, 5))
                    graficar_comparativa(ax, zona, etapa, variable=variable)
                    plt.tight_layout()
                    ruta = os.path.join(OUTPUT_PATH, f"{variable}_{etapa}_{zona}.png")
                    plt.savefig(ruta, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ {ruta}")

        # ── Boxplots magnitud |V| en vigas (para comparar con PTV) ───────
        print(f"\n{'='*58}")
        print(f"  Boxplots magnitud |V| — solo vigas (comparación PTV)")
        print(f"{'='*58}")

        for etapa in ['transicion', 'cuasi']:
            OUTPUT_MAG = f"{OUT_BASE}/boxplots_vel_magnitud_{etapa}"
            os.makedirs(OUTPUT_MAG, exist_ok=True)

            for zona in ['L', 'viga175', 'viga250']:
                for reo in ['02', '05']:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    graficar_una_reologia(reo, ax, zona, etapa, variable='vel', usar_magnitud=True)
                    plt.tight_layout()
                    ruta = os.path.join(OUTPUT_MAG, f"boxplot_mag_car{reo}_{zona}.png")
                    plt.savefig(ruta, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ {ruta}")

                fig, ax = plt.subplots(figsize=(12, 5))
                graficar_comparativa(ax, zona, etapa, variable='vel', usar_magnitud=True)
                plt.tight_layout()
                ruta = os.path.join(OUTPUT_MAG, f"vel_magnitud_{etapa}_{zona}.png")
                plt.savefig(ruta, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ {ruta}")

    print(f"\n✅ Listo. Figuras en {OUT_BASE}")