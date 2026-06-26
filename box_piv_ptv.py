"""
box_piv_ptv.py
==============
Boxplots comparativos PIV vs PTV por etapa (transicion y cuasi).

Un solo grafico con 4 cajas agrupadas en 2 pares:
    [PIV transicion | PTV transicion]   [PIV cuasi | PTV cuasi]

VARIABLE: magnitud de velocidad |V| en AMBOS metodos, para que sean
comparables:
    PIV : caches _mag  (usar_magnitud=True)  -> sqrt(u^2 + v^2)
    PTV : sqrt(vx_mm_s^2 + vy_mm_s^2)  calculado del CSV

ETAPAS: las tuyas (etapas_polilinea.json), definidas por indices de frame
por toma. Se convierten a segundos con el vector de tiempos de la cache de
cada toma para poder filtrar el PTV (que esta en segundos).
"""

import os
import re
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from utils_etapas import natural_sort_key
from construir_caches import cargar_cache_completo, nombre_base_carpeta

# ============================================================
# CONFIGURACION
# ============================================================

CACHE_DIR             = "cache_completo"
ETAPAS_POLILINEA_JSON = "etapas_polilinea.json"
OUT_DIR               = "Boxplots/Boxplots_PIV_PTV"

# Grupo y zona a comparar
REO  = ["02", "05"]  # "02", "05"
CONC = ["0750", "1500", "3000"]
ZONAS = [
    ("L",       "",        "Salida de la L  (canal cerrado)"),
    ("viga175", "viga175", "Viga 175 mm"),
    ("viga250", "viga250", "Viga 250 mm"),
]

PTV_PATH  = "datos_esp_ptv"

ETAPAS = ['inicio', 'transicion', 'cuasi']

# Recorte de outliers para las cajas (igual criterio que box_act.py)
PCT_LO, PCT_HI = 5, 95

COLOR_PIV = 'steelblue'
COLOR_PTV = 'darkorange'


# ============================================================
# UTILIDADES
# ============================================================

def cargar_etapas():
    with open(ETAPAS_POLILINEA_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def carpetas_del_grupo(etapas_poli, reo, conc):
    carps = set()
    for clave in etapas_poli.keys():
        carpeta = nombre_base_carpeta(clave)
        if re.search(rf'car-{reo}', carpeta) and re.search(rf'n-{conc}', carpeta):
            carps.add(carpeta)
    return sorted(carps, key=natural_sort_key)


# ============================================================
# PIV: magnitud por etapa (estilo box_act, con usar_magnitud=True)
# ============================================================

def recolectar_piv(etapas_poli, reo, conc, zona, etapa, prefijo):
    """Reune todos los valores de |V| del PIV para la etapa, todas las tomas."""
    todos = []
    for carpeta in carpetas_del_grupo(etapas_poli, reo, conc):
        clave = f"{carpeta}_{zona}"
        if clave not in etapas_poli:
            continue
        i0, i1 = etapas_poli[clave]['etapas'][etapa]
        mat, _ = cargar_cache_completo(carpeta, prefijo, CACHE_DIR, usar_magnitud=True)
        if mat is None:
            print(f"  ⚠ Sin cache _mag: {carpeta}")
            continue
        i1 = min(i1, mat.shape[0])
        if i0 >= i1:
            continue
        sub = mat[i0:i1]
        todos.append(sub[~np.isnan(sub)].flatten())
    return np.concatenate(todos) if todos else np.array([])


# ============================================================
# CARGA PTV  (formato npz)
# ============================================================

def ruta_ptv(reo, conc, prefijo):
    """Ruta al npz de PTV de la zona, con el mismo esquema de nombres del PIV."""
    base = f"car-{reo}_n-{conc}_ptv_datos_suavizados.npz"
    nombre = f"{prefijo}_{base}" if prefijo else base
    return os.path.join(PTV_PATH, nombre)


def cargar_ptv_magnitud(path):
    """
    Lee un npz de PTV con columnas:
      t, s, v_paralela, track_id, x_mm, y_mm, vx_mm_s, vy_mm_s
    Devuelve (t, s, v, tid) donde v = |V| = hypot(vx, vy).
    """
    data = np.load(path, allow_pickle=True)
    t  = np.asarray(data['t'], dtype=float)
    s  = np.asarray(data['s'], dtype=float)
    vx = np.asarray(data['vx_mm_s'], dtype=float)
    vy = np.asarray(data['vy_mm_s'], dtype=float)
    v  = np.hypot(vx, vy)
    tid = np.asarray(data['track_id']) if 'track_id' in data.files else None
    return t, v

# ============================================================
# PTV: magnitud por etapa, filtrando por ventana temporal de la etapa
# ============================================================

def ventana_temporal_etapa(etapas_poli, reo, conc, zona, etapa, prefijo):
    """
    Convierte los indices de frame (i0,i1) de la etapa a segundos usando el
    vector de tiempos de la cache de cada toma. Devuelve (t_ini, t_fin)
    promedio sobre las tomas del grupo (las etapas se definieron asi).
    """
    t_inis, t_fins = [], []
    for carpeta in carpetas_del_grupo(etapas_poli, reo, conc):
        clave = f"{carpeta}_{zona}"
        if clave not in etapas_poli:
            continue
        i0, i1 = etapas_poli[clave]['etapas'][etapa]
        _, tiempos = cargar_cache_completo(carpeta, prefijo, CACHE_DIR, usar_magnitud=True)
        if tiempos is None:
            continue
        # Mismo limite que el PIV en recolectar_piv (i1 exclusivo = mat.shape[0]).
        i1 = min(i1, len(tiempos))
        if i0 >= i1:
            continue
        t_inis.append(tiempos[i0])
        t_fins.append(tiempos[i1 - 1])  # ultimo frame realmente incluido en el PIV
    if not t_inis:
        return None, None
    return float(np.mean(t_inis)), float(np.mean(t_fins))


def recolectar_ptv(t_ini, t_fin, reo, conc, prefijo):
    ptv_path = ruta_ptv(reo, conc, prefijo)
    if not os.path.exists(ptv_path):
        print(f"  ⚠ Sin PTV: {ptv_path} (omitida)")
        return np.array([])

    t_ptv, mag_ptv = cargar_ptv_magnitud(ptv_path)
    m = (t_ptv >= t_ini) & (t_ptv <= t_fin)
    return mag_ptv[m]


# ============================================================
# RECORTE DE OUTLIERS
# ============================================================

def recortar(vals):
    if vals is None or len(vals) == 0:
        return np.array([])
    lo, hi = np.percentile(vals, [PCT_LO, PCT_HI])
    return vals[(vals >= lo) & (vals <= hi)]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    etapas_poli = cargar_etapas()

    
    pos = 1
    tablas = []  # cada elemento: (titulo, filas_muestra)
    for zona, prefijo, zona_label in ZONAS:
        for conc in CONC:
            for reo in REO:
                datos, labels, colores, posiciones = [], [], [], []
                filas_muestra = []
                print(f"\n{'='*55}\nCarbopol {reo} n-{conc}  zona {zona}\n{'='*55}")
                for etapa in ETAPAS:
                    # PIV
                    v_piv = recortar(recolectar_piv(etapas_poli, reo, conc, zona, etapa, prefijo))
                    # PTV en la misma ventana temporal de la etapa
                    t_ini, t_fin = ventana_temporal_etapa(etapas_poli, reo, conc, zona, etapa, prefijo)
                    if t_ini is None:
                        print(f"  ⚠ Sin ventana temporal para {etapa}")
                        v_ptv = np.array([])
                    else:
                        v_ptv = recortar(recolectar_ptv(t_ini, t_fin, reo, conc, prefijo))
                        print(f"  {etapa}: ventana [{t_ini:.2f}, {t_fin:.2f}] s  "
                            f"PIV n={len(v_piv):,}  PTV n={len(v_ptv):,}")

                    for vals, metodo, color in [(v_piv, 'PIV', COLOR_PIV),
                                                (v_ptv, 'PTV', COLOR_PTV)]:
                        datos.append(vals if len(vals) else np.array([np.nan]))
                        med = np.median(vals) if len(vals) else np.nan
                        labels.append(f"{metodo}\n{etapa}\nmed={med:.1f}\nn={len(vals)/1e3:.1f}K")
                        colores.append(color)
                        posiciones.append(pos)
                        pos += 1
                        filas_muestra.append((etapa, metodo, med, len(vals)))
                    pos += 0.7  # separacion entre pares de etapa

                fig, ax = plt.subplots(figsize=(11, 6))
                bp = ax.boxplot(
                    datos, positions=posiciones, widths=0.6,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                )
                for patch, color in zip(bp['boxes'], colores):
                    patch.set_facecolor(color); patch.set_alpha(0.45); patch.set_edgecolor(color)
                for elem in ['whiskers', 'caps']:
                    for i, ln in enumerate(bp[elem]):
                        ln.set_color(colores[i // 2]); ln.set_linewidth(1.3)

                ax.set_xticks(posiciones)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel('Magnitud de velocidad |V| [mm/s]')
                reo_label = reo.replace('02', '0.2%').replace('05', '0.5%')
                ax.set_title(f'Carbopol {reo_label}  n-{conc}  —  {zona_label}\n'
                            f'Comparacion PIV vs PTV por etapa  (magnitud |V|, '
                            f'whiskers = percentil {PCT_LO}-{PCT_HI})',
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

                # leyenda manual
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(facecolor=COLOR_PIV, alpha=0.45, label='PIV'),
                                Patch(facecolor=COLOR_PTV, alpha=0.45, label='PTV')],
                        fontsize=10)

                plt.tight_layout()
                ruta = os.path.join(OUT_DIR, zona)
                os.makedirs(ruta, exist_ok=True)
                ruta = os.path.join(ruta, f"box_piv_ptv_car{reo}_n{conc}_{zona}.png")
                plt.savefig(ruta, dpi=300, bbox_inches='tight'); plt.close()
                print(f"\n✅ {ruta}")

                # --- Guardar tabla resumen de esta muestra para imprimir al final ---
                reo_label = reo.replace('02', '0.2%').replace('05', '0.5%')
                titulo = f"Carbopol {reo_label}  n-{conc}  zona {zona}"
                tablas.append((titulo, filas_muestra))

    # ============================================================
    # TABLAS RESUMEN (todas al final)
    # ============================================================
    print(f"\n\n{'#'*60}\nRESUMEN — mediana |V| por muestra\n{'#'*60}")
    for titulo, filas_muestra in tablas:
        print(f"\n{titulo}")
        print(f"  {'Etapa':<11} {'Metodo':<6} {'Mediana |V|':>13}  {'n':>10}")
        print(f"  {'-'*11} {'-'*6} {'-'*13}  {'-'*10}")
        for etapa, metodo, med, n in filas_muestra:
            med_str = f"{med:7.2f} mm/s" if not np.isnan(med) else f"{'s/d':>12}"
            print(f"  {etapa:<11} {metodo:<6} {med_str:>13}  {n:>10,}")