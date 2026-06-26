"""
esp_overlay_piv_ptv.py
======================
Overlay PIV (campo promediado = BASE/fondo) + PTV (puntos, encima).

Filosofia de la figura:
  - El espectrograma PIV es la BASE continua: cubre todos los instantes.
  - El PTV se dibuja SOLO donde hay observacion real. Los huecos del PTV
    son informacion, no se rellenan.
  - PIV se muestra atenuado (alpha < 1) para que los puntos PTV resalten.

Entradas PTV (formato npz, una por zona, mismo esquema de nombres que el PIV):
  zona L        -> datos_esp_ptv/car-{reo}_n-{conc}_ptv_datos_suavizados.npz
  zona viga175  -> datos_esp_ptv/viga175_car-{reo}_n-{conc}_ptv_datos_suavizados.npz
  zona viga250  -> datos_esp_ptv/viga250_car-{reo}_n-{conc}_ptv_datos_suavizados.npz
Columnas del npz: t, s, v_paralela, track_id, x_mm, y_mm, vx_mm_s, vy_mm_s.
El caso SIN FIBRA no se estudia (no hay PTV para el).

Genera overlay + diferencia para las TRES polilineas y TODAS las muestras
(combinaciones reo x conc) configuradas.
"""

import os
import re
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from construir_caches import (
    cargar_cache_completo, natural_sort_key, nombre_base_carpeta,
    ss_linea_L, ss_viga,
)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACION
# ============================================================

ETAPAS_JSON = "etapas_polilinea.json"
CACHE_DIR   = "cache_completo"
OUTPUT_DIR  = "Esp_PIV-PTV/Overlay"
PTV_DIR     = "datos_esp_ptv"

# --- Muestras a procesar: todas las combinaciones reo x conc ---
# (sin fibra no se incluye: no hay datos PTV)
REOS  = ["02", "05"]
CONCS = ["0750", "1500", "3000"]

# --- Polilineas a procesar (zona_key, prefijo, etiqueta) ---
ZONAS = [
    ("L",       "",        "Salida de la L  (canal cerrado)"),
    ("viga175", "viga175", "Viga 175 mm"),
    ("viga250", "viga250", "Viga 250 mm"),
]

DT_OFFSET = 0.0

# --- Aspecto del overlay ---
PIV_ALPHA   = 0.8
PTV_SIZE    = 18
PTV_EDGE_LW = 0.15
DIBUJAR_TRAYECTORIAS = False

DPI = 160
PCT_LO, PCT_HI = 5, 95

# ============================================================
# CARGA PTV  (formato npz)
# ============================================================

def ruta_ptv(reo, conc, prefijo):
    """Ruta al npz de PTV de la zona, con el mismo esquema de nombres del PIV."""
    base = f"car-{reo}_n-{conc}_ptv_datos_suavizados.npz"
    nombre = f"{prefijo}_{base}" if prefijo else base
    return os.path.join(PTV_DIR, nombre)


def cargar_ptv(path):
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
    return t, s, v, tid


# ============================================================
# CAMPO PIV PROMEDIADO
# ============================================================

def construir_campo_promedio(etapas, reo, conc, prefijo, zona_key):
    nombre_grupo = f"car-{reo}_n-{conc}"
    carpetas = set()
    for clave in etapas.keys():
        carpeta = nombre_base_carpeta(clave)
        if re.search(rf'car-{reo}', carpeta) and re.search(rf'n-{conc}', carpeta):
            carpetas.add(carpeta)
    carpetas = sorted(carpetas, key=natural_sort_key)

    tomas_mat, tomas_tiem, tiempos_list = [], [], None
    for carpeta in carpetas:
        matriz_full, tiempos_full = cargar_cache_completo(carpeta, prefijo, CACHE_DIR, usar_magnitud=True)
        if matriz_full is None:
            print(f"  ⚠ Sin cache: {carpeta}"); continue
        clave = f"{carpeta}_{zona_key}"
        if clave not in etapas:
            print(f"  ⚠ Sin etapas: {clave}"); continue

        etapas_mat, etapas_tim, fronteras = [], [], []
        for etapa in ['inicio', 'transicion', 'cuasi']:
            i0, i1 = etapas[clave]['etapas'][etapa]
            i1 = min(i1, matriz_full.shape[0])
            if i0 >= i1:
                continue
            etapas_mat.append(matriz_full[i0:i1])
            etapas_tim.append(tiempos_full[i0:i1])
            fronteras.append(tiempos_full[i0:i1])
        if not etapas_mat:
            continue
        tomas_mat.append(np.concatenate(etapas_mat, axis=0))
        tomas_tiem.append(np.concatenate(etapas_tim, axis=0))
        if tiempos_list is None:
            tiempos_list = fronteras

    if not tomas_mat:
        raise SystemExit(f"Sin datos PIV para {nombre_grupo} zona {zona_key}")

    if len({m.shape for m in tomas_mat}) > 1:
        n_min = min(m.shape[0] for m in tomas_mat)
        tomas_mat  = [m[:n_min] for m in tomas_mat]
        tomas_tiem = [t[:n_min] for t in tomas_tiem]

    stack        = np.stack(tomas_mat, axis=0)
    matriz_full  = np.nanmean(stack, axis=0)
    tiempos_full = tomas_tiem[0]
    print(f"  {nombre_grupo}: {stack.shape[0]} toma(s) → campo medio {matriz_full.shape}")
    return matriz_full, tiempos_full, nombre_grupo, tiempos_list


# ============================================================
# COLORMAP COMPARTIDO
# ============================================================

def construir_cmap_y_norma(matriz_vel):
    v_min    = np.nanpercentile(matriz_vel, 5)
    v_max    = np.nanpercentile(matriz_vel, 95)
    v_median = np.nanmedian(matriz_vel)
    frac   = (v_median - v_min) / max(v_max - v_min, 1e-10)
    n_gris = max(1, int(frac * 256))
    n_rojo = max(1, 256 - n_gris)
    colores = (
        [plt.cm.Greys(x) for x in np.linspace(0.8, 0.2, n_gris)] +
        [plt.cm.Reds(x)  for x in np.linspace(0.2, 1.0, n_rojo)]
    )
    return LinearSegmentedColormap.from_list('gris_rojo', colores), v_min, v_max

def _agregar_divisores(ax, tiempos_list, color='white'):
    for tl in tiempos_list[:-1]:
        ax.axvline(tl[-1], color=color, lw=1.5, ls='--', alpha=0.8)


# ============================================================
# OVERLAY:  PIV base (difuminado) + PTV solo donde hay dato
# ============================================================

def graficar_overlay(matriz_vel, tiempos, tiempos_list, ss, t_ptv, s_ptv, v_ptv, tid_ptv,
                     cmap, vmin, vmax, nombre, zona_label, output_path):
    fig, ax = plt.subplots(figsize=(15, 6))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(tiempos, ss, matriz_vel.T,
                        cmap=cmap, norm=norm, shading='auto', alpha=PIV_ALPHA)

    if DIBUJAR_TRAYECTORIAS and tid_ptv is not None:
        for uid in np.unique(tid_ptv):
            m = tid_ptv == uid
            if m.sum() < 2:
                continue
            orden = np.argsort(t_ptv[m])
            ax.plot(t_ptv[m][orden], s_ptv[m][orden], '-', lw=0.6, alpha=0.45,
                    color=cmap(norm(np.nanmean(v_ptv[m]))), zorder=3)
    ax.scatter(t_ptv, s_ptv, c=v_ptv, cmap=cmap, norm=norm,
               s=PTV_SIZE, edgecolors='black', linewidths=PTV_EDGE_LW,
               alpha=0.75, zorder=4)

    cax = ax.inset_axes([0, 1.04, 1, 0.04])
    fig.colorbar(pcm, cax=cax, orientation='horizontal',
                 label='|V| [mm/s]   (fondo PIV = base,  puntos = PTV)')
    cax.xaxis.set_ticks_position('top'); cax.xaxis.set_label_position('top')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Posicion s [mm]')
    ax.set_ylim(ss[0], ss[-1]); ax.set_xlim(tiempos[0], tiempos[-1])
    ax.set_title(f'{nombre}  —  {zona_label}\n'
                 f'PIV (base) + PTV (puntos donde hay medicion)',
                 fontsize=11, fontweight='bold')
    _agregar_divisores(ax, tiempos_list, color='white')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  OK -> {os.path.basename(output_path)}")


def graficar_diferencia(matriz_vel, tiempos, ss, t_ptv, s_ptv, v_ptv,
                        nombre, zona_label, output_path):
    t_edges = np.concatenate([tiempos, [tiempos[-1] + (tiempos[-1]-tiempos[-2])]])
    s_edges = np.concatenate([ss, [ss[-1] + (ss[-1]-ss[-2])]])
    suma, _, _   = np.histogram2d(t_ptv, s_ptv, bins=[t_edges, s_edges], weights=v_ptv)
    cuenta, _, _ = np.histogram2d(t_ptv, s_ptv, bins=[t_edges, s_edges])
    with np.errstate(invalid='ignore'):
        v_ptv_grid = np.where(cuenta > 0, suma / cuenta, np.nan)
    diff = matriz_vel - v_ptv_grid
    lim  = np.nanpercentile(np.abs(diff), 95)
    fig, ax = plt.subplots(figsize=(15, 6))
    pcm = ax.pcolormesh(tiempos, ss, diff.T, cmap='RdBu_r',
                        shading='auto', vmin=-lim, vmax=lim)
    cax = ax.inset_axes([0, 1.04, 1, 0.04])
    fig.colorbar(pcm, cax=cax, orientation='horizontal',
                 label='|V|_PIV - |V|_PTV [mm/s]   (blanco = sin dato PTV o acuerdo)')
    cax.xaxis.set_ticks_position('top'); cax.xaxis.set_label_position('top')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Posicion s [mm]')
    ax.set_ylim(ss[0], ss[-1]); ax.set_xlim(tiempos[0], tiempos[-1])
    ax.set_title(f'{nombre}  —  {zona_label}\nDiferencia PIV - PTV',
                 fontsize=11, fontweight='bold')
    valido = ~np.isnan(diff)
    if valido.any():
        rms  = np.sqrt(np.nanmean(diff[valido]**2))
        bias = np.nanmean(diff[valido])
        ax.text(0.99, 0.02, f'RMS={rms:.1f} mm/s   sesgo={bias:+.1f} mm/s',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    out2 = output_path.replace('Overlay', 'Diferencia').replace('overlay', 'diferencia')
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    plt.savefig(out2, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  OK -> {os.path.basename(out2)}")


# ============================================================
# PROCESAR UNA MUESTRA EN UNA ZONA
# ============================================================

def procesar(etapas, reo, conc, zona_key, prefijo, zona_label):
    print(f"\n=== car-{reo}_n-{conc}  |  zona {zona_key} ===")

    ptv_path = ruta_ptv(reo, conc, prefijo)
    if not os.path.exists(ptv_path):
        print(f"  ⚠ Sin PTV: {ptv_path} (omitida)")
        return

    matriz_full, tiempos_full, nombre, tiempos_list = construir_campo_promedio(
        etapas, reo, conc, prefijo, zona_key)

    ss = ss_linea_L() if prefijo == "" else ss_viga()
    if len(ss) != matriz_full.shape[1]:
        ss = np.linspace(ss[0], ss[-1], matriz_full.shape[1])

    t_ptv, s_ptv, v_ptv, tid_ptv = cargar_ptv(ptv_path)
    t_ptv = t_ptv + DT_OFFSET

    print("  [Verificacion de ejes]")
    print(f"    PIV t: [{tiempos_full[0]:.3f}, {tiempos_full[-1]:.3f}] s, {len(tiempos_full)} instantes")
    print(f"    PTV t: [{t_ptv.min():.3f}, {t_ptv.max():.3f}] s, {len(np.unique(t_ptv))} instantes unicos")
    print(f"    PIV s: [{ss[0]:.2f}, {ss[-1]:.2f}] mm   PTV s: [{s_ptv.min():.2f}, {s_ptv.max():.2f}] mm")

    m = (t_ptv >= tiempos_full[0]) & (t_ptv <= tiempos_full[-1])
    n_fuera = (~m).sum()
    if n_fuera:
        print(f"    {n_fuera} obs PTV fuera de la ventana PIV (descartadas)")
    t_ptv, s_ptv, v_ptv = t_ptv[m], s_ptv[m], v_ptv[m]
    tid_ptv = tid_ptv[m] if tid_ptv is not None else None
    print(f"    PTV dibujado: {len(t_ptv)} obs")
    if len(v_ptv):
        lo, hi = np.nanpercentile(v_ptv, [PCT_LO, PCT_HI])
        m_pct = (v_ptv >= lo) & (v_ptv <= hi)
        t_ptv, s_ptv, v_ptv = t_ptv[m_pct], s_ptv[m_pct], v_ptv[m_pct]
        tid_ptv = tid_ptv[m_pct] if tid_ptv is not None else None
        print(f"    PTV tras recorte P{PCT_LO}-P{PCT_HI}: {len(t_ptv)} obs")

    lo_piv, hi_piv = np.nanpercentile(matriz_full, [PCT_LO, PCT_HI])
    matriz_full = np.clip(matriz_full, lo_piv, hi_piv)
    cmap, vmin, vmax = construir_cmap_y_norma(matriz_full)

    sufijo = zona_key  # L / viga175 / viga250
    ruta = os.path.join(OUTPUT_DIR, zona_key)
    os.makedirs(ruta, exist_ok=True)
    out = os.path.join(ruta, f"overlay_{nombre}_{sufijo}.png")
    graficar_overlay(matriz_full, tiempos_full, tiempos_list, ss, t_ptv, s_ptv, v_ptv,
                     tid_ptv, cmap, vmin, vmax, nombre, zona_label, out)
    graficar_diferencia(matriz_full, tiempos_full, ss, t_ptv, s_ptv, v_ptv,
                        nombre, zona_label, out)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ETAPAS_JSON, 'r', encoding='utf-8') as f:
        etapas = json.load(f)

    for reo in REOS:
        for conc in CONCS:
            for zona_key, prefijo, zona_label in ZONAS:
                try:
                    procesar(etapas, reo, conc, zona_key, prefijo, zona_label)
                except SystemExit as e:
                    print(f"  ⚠ {e}")
                except Exception as e:
                    print(f"  ⚠ Error en car-{reo}_n-{conc} zona {zona_key}: {e}")

    print(f"\nListo. Resultados en: {OUTPUT_DIR}")