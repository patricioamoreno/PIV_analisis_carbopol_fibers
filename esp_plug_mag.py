"""
esp_plug.py
===========
Genera espectrogramas de PLUG / NO-PLUG para:
  1. Salida de la L   (canal cerrado,  Cam4 660fps)
  2. Viga x=175 mm    (canal abierto,  Cam3 220fps)
  3. Viga x=250 mm    (canal abierto,  Cam2 220fps)

PIPELINE
--------
  1. Lee etapas desde etapas_polilinea.json (índices por zona y toma).
  2. Carga el caché completo por toma y zona (generado por construir_caches.py).
  3. Clasifica plug/no-plug como γ̇(s,t) = |dv_para/ds| ≤ umbral_PIV.
  4. Genera figuras: espectrograma velocidad + plug + v_media(t),
     más figuras de Reynolds efectivo e histograma de γ̇.

UMBRALES γ̇ (puramente PIV, Westerweel 1997)
--------------------------------------------
  δγ̇ = √2 · (ε_corr · mm_per_px / Δt) / (2 · Δy_PIV)   [s⁻¹]

  Zona      Cam  fps   Δy_PIV    car-02      car-05
  -------   ---  ---   ------    ------      ------
  L         4    660   0.748mm   2.917 s⁻¹  2.917 s⁻¹
  viga175   3    220   1.026mm   0.972 s⁻¹  0.972 s⁻¹
  viga250   2    220   2.051mm   0.486 s⁻¹  0.972 s⁻¹  ← Δy distinto por PIV_PARAMS
"""

import os
import re
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import uniform_filter1d

from construir_caches import (
    cargar_cache_completo,
    natural_sort_key,
    nombre_base_carpeta,
    ss_linea_L,
    ss_viga,
)

import sys
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN — editar aquí
# ============================================================

ETAPAS_JSON = "etapas_polilinea.json"
TIMELINE_JSON = "timeline_ref.json"
CACHE_DIR   = "cache_completo"
OUTPUT_DIR  = "espectrogramas_plug"


# True → procesa solo las tomas sin fibras (n-0000)
SOLO_0000 = False

DPI = 160

# ── Parámetros reológicos Herschel-Bulkley ────────────────────
REOLOGIA_REF = {
    "02": {"tau0": 46.0930, "k":  5.1857, "n": 0.5692},
    "05": {"tau0": 109.78,  "k": 11.6227, "n": 0.5623},
}

# ── Umbrales γ̇ por zona y reología [s⁻¹] ─────────────────────
UMBRAL_GAMMA = {
    'L':       {'02': 2.917, '05': 2.917},
    'viga175': {'02': 0.972, '05': 0.972},
    'viga250': {'02': 0.486, '05': 0.972},
}

# ── Suavizado espacial para el shear rate ────────────────────
VENTANA_SUAVIZADO = {
    'L':       7,
    'viga175': 1,
    'viga250': 1,
}

# Tomas excluidas del análisis (el caché se construye igual en construir_caches.py)
CARPETAS_EXCLUIR = [
]

# ============================================================
# UTILIDADES
# ============================================================

def agrupar_carpetas(etapas):
    """
    Agrupa las carpetas del JSON por (reo, conc).
    Retorna {('02','0000'): ['m71-toma-1-...', ...], ...}
    """
    grupos = {}
    for clave in etapas.keys():
        carpeta = nombre_base_carpeta(clave)
        m_reo   = re.search(r'car-(\d+)', carpeta)
        m_conc  = re.search(r'n-(\d+)',   carpeta)
        if not m_reo or not m_conc:
            continue
        key = (m_reo.group(1), m_conc.group(1))
        grupos.setdefault(key, set()).add(carpeta)
    return {k: sorted(v, key=natural_sort_key) for k, v in grupos.items()}


# ============================================================
# CÁLCULO DE PLUG / NO-PLUG
# ============================================================

def calcular_plug(matriz_vel, ss, tau0, k, n,
                  ventana=1, umbral_gamma=2.92,
                  output_path=None, nombre_grupo='', zona_label='', L_m=0.02):
    """
    Clasifica cada píxel (t, s) como plug (γ̇ ≤ umbral) o no-plug.

    Retorna
    -------
    mat_ratio   : (N_t, N_s)  γ̇ [s⁻¹]
    mat_plug    : (N_t, N_s)  bool — True = plug
    mat_re      : (N_t, N_s)  Reynolds efectivo
    mat_mueff   : (N_t, N_s)  viscosidad efectiva [Pa·s]
    hist_gamma  : (N_t, N_s)  γ̇ con signo (para histograma)
    """
    N_t, N_s = matriz_vel.shape
    if len(ss) != N_s:
        ss = np.linspace(ss[0], ss[-1], N_s)

    RHO       = 1000.0
    mat_ratio = np.full((N_t, N_s), np.nan)
    mat_re    = np.full((N_t, N_s), np.nan)
    mat_mueff = np.full((N_t, N_s), np.nan)
    hist_gamma = np.full((N_t, N_s), np.nan)

    for i in range(N_t):
        v = matriz_vel[i, :]
        if np.all(np.isnan(v)):
            continue

        nan_mask = np.isnan(v)
        v_fill   = v.copy()
        if nan_mask.any():
            idx_arr = np.arange(N_s)
            v_fill[nan_mask] = np.interp(
                idx_arr[nan_mask], idx_arr[~nan_mask], v_fill[~nan_mask])

        v_s       = uniform_filter1d(v_fill, size=min(ventana, N_s), mode='nearest')
        gamma_dot = np.abs(np.gradient(v_s, ss))
        g_signed  = np.gradient(v_s, ss)

        if L_m is not None:
            L_frame = L_m
        else:
            valid = ~nan_mask
            L_frame = ((ss[valid][-1] - ss[valid][0]) / 1000
                       if valid.sum() > 1 else ss[-1] / 1000)

        ratio           = gamma_dot.copy()
        ratio[nan_mask] = np.nan
        mat_ratio[i, :] = ratio
        hist_gamma[i, :] = g_signed

        mu_eff = np.where(gamma_dot > 0,
                          (tau0 + k * np.power(gamma_dot, n)) / gamma_dot,
                          np.nan)
        mu_eff[nan_mask] = np.nan
        mat_mueff[i, :]  = mu_eff

        v_ms             = np.abs(v_s) / 1000
        re_row           = (RHO * v_ms * L_frame) / mu_eff
        re_row[nan_mask] = np.nan
        mat_re[i, :]     = re_row

    mat_plug = mat_ratio <= umbral_gamma
    # Donde no hay dato (NaN en ratio), la comparación da False y se pintaría
    # como no-plug. Forzamos que quede fuera de la clasificación: el plug se
    # representa con máscara, y las celdas sin material no son ni plug ni
    # no-plug. Devolvemos mat_plug como float con NaN para que el graficado
    # y los promedios las excluyan.
    mat_plug = mat_plug.astype(float)
    mat_plug[np.isnan(mat_ratio)] = np.nan
    return mat_ratio, mat_plug, mat_re, mat_mueff, hist_gamma


# ============================================================
# GRAFICADO
# ============================================================

def _agregar_divisores(ax, tiempos_list, color='white'):
    for tl in tiempos_list[:-1]:
        ax.axvline(tl[-1], color=color, lw=1.5, ls='--', alpha=0.7)


def graficar_histograma_gamma(hist_gamma, umbral_gamma, nombre_grupo, zona_label, output_path):
    valores = hist_gamma[~np.isnan(hist_gamma)].flatten()
    if len(valores) == 0:
        return

    p5, p95 = np.percentile(valores, [5, 95])
    valores  = valores[(valores > p5) & (valores < p95)]

    plt.figure(figsize=(6, 4))
    plt.hist(valores, bins=200, density=True, color='steelblue', alpha=0.7)
    plt.xlabel('γ̇ [s⁻¹]')
    plt.ylabel('Densidad')
    plt.axvspan(-umbral_gamma, umbral_gamma, alpha=0.1, color='gray', label='zona plug')
    plt.axvline( umbral_gamma, color='red', ls='--', lw=1.5, label=f'+{umbral_gamma} s⁻¹')
    plt.axvline(-umbral_gamma, color='red', ls='--', lw=1.5, label=f'-{umbral_gamma} s⁻¹')
    plt.title(f'Distribución de γ̇ — {nombre_grupo}, {zona_label}')
    plt.legend()
    if output_path:
        hist_path = output_path.replace('PLUG', 'HIST_GAMMA').replace('.png', '_hist_gamma.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"    ✅ {os.path.basename(hist_path)}", flush=True)
    plt.close()


def graficar(matriz_vel, mat_plug, mat_ratio, tiempos_full, tiempos_list,
             ss, nombre_grupo, zona_label, reo_ref, output_path, umbral_gamma=2.92,
             tomas_vivas=None, n_tomas=None):
    """
    Figura de 3 paneles:
      [top]    espectrograma velocidad (gris→rojo)
      [medio]  espectrograma γ̇ / plug  (gris=plug, rojo=no-plug)
      [bottom] v_media(t) con perfiles individuales
    """
    fig = plt.figure(figsize=(14, 13))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[2.5, 2.5, 1.2], hspace=0.45)
    ax_vel  = fig.add_subplot(gs[0])
    ax_plug = fig.add_subplot(gs[1], sharex=ax_vel)
    ax_v    = fig.add_subplot(gs[2], sharex=ax_vel)

    # ── Panel 1: velocidad ────────────────────────────────────
    v_min  = np.nanpercentile(matriz_vel, 2)
    v_max  = np.nanpercentile(matriz_vel, 98)
    v_median = np.nanmedian(matriz_vel)
    frac   = (v_median - v_min) / max(v_max - v_min, 1e-10)
    n_gris = max(1, int(frac * 256))
    n_rojo = max(1, 256 - n_gris)
    colores_vel = (
        [plt.cm.Greys(x) for x in np.linspace(0.8, 0.2, n_gris)] +
        [plt.cm.Reds(x)  for x in np.linspace(0.2, 1.0, n_rojo)]
    )
    cmap_vel = LinearSegmentedColormap.from_list('gris_rojo', colores_vel)
    pcm1 = ax_vel.pcolormesh(tiempos_full, ss, matriz_vel.T,
                             cmap=cmap_vel, shading='auto', vmin=v_min, vmax=v_max)
    cax1 = ax_vel.inset_axes([0, 1.02, 1, 0.04])
    fig.colorbar(pcm1, cax=cax1, orientation='horizontal', label='|V| [mm/s]')
    cax1.xaxis.set_ticks_position('top')
    cax1.xaxis.set_label_position('top')
    ax_vel.set_ylabel('Posición s [mm]', fontsize=10)
    ax_vel.set_ylim(ss[0], ss[-1])
    ax_vel.set_title(f'{nombre_grupo}  —  {zona_label}', fontsize=11, fontweight='bold')
    _agregar_divisores(ax_vel, tiempos_list, 'white')

    # ── Panel 2: plug / no-plug ───────────────────────────────
    colores_plug = (
        [plt.cm.Greys(x) for x in np.linspace(0.65, 0.10, 128)] +
        [plt.cm.Reds(x)  for x in np.linspace(0.20, 1.00, 128)]
    )
    cmap_plug = mcolors.LinearSegmentedColormap.from_list('plug_cont', colores_plug)
    cmap_plug.set_bad(color='white')
    vmax_plot  = umbral_gamma * 2
    ratio_plot = np.clip(mat_ratio, 0, vmax_plot)
    pcm2 = ax_plug.pcolormesh(tiempos_full, ss, ratio_plot.T,
                              cmap=cmap_plug, shading='auto', vmin=0, vmax=vmax_plot)
    cax2 = ax_plug.inset_axes([0, 1.02, 1, 0.04])
    cb2  = fig.colorbar(pcm2, cax=cax2, orientation='horizontal',
                        ticks=[0, umbral_gamma, vmax_plot])
    cb2.ax.set_xticklabels(
        ['0 (plug puro)', f'{umbral_gamma:.2f} s⁻¹ (umbral PIV)',
         f'≥{vmax_plot:.2f} s⁻¹ (no-plug)'], fontsize=8)
    cax2.xaxis.set_ticks_position('top')
    cax2.xaxis.set_label_position('top')
    ax_plug.set_ylabel('Posición s [mm]', fontsize=10)
    ax_plug.set_ylim(ss[0], ss[-1])
    ax_plug.set_title(
        f'γ̇ = |dv/ds|  [s⁻¹]   (umbral PIV={umbral_gamma:.2f} s⁻¹,  τ₀={reo_ref["tau0"]} Pa)',
        fontsize=10)
    ax_plug.contour(tiempos_full, ss, ratio_plot.T,
                    levels=[umbral_gamma], colors='white', linewidths=0.8, alpha=0.7)
    _agregar_divisores(ax_plug, tiempos_list, 'white')

    # ── Panel 3: v_media(t) ───────────────────────────────────
    v_mean = np.nanmean(matriz_vel, axis=1)
    for j in range(0, matriz_vel.shape[1], max(1, matriz_vel.shape[1] // 15)):
        ax_v.plot(tiempos_full, matriz_vel[:, j],
                  color='#CCCCCC', lw=0.6, alpha=0.5, zorder=1)
    ax_v.plot(tiempos_full, v_mean, color='steelblue', lw=2.0, label='v media', zorder=3)
    ax_v.axhline(v_median, color='red', lw=1, ls='--', alpha=0.7, label=f'mediana={v_median:.1f}')
    _agregar_divisores(ax_v, tiempos_list, 'gray')
    ax_v.set_xlabel('Tiempo [s]',     fontsize=10)
    ax_v.set_ylabel('|V| [mm/s]', fontsize=10)
    ax_v.legend(fontsize=8)
    ax_v.set_xlim(tiempos_full[0], np.ceil(tiempos_full[-1]))
    ax_v.grid(alpha=0.3)

    # Franja: nº de tomas que sostienen el promedio en cada instante.
    # Avisa visualmente dónde la cola se apoya en pocas tomas.
    if tomas_vivas is not None and n_tomas is not None and n_tomas > 1:
        ax_n = ax_v.twinx()
        ax_n.fill_between(tiempos_full, 0, tomas_vivas,
                          step='mid', color='darkorange', alpha=0.15, zorder=0)
        ax_n.plot(tiempos_full, tomas_vivas, color='darkorange',
                  lw=1.2, alpha=0.7, drawstyle='steps-mid',
                  label='nº tomas')
        ax_n.set_ylabel('nº tomas activas', fontsize=9, color='darkorange')
        ax_n.set_ylim(0, n_tomas + 0.5)
        ax_n.set_yticks(range(0, n_tomas + 1))
        ax_n.tick_params(axis='y', labelcolor='darkorange', labelsize=8)

    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✅ {os.path.basename(output_path)}", flush=True)


def graficar_reynolds(mat_re, mat_mueff, tiempos_full, tiempos_list,
                      ss, nombre_grupo, zona_label, output_path):
    """
    Figura de 3 paneles: μ_eff, Re_eff, Re_medio(t).
    """
    fig = plt.figure(figsize=(14, 13))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[2.5, 2.5, 1.2], hspace=0.45)
    ax_mu  = fig.add_subplot(gs[0])
    ax_re  = fig.add_subplot(gs[1], sharex=ax_mu)
    ax_t   = fig.add_subplot(gs[2], sharex=ax_mu)

    # ── μ_eff ─────────────────────────────────────────────────
    vmax_mu = np.nanpercentile(mat_mueff, 95)
    pcm_mu  = ax_mu.pcolormesh(tiempos_full, ss,
                               np.clip(mat_mueff, 0, vmax_mu).T,
                               cmap='viridis', shading='auto', vmin=0, vmax=vmax_mu)
    cax_mu = ax_mu.inset_axes([0, 1.02, 1, 0.04])
    fig.colorbar(pcm_mu, cax=cax_mu, orientation='horizontal', label='μ_eff [Pa·s]')
    cax_mu.xaxis.set_ticks_position('top')
    cax_mu.xaxis.set_label_position('top')
    ax_mu.set_ylabel('Posición s [mm]', fontsize=10)
    ax_mu.set_ylim(ss[0], ss[-1])
    ax_mu.set_title(f'{nombre_grupo}  —  {zona_label}\nViscosidad efectiva  μ_eff',
                    fontsize=11, fontweight='bold')
    _agregar_divisores(ax_mu, tiempos_list, 'white')

    # ── Re_eff ────────────────────────────────────────────────
    vmax_re = np.nanpercentile(mat_re, 95)
    pcm_re  = ax_re.pcolormesh(tiempos_full, ss,
                               np.clip(mat_re, 0, vmax_re).T,
                               cmap='viridis', shading='auto', vmin=0, vmax=vmax_re)
    cax_re = ax_re.inset_axes([0, 1.02, 1, 0.04])
    fig.colorbar(pcm_re, cax=cax_re, orientation='horizontal', label='Re_eff [-]')
    cax_re.xaxis.set_ticks_position('top')
    cax_re.xaxis.set_label_position('top')
    ax_re.set_ylabel('Posición s [mm]', fontsize=10)
    ax_re.set_ylim(ss[0], ss[-1])
    ax_re.set_title('Reynolds efectivo  Re = ρvL/μ_eff', fontsize=11, fontweight='bold')
    _agregar_divisores(ax_re, tiempos_list, 'white')

    # ── Re_medio(t) ───────────────────────────────────────────
    ax_t.plot(tiempos_full, np.nanmean(mat_re, axis=1),
              color='steelblue', lw=2.0, label='Re medio')
    ax_t.plot(tiempos_full, np.nanmax(mat_re,  axis=1),
              color='red', lw=1.0, ls='--', alpha=0.7, label='Re máx')
    ax_t.axhline(1.0, color='gray', lw=1, ls=':', label='Re=1 (Stokes)')
    _agregar_divisores(ax_t, tiempos_list, 'gray')
    ax_t.set_xlabel('Tiempo [s]',  fontsize=10)
    ax_t.set_ylabel('Re_eff [-]',  fontsize=10)
    ax_t.legend(fontsize=8)
    ax_t.set_xlim(tiempos_full[0], tiempos_full[-1])
    ax_t.grid(alpha=0.3)

    re_path = output_path.replace('PLUG', 'REYNOLDS').replace('.png', '_reynolds.png')
    plt.savefig(re_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✅ {os.path.basename(re_path)}", flush=True)


def graficar_fraccion_plug(f_plug, tiempos_full, tiempos_list, ss,
                           nombre_grupo, zona_label, n_tomas, output_path):
    """
    Vía B: mapa de la fracción de tomas en plug en cada (t, s).
    0 = ninguna toma en plug (no-plug), 1 = todas en plug.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[2.8, 1.2], hspace=0.4)
    ax_f = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1], sharex=ax_f)

    cmap = plt.cm.RdYlGn   # rojo=no-plug(0) → verde=plug(1)
    cmap = cmap.copy(); cmap.set_bad(color='white')
    pcm = ax_f.pcolormesh(tiempos_full, ss, f_plug.T,
                          cmap=cmap, shading='auto', vmin=0, vmax=1)
    cax = ax_f.inset_axes([0, 1.02, 1, 0.04])
    fig.colorbar(pcm, cax=cax, orientation='horizontal',
                 label='fracción de tomas en plug', ticks=[0, 0.5, 1])
    cax.xaxis.set_ticks_position('top'); cax.xaxis.set_label_position('top')
    ax_f.set_ylabel('Posición s [mm]', fontsize=10)
    ax_f.set_ylim(ss[0], ss[-1])
    ax_f.set_title(f'{nombre_grupo}  —  {zona_label}\n'
                   f'Fracción de plug (promedio sobre {n_tomas} toma(s))',
                   fontsize=11, fontweight='bold')
    _agregar_divisores(ax_f, tiempos_list, 'black')

    ax_t.plot(tiempos_full, np.nanmean(f_plug, axis=1),
              color='seagreen', lw=2.0, label='fracción media en s')
    ax_t.axhline(0.5, color='gray', lw=1, ls=':', label='50%')
    _agregar_divisores(ax_t, tiempos_list, 'gray')
    ax_t.set_xlabel('Tiempo [s]', fontsize=10)
    ax_t.set_ylabel('frac. plug [-]', fontsize=10)
    ax_t.set_ylim(0, 1)
    ax_t.legend(fontsize=8)
    ax_t.set_xlim(tiempos_full[0], tiempos_full[-1])
    ax_t.grid(alpha=0.3)

    f_path = output_path.replace('PLUG', 'FRAC_PLUG').replace('.png', '_frac_plug.png')
    os.makedirs(os.path.dirname(f_path), exist_ok=True)
    plt.savefig(f_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✅ {os.path.basename(f_path)}", flush=True)


# ============================================================
# PIPELINE POR ZONA
# ============================================================

def interpolar_a_grilla(mat, t_toma, t_ref):
    """
    Remuestrea una toma (N_t, N_s) a la grilla temporal fija t_ref (M,).
    Interpola en el tiempo, por cada posición s. Fuera del rango temporal
    de la toma deja NaN (no extrapola): así el material ausente al final
    o el arranque tardío quedan en blanco en el espectrograma.

    Devuelve (M, N_s).
    """
    t_toma = np.asarray(t_toma, dtype=float)
    t_toma = t_toma - t_toma[0]            # normalizar a 0 (igual que t_ref)
    M      = len(t_ref)
    N_s    = mat.shape[1]
    out    = np.full((M, N_s), np.nan)

    t_min, t_max = t_toma[0], t_toma[-1]
    dentro = (t_ref >= t_min) & (t_ref <= t_max)

    for j in range(N_s):
        col = mat[:, j]
        valido = np.isfinite(col)
        if valido.sum() < 2:
            continue
        # interp solo dentro del rango temporal con dato de esta toma
        interp_vals = np.interp(t_ref[dentro], t_toma[valido], col[valido])
        out[dentro, j] = interp_vals
        # re-NaN donde la toma no tenía dato cerca (huecos largos): si el
        # punto de t_ref cae a más de 1.5 pasos del dato válido más cercano,
        # se considera sin material.
        if not valido.all():
            t_val = t_toma[valido]
            paso  = np.median(np.diff(t_toma)) if len(t_toma) > 1 else 0.0
            for idx in np.where(dentro)[0]:
                if np.min(np.abs(t_val - t_ref[idx])) > 1.5 * paso:
                    out[idx, j] = np.nan
    return out


def procesar_zona(etapas, grupos, ss, prefijo, zona_label,
                  output_suffix, L_m=0.02, zona_key='L', timeline=None):
    dir_plug      = os.path.join(OUTPUT_DIR, "PLUG");       os.makedirs(dir_plug,      exist_ok=True)
    dir_reynolds  = os.path.join(OUTPUT_DIR, "REYNOLDS");   os.makedirs(dir_reynolds,  exist_ok=True)
    dir_hist      = os.path.join(OUTPUT_DIR, "HIST_GAMMA"); os.makedirs(dir_hist,      exist_ok=True)
    dir_frac      = os.path.join(OUTPUT_DIR, "FRAC_PLUG");  os.makedirs(dir_frac,      exist_ok=True)
    dir_cache_plug = os.path.join(OUTPUT_DIR, "cache_plug"); os.makedirs(dir_cache_plug, exist_ok=True)

    for (reo, conc), carpetas in sorted(grupos.items()):
        if SOLO_0000 and conc != '0000':
            continue
        if reo not in REOLOGIA_REF:
            print(f"  ⚠ Reología car-{reo} no definida.", flush=True)
            continue

        reo_ref      = REOLOGIA_REF[reo]
        nombre_grupo = f"car-{reo}_n-{conc}"
        umbral_zona  = UMBRAL_GAMMA[zona_key][reo]
        ventana_zona = VENTANA_SUAVIZADO[zona_key]

        print(f"\n  → {nombre_grupo}", flush=True)
        print(f"    umbral γ̇={umbral_zona:.2f} s⁻¹  ventana_suav={ventana_zona}", flush=True)

        # Grilla temporal fija de esta reología (eje x del espectrograma).
        if timeline is None or reo not in timeline:
            print(f"    ⚠ Sin timeline para car-{reo}; corre generar_timeline.py", flush=True)
            continue
        t_ref = np.asarray(timeline[reo]['t_ref'], dtype=float)

        # Cada toma se interpola a t_ref; las etapas se leen del JSON solo
        # para los divisores (tiempos), NO para recortar el eje.
        tomas_grilla = []      # lista de (M, N_s) ya en la grilla t_ref
        fronteras_tiempo = None  # tiempos de fin de inicio / transición (divisores)

        for carpeta in carpetas:
            if carpeta in CARPETAS_EXCLUIR:
                print(f"    ⚠ Excluida: {carpeta}", flush=True)
                continue

            matriz_full, tiempos_full = cargar_cache_completo(carpeta, prefijo, CACHE_DIR, usar_magnitud=True)
            if matriz_full is None:
                print(f"    ⚠ Sin caché: {carpeta} — corre construir_caches.py primero", flush=True)
                continue

            clave = f"{carpeta}_{zona_key}"
            if clave not in etapas:
                print(f"    ⚠ Sin etapas en JSON: {clave}", flush=True)
                continue

            # Interpolar toda la toma a la grilla fija
            mat_g = interpolar_a_grilla(matriz_full, tiempos_full, t_ref)
            tomas_grilla.append(mat_g)
            print(f"    {carpeta} {zona_key}: interpolada a {len(t_ref)} puntos", flush=True)

            # Fronteras de etapa (divisores) — del JSON, como tiempos sobre t_ref.
            # Se toman de la primera toma válida del grupo (etapas promediadas,
            # comunes a la reología).
            if fronteras_tiempo is None:
                t_norm = np.asarray(tiempos_full, dtype=float)
                t_norm = t_norm - t_norm[0]
                fronteras_tiempo = []
                for etapa in ['inicio', 'transicion']:   # fin de inicio y de transición
                    _, i1 = etapas[clave]['etapas'][etapa]
                    i1 = min(i1, len(t_norm) - 1)
                    fronteras_tiempo.append(float(t_norm[i1]))

        if not tomas_grilla:
            print(f"    ⚠ Sin datos para {nombre_grupo}", flush=True)
            continue

        # Apilar (N_tomas, M, N_s) y promediar sobre tomas en la grilla común.
        stack        = np.stack(tomas_grilla, axis=0)
        matriz_full  = np.nanmean(stack, axis=0)   # campo de velocidad promedio
        tiempos_full = t_ref                       # eje x fijo de la reología
        n_tomas      = stack.shape[0]
        # divisores como lista de "tiempos de frontera" para _agregar_divisores
        tiempos_list = ([np.array([f]) for f in fronteras_tiempo] +
                        [np.array([t_ref[-1]])]) if fronteras_tiempo else [np.array([t_ref[-1]])]
        # nº de tomas con material en cada instante
        tomas_vivas  = np.sum(np.any(np.isfinite(stack), axis=2), axis=0)
        print(f"    Promediando {n_tomas} toma(s) sobre grilla fija "
              f"(0→{t_ref[-1]:.1f}s, cola mín {int(tomas_vivas.min())} toma/s)", flush=True)

        # Calcular plug
        output_path = os.path.join(dir_plug, f"plug_{output_suffix}_{nombre_grupo}.png")
        mat_ratio, mat_plug, mat_re, mat_mueff, hist_gamma = calcular_plug(
            matriz_full, ss,
            tau0=reo_ref['tau0'], k=reo_ref['k'], n=reo_ref['n'],
            ventana=ventana_zona, umbral_gamma=umbral_zona,
            output_path=output_path, nombre_grupo=nombre_grupo,
            zona_label=zona_label, L_m=L_m
        )

        # frac_plug del campo PROMEDIO. mat_plug ya es float con NaN donde no
        # hay material (1.0=plug, 0.0=no-plug, NaN=sin dato), así que basta
        # promediar ignorando NaN.
        valido    = np.isfinite(mat_plug)
        frac_plug = (np.nansum(mat_plug) / np.sum(valido)
                     if np.sum(valido) > 0 else np.nan)
        print(f"    Plug global (campo medio): {frac_plug*100:.1f}%", flush=True)

        # ── Vía B: plug por toma, luego promediar la máscara ─────────
        # f_plug(t,s) ∈ [0,1] = proporción de tomas en plug en ese (t,s)
        plug_por_toma = np.full(stack.shape, np.nan, dtype=float)
        for it in range(n_tomas):
            _, mp_it, _, _, _ = calcular_plug(
                stack[it], ss,
                tau0=reo_ref['tau0'], k=reo_ref['k'], n=reo_ref['n'],
                ventana=ventana_zona, umbral_gamma=umbral_zona,
                L_m=L_m
            )
            # mp_it ya es float con NaN donde no había material; se usa tal cual.
            plug_por_toma[it] = mp_it
        f_plug = np.nanmean(plug_por_toma, axis=0)   # (N_t, N_s)

        # Guardar caché plug
        np.savez_compressed(
            os.path.join(dir_cache_plug, f"plug_{output_suffix}_{nombre_grupo}.npz"),
            mat_plug=mat_plug, mat_ratio=mat_ratio,
            matriz_vel=matriz_full, tiempos=tiempos_full,
            f_plug=f_plug, n_tomas=n_tomas
        )
        print(f"    💾 Caché plug guardado: plug_{output_suffix}_{nombre_grupo}.npz")

        # Figuras
        graficar(
            matriz_full, mat_plug, mat_ratio, tiempos_full, tiempos_list,
            ss, nombre_grupo, zona_label, reo_ref, output_path,
            umbral_gamma=umbral_zona, tomas_vivas=tomas_vivas, n_tomas=n_tomas
        )
        graficar_reynolds(
            mat_re, mat_mueff, tiempos_full, tiempos_list,
            ss, nombre_grupo, zona_label, output_path
        )
        graficar_histograma_gamma(
            hist_gamma, umbral_zona, nombre_grupo, zona_label, output_path
        )
        graficar_fraccion_plug(
            f_plug, tiempos_full, tiempos_list, ss,
            nombre_grupo, zona_label, n_tomas, output_path
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(ETAPAS_JSON, 'r', encoding='utf-8') as f:
        etapas = json.load(f)

    try:
        with open(TIMELINE_JSON, 'r', encoding='utf-8') as f:
            timeline = json.load(f)
        print(f"✅ Timeline cargado: " +
              ", ".join(f"car-{r} (0→{timeline[r]['t_final']:.1f}s)" for r in timeline))
    except FileNotFoundError:
        print(f"⚠ No existe {TIMELINE_JSON}. Corre generar_timeline.py primero.")
        raise SystemExit(1)

    grupos = agrupar_carpetas(etapas)
    print(f"✅ {len(grupos)} grupo(s) encontrados\n")

    # Zona L
    procesar_zona(etapas, grupos, ss_linea_L(),
                  prefijo="", zona_label="Salida de la L  (canal cerrado)",
                  output_suffix="L", L_m=0.02, zona_key='L', timeline=timeline)

    # Vigas
    for x_viga in [175, 250]:
        procesar_zona(etapas, grupos, ss_viga(),
                      prefijo=f"viga{x_viga}",
                      zona_label=f"Viga  x={x_viga} mm  (canal abierto)",
                      output_suffix=f"viga{x_viga}",
                      L_m=None,
                      zona_key=f"viga{x_viga}", timeline=timeline)

    print(f"\n✅ Listo. Resultados en: {OUTPUT_DIR}")