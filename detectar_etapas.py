"""
detectar_etapas.py
==================
Función para detectar las tres etapas del flujo PIV:
  - Inicio             : t=0 → peak de velocidad
  - Transición         : peak → inicio cuasi-estacionario
  - Cuasi-estacionario : desde que la tasa de cambio normalizada es estable

Criterio cuasi-estacionario (V3 — pendiente normalizada):
  Se busca el PRIMER instante en que |dV/dt| / V_peak < EPS_CUASI
  de forma sostenida durante al menos VF_CONFIRM frames consecutivos.
  Este criterio tiene fundamento físico directo: el cuasi-estacionario
  comienza cuando la aceleración local es despreciable respecto a la
  escala del evento (< 2% del valor pico por segundo).

  Referencia metodológica: análogo al criterio de convergencia temporal
  de Zhao et al. (2021), Int. J. Numer. Methods Fluids, adaptado al
  contexto experimental PIV.

Uso:
  from detectar_etapas import detectar_etapas, graficar_etapas, natural_sort_key
  resultado = detectar_etapas(tiempos, v_medias)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# PARÁMETROS AJUSTABLES
# ==========================================

VENTANA_SUAVIZADO = 50    # frames de media móvil centrada para suavizar la serie
T_IGNORAR         = 0.5   # segundos a ignorar al inicio (tirón de compuerta)
MARGEN_POST_PEAK  = 30    # frames mínimos de margen después del peak
FALLBACK_PCT      = 0.60  # fracción del total usada como fallback si no converge
EPS_CUASI         = 0.02  # s⁻¹ — umbral de tasa normalizada |dV/dt|/V_peak
VF_CONFIRM        = 20    # frames consecutivos requeridos bajo EPS_CUASI

# ==========================================
# UTILIDADES
# ==========================================

def natural_sort_key(s):
    """Clave de ordenamiento natural (e.g. frame_9 < frame_10)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def detectar_etapas(tiempos, v_medias,
                    ventana_suavizado=VENTANA_SUAVIZADO,
                    t_ignorar=T_IGNORAR,
                    margen_post_peak=MARGEN_POST_PEAK,
                    eps_cuasi=EPS_CUASI,
                    vf_confirm=VF_CONFIRM,
                    nombre_carpeta=""):
    """
    Detecta las tres etapas del flujo a partir de la serie temporal.

    Parámetros
    ----------
    tiempos           : array — timestamps reales en segundos
    v_medias          : list  — velocidad media por frame [mm/s]
    ventana_suavizado : int   — frames de media móvil centrada (suavizado previo)
    t_ignorar         : float — segundos a ignorar al inicio (tirón de compuerta)
    margen_post_peak  : int   — frames mínimos de margen después del peak
    eps_cuasi         : float — umbral de tasa normalizada |dV/dt|/V_peak [s⁻¹]
    vf_confirm        : int   — frames consecutivos requeridos bajo eps_cuasi
    nombre_carpeta    : str   — nombre para mensajes de advertencia

    Retorna
    -------
    dict con:
      'tiempos'    : array timestamps reales [s]
      'v_smooth'   : serie suavizada
      'idx_peak'   : índice del peak (fin del inicio)
      'idx_quasi'  : índice inicio cuasi-estacionario
      't_peak'     : tiempo real del peak [s]
      't_quasi'    : tiempo real del cuasi-estacionario [s]
      'fallback'   : True si no se encontró ventana estable (revisar manualmente)
      'etapas'     : dict con tuplas (i_ini, i_fin) por etapa
    """
    tiempos  = np.array(tiempos)
    v_arr    = np.asarray(v_medias, dtype=float)
    n_total  = len(tiempos)   # ← largo COMPLETO (para que 'cuasi' cubra todo el eje)

    # ── Paso 0: recortar cola de NaN (frames sin material en la línea) ────
    # Los cachés ahora preservan todos los timestamps, con filas NaN cuando
    # el material ya no llega a la polilínea (típico al final de viga250).
    # La DETECCIÓN de peak/cuasi debe correr solo sobre el tramo con datos;
    # de lo contrario la cola vacía desplaza el peak o fuerza el fallback.
    # El índice 'n' que se reporta sigue siendo n_total, de modo que la
    # etapa 'cuasi' = (idx_quasi, n_total) abarca toda la corrida.
    finitos = np.where(np.isfinite(v_arr))[0]
    if len(finitos) == 0:
        # No hay ningún dato válido: devolver fallback trivial.
        return {
            'tiempos': tiempos, 'v_smooth': v_arr,
            'idx_peak': 0, 'idx_quasi': min(1, n_total - 1),
            't_peak': float(tiempos[0]), 't_quasi': float(tiempos[min(1, n_total-1)]),
            'fallback': True,
            'etapas': {'inicio': (0, 0),
                       'transicion': (0, min(1, n_total - 1)),
                       'cuasi': (min(1, n_total - 1), n_total)},
        }
    ult_valido = int(finitos[-1])           # último frame con material
    tiempos_d  = tiempos[:ult_valido + 1]   # tramo de detección
    v_medias   = v_arr[:ult_valido + 1]

    v_series = pd.Series(v_medias).interpolate()

    # ── Paso 1: Suavizado ─────────────────────────────────────────────────
    v_smooth = v_series.rolling(window=ventana_suavizado, center=True).mean()
    v_smooth = v_smooth.fillna(v_series).values

    n              = len(tiempos_d)   # ← largo del tramo CON datos (detección)
    tiempos        = tiempos_d        # las referencias siguientes operan sobre el tramo válido
    fallback_usado = False

    # ── Paso 2: Peak (fin del inicio) ─────────────────────────────────────
    # Se ignoran los primeros t_ignorar segundos (tirón de compuerta).
    # Se busca el último pico local que supere el 50% del máximo global.
    frames_ignorar = np.searchsorted(tiempos, tiempos[0] + t_ignorar)
    frames_ignorar = max(frames_ignorar, 1)

    v_busqueda   = v_smooth[frames_ignorar:]
    v_max_global = np.nanmax(v_busqueda)
    umbral_peak  = 0.50 * v_max_global

    from scipy.signal import find_peaks
    picos, _ = find_peaks(v_busqueda, height=umbral_peak, distance=10)

    if len(picos) > 0:
        idx_peak = int(picos[-1]) + frames_ignorar
    else:
        idx_peak = int(np.nanargmax(v_busqueda)) + frames_ignorar

    # ── Paso 3: Cuasi-estacionario (criterio V3 — pendiente normalizada) ──
    # Se calcula |dV/dt| / V_peak frame a frame sobre la curva suavizada.
    # El cuasi arranca en el PRIMER instante donde esa tasa se mantiene
    # bajo eps_cuasi durante vf_confirm frames consecutivos.
    # Este criterio es robusto en la cola porque cuando V es pequeño y
    # estable, dV/dt también lo es — no hay división por números pequeños.
    dt        = np.diff(tiempos)
    dv_dt     = np.diff(v_smooth) / dt
    dv_dt     = np.append(dv_dt, dv_dt[-1])   # mismo largo que tiempos
    v_peak    = np.nanmax(v_smooth)
    tasa_norm = np.abs(dv_dt) / v_peak

    idx_quasi       = None
    inicio_busqueda = idx_peak + margen_post_peak

    for i in range(inicio_busqueda, n - vf_confirm):
        if np.all(tasa_norm[i : i + vf_confirm] < eps_cuasi):
            idx_quasi = i
            break

    if idx_quasi is None:
        idx_quasi = inicio_busqueda

    # ── Fallback si quedan muy pocos frames de cuasi ──────────────────────
    frames_restantes = n - idx_quasi
    if frames_restantes < vf_confirm:
        idx_quasi_original = idx_quasi
        idx_quasi          = min(int(n * FALLBACK_PCT), n - 1)
        fallback_usado     = True
        label              = nombre_carpeta or "carpeta desconocida"
        print(
            f"\n⚠️  ADVERTENCIA [{label}]: cuasi detectado muy tarde "
            f"(idx={idx_quasi_original}, solo {frames_restantes} frames).\n"
            f"   → Fallback: idx={idx_quasi} "
            f"(t={tiempos[idx_quasi]:.2f}s, {FALLBACK_PCT*100:.0f}% del total).\n"
            f"   → Revisa este caso manualmente."
        )

    # ── Validación: cuasi no puede estar antes del peak ───────────────────
    if idx_quasi <= idx_peak:
        idx_quasi = min(idx_peak + margen_post_peak, n - 1)
        print(
            f"⚠️  ADVERTENCIA [{nombre_carpeta}]: idx_quasi <= idx_peak. "
            f"Corregido a idx_peak + {margen_post_peak}."
        )

    print(f"  [{nombre_carpeta}]  "
          f"peak={tiempos[idx_peak]:.2f}s  "
          f"cuasi={tiempos[idx_quasi]:.2f}s  "
          f"frames_cuasi={n - idx_quasi}  "
          f"fallback={fallback_usado}")

    return {
        'tiempos'  : tiempos,
        'v_smooth' : v_smooth,
        'idx_peak' : idx_peak,
        'idx_quasi': idx_quasi,
        't_peak'   : float(tiempos[idx_peak]),
        't_quasi'  : float(tiempos[idx_quasi]),
        'fallback' : fallback_usado,
        'etapas'   : {
            'inicio'    : (0, idx_peak),
            'transicion': (idx_peak, idx_quasi),
            'cuasi'     : (idx_quasi, n_total),
        }
    }

# ==========================================
# FIGURA DE DIAGNÓSTICO
# ==========================================

def graficar_etapas(resultado, tiempos_raw=None, v_raw=None,
                    titulo="", output_path=None):
    """
    Genera figura de diagnóstico con las tres etapas marcadas.

    Parámetros
    ----------
    resultado    : dict — salida de detectar_etapas()
    tiempos_raw  : array — timestamps reales [s] para la curva cruda (opcional)
    v_raw        : list  — velocidad cruda por frame (opcional)
    titulo       : str
    output_path  : str  — si se pasa, guarda la figura en disco
    """
    t        = resultado['tiempos']
    v_smooth = resultado['v_smooth']
    ip       = resultado['idx_peak']
    iq       = resultado['idx_quasi']

    fig, ax = plt.subplots(figsize=(12, 5))

    if tiempos_raw is not None and v_raw is not None:
        ax.plot(tiempos_raw, v_raw, color='steelblue', lw=0.8,
                alpha=0.35, label='V media cruda', zorder=2)

    ax.plot(t, v_smooth, color='steelblue', lw=2,
            label='V media suavizada', zorder=3)

    ax.axvspan(t[0],  t[ip], alpha=0.10, color='royalblue',   label='Inicio')
    ax.axvspan(t[ip], t[iq], alpha=0.10, color='darkorange',  label='Transición')
    ax.axvspan(t[iq], t[-1], alpha=0.10, color='forestgreen', label='Cuasi-estacionario')

    ax.axvline(t[ip], color='royalblue',   lw=1.5, linestyle='--',
               label=f'Peak  t={t[ip]:.2f}s')
    ax.axvline(t[iq], color='forestgreen', lw=1.5, linestyle='--',
               label=f'Cuasi t={t[iq]:.2f}s')

    if resultado['fallback']:
        ax.axvline(t[iq], color='red', lw=2.5, linestyle=':',
                   label='⚠ Fallback — revisar manualmente')

    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('V media (mm/s)')
    # Título omitido: descripción en el caption de la memoria (figura tipo).
    # El identificador de corrida/zona (parámetro `titulo`) migra al caption.
    # ax.set_title(titulo or "Detección de etapas")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.close(fig)