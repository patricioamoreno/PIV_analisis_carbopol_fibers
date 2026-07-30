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
EPS_CUASI         = 0.02  # s⁻¹ — umbral de tasa normalizada Γ(t) = |dV/dt| / V(t_peak)
VF_CONFIRM        = 20    # frames consecutivos requeridos bajo EPS_CUASI

# Máximo hueco interno (en frames) que se rellena por interpolación lineal
# antes de suavizar. Huecos mayores conservan sus NaN.
#   POR QUÉ: en una zona que se llena tarde, los frames con menos de
#   MIN_PTS_FRAME vectores quedan NaN. Rellenarlos sin límite produce una
#   rampa lineal de pendiente pequeña y CONSTANTE, que es exactamente la
#   firma que busca el criterio V3 -> se declararía cuasi-estacionario
#   dentro de un vacío de datos. Mismo criterio que MAX_HUECO_INTERP en
#   construir_caches.py.
MAX_HUECO_INTERP = 5

# Cómo se normaliza Γ(t).
#   "t_peak"     -> Γ(t) = |dV/dt| / Ṽ(t_peak)     [definición de la memoria]
#   "max_global" -> Γ(t) = |dV/dt| / max(Ṽ)        [comportamiento HISTÓRICO]
# Se conserva "max_global" SOLO para poder reproducir resultados antiguos.
# No son equivalentes: t_peak es el ÚLTIMO pico local sobre el 50 % del
# máximo, no el máximo. Sobre este dataset la razón entre ambos tiene
# mediana 1,70 y llega a 5,4, de modo que "max_global" vuelve el umbral
# efectivo entre 1,7 y 5,4 veces más permisivo, y VARIABLE entre corridas.
# Además el máximo global puede caer dentro de los T_IGNORAR segundos que
# el propio método declara descartar.
NORMALIZACION_GAMMA = "t_peak"

# Ventana de suavizado usada SOLO para evaluar Γ(t) (la derivada).
#   None -> usa ventana_suavizado (comportamiento histórico).
#   int  -> ventana fija en frames.
#   "auto" -> FACTOR_VENTANA_GAMMA * idx_peak, acotado a [W_GAMMA_MIN, W_GAMMA_MAX].
#
# POR QUÉ DOS VENTANAS. La detección del peak y la evaluación de Γ tienen
# requisitos opuestos: el peak necesita POCO suavizado (si no, se desplaza
# y se aplana), mientras que Γ es una derivada numérica y necesita MUCHO
# (el ruido de alta frecuencia domina |dV/dt| aunque sea invisible en V).
# Usar una sola ventana obliga a sacrificar uno de los dos.
#
# Además, la ventana correcta para Γ no es la misma en frames para ambas
# reologías: Car-0,5 % escurre ~20x más lento, así que 15 frames (0,45 s)
# suavizan mucho MENOS, en términos de la escala del propio evento, que
# los 50 frames (1,50 s) de Car-0,2 %. La regla "auto" fija la ventana en
# unidades de la escala temporal del evento (idx_peak), eliminando el
# número mágico por mezcla.
VENTANA_GAMMA        = "auto"
FACTOR_VENTANA_GAMMA = 0.75
W_GAMMA_MIN          = 25
W_GAMMA_MAX          = 250

# ==========================================
# UTILIDADES
# ==========================================

def natural_sort_key(s):
    """Clave de ordenamiento natural (e.g. frame_9 < frame_10)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]

# ==========================================
# DIAGNÓSTICO DE CONVERGENCIA
# ==========================================

def eps_minimo_convergencia(tasa_norm, inicio, n, vf_confirm):
    """
    Menor umbral ε para el que EXISTIRÍA una ventana de vf_confirm frames
    consecutivos con Γ(t) < ε, buscando desde 'inicio'.

    Sirve para reportar de forma honesta los casos que no convergen: en vez
    de decir "no se detectó", permite decir "no existe ningún tramo estable
    bajo 0,02 s⁻¹; el mínimo alcanzable es 0,047 s⁻¹". Es el máximo de Γ
    dentro de la ventana, minimizado sobre todas las ventanas posibles.

    Devuelve np.nan si no hay ninguna ventana completa disponible.
    """
    mejor = np.inf
    for i in range(inicio, n - vf_confirm):
        w = tasa_norm[i : i + vf_confirm]
        if not np.all(np.isfinite(w)):
            continue
        mejor = min(mejor, float(np.max(w)))
    return mejor if np.isfinite(mejor) else np.nan


# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def detectar_etapas(tiempos, v_medias,
                    ventana_suavizado=VENTANA_SUAVIZADO,
                    t_ignorar=T_IGNORAR,
                    margen_post_peak=MARGEN_POST_PEAK,
                    eps_cuasi=EPS_CUASI,
                    vf_confirm=VF_CONFIRM,
                    max_hueco_interp=MAX_HUECO_INTERP,
                    normalizacion=NORMALIZACION_GAMMA,
                    ventana_gamma=VENTANA_GAMMA,
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
            'convergio': False,
            'motivo': 'sin_datos',
            'eps_minimo': np.nan,
            'v_norm': np.nan,
            'etapas': {'inicio': (0, 0),
                       'transicion': (0, min(1, n_total - 1)),
                       'cuasi': (min(1, n_total - 1), n_total)},
        }
    ult_valido = int(finitos[-1])           # último frame con material
    tiempos_d  = tiempos[:ult_valido + 1]   # tramo de detección
    v_medias   = v_arr[:ult_valido + 1]

    # Relleno ACOTADO de huecos internos (ver MAX_HUECO_INTERP).
    # limit_area="inside" impide extrapolar antes del primer dato válido o
    # después del último; limit acota la longitud del hueco rellenable.
    v_series = pd.Series(v_medias).interpolate(
        limit=max_hueco_interp, limit_area="inside")

    # ── Paso 1: Suavizado ─────────────────────────────────────────────────
    # min_periods=1: en los bordes la media móvil usa la ventana PARCIAL
    # disponible en vez de dejar NaN.
    #   POR QUÉ: antes se hacía .fillna(v_series), es decir, los primeros y
    #   últimos ~w/2 frames quedaban CRUDOS (sin suavizar). Con w=50 a
    #   33,3 fps eso son 0,72 s, y la búsqueda del peak arranca en 0,50 s:
    #   la detección operaba dentro del tramo sin suavizar, y un pico de
    #   ruido en esa región podía fijar el máximo de la serie.
    v_smooth = (v_series
                .rolling(window=ventana_suavizado, center=True, min_periods=1)
                .mean()
                .values)

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
    #
    #   Γ(t_i) = |Ṽ(t_i) - Ṽ(t_{i-1})| / [ (t_i - t_{i-1}) · Ṽ(t_peak) ]
    #
    # DIFERENCIA EN DIFERENCIAS: se usa la diferencia HACIA ATRÁS, de modo
    # que Γ[i] queda asociado al instante t_i, tal como en la ecuación.
    # (Antes se usaba np.diff, que es hacia adelante: Γ[i] correspondía en
    # realidad a t_{i+1}, un desfase de un frame respecto a la definición.)
    # Serie propia para la derivada (ver VENTANA_GAMMA).
    if ventana_gamma is None:
        w_g = ventana_suavizado
    elif ventana_gamma == "auto":
        w_g = int(np.clip(round(FACTOR_VENTANA_GAMMA * max(idx_peak, 1)),
                          W_GAMMA_MIN, W_GAMMA_MAX))
    else:
        w_g = int(ventana_gamma)
    w_g = max(1, min(w_g, n))

    v_gamma = (v_series
               .rolling(window=w_g, center=True, min_periods=1)
               .mean()
               .values)

    dt        = np.diff(tiempos)
    dv_dt     = np.empty(n, dtype=float)
    dv_dt[1:] = np.diff(v_gamma) / dt
    dv_dt[0]  = dv_dt[1] if n > 1 else 0.0

    # NORMALIZACIÓN: el denominador es Ṽ(t_peak), NO el máximo global.
    # Ver la nota de NORMALIZACION_GAMMA arriba: no son la misma cantidad.
    if normalizacion == "t_peak":
        v_norm = float(v_smooth[idx_peak])
    elif normalizacion == "max_global":
        v_norm = float(np.nanmax(v_smooth))
    else:
        raise ValueError(f"normalizacion desconocida: {normalizacion!r}")

    if not np.isfinite(v_norm) or v_norm <= 0:
        v_norm = float(np.nanmax(v_smooth))

    tasa_norm = np.abs(dv_dt) / v_norm

    idx_quasi       = None
    inicio_busqueda = idx_peak + margen_post_peak
    convergio       = False
    motivo          = "ok"

    for i in range(inicio_busqueda, n - vf_confirm):
        if np.all(tasa_norm[i : i + vf_confirm] < eps_cuasi):
            idx_quasi = i
            convergio = True
            break

    # ── NO CONVERGENCIA: antes esto pasaba EN SILENCIO ────────────────────
    # El código original hacía  idx_quasi = inicio_busqueda  y dejaba
    # fallback=False, sin imprimir nada. El resultado era que t_quasi valía
    # exactamente t_peak + margen_post_peak (1,00 s a 33,3 fps) sin que
    # nada lo delatara. Sobre este dataset ocurría en 15 de 58 series de
    # polilínea, todas Car-0,5 % en viga175/viga250, y esos valores
    # llegaron a las tablas de la memoria como si fueran mediciones.
    eps_min = np.nan
    if idx_quasi is None:
        motivo    = "no_converge"
        eps_min   = eps_minimo_convergencia(tasa_norm, inicio_busqueda,
                                            n, vf_confirm)
        idx_quasi = inicio_busqueda
        label     = nombre_carpeta or "carpeta desconocida"
        print(
            f"\n⛔ NO CONVERGE [{label}]: ningún tramo de {vf_confirm} frames "
            f"consecutivos cumple Γ(t) < {eps_cuasi:g} s⁻¹.\n"
            f"   → t_quasi NO es una medición: quedó fijado en "
            f"t_peak + {margen_post_peak} frames.\n"
            f"   → ε mínimo que SÍ convergería: "
            f"{eps_min:.4f} s⁻¹ ({eps_min/eps_cuasi:.1f}× el nominal).\n"
            f"   → Revisar antes de usar este valor en cualquier tabla."
        )

    # ── Fallback si quedan muy pocos frames de cuasi ──────────────────────
    frames_restantes = n - idx_quasi
    if frames_restantes < vf_confirm:
        idx_quasi_original = idx_quasi
        idx_quasi          = min(int(n * FALLBACK_PCT), n - 1)
        fallback_usado     = True
        convergio          = False
        motivo             = "cuasi_muy_tarde"
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
        convergio = False
        motivo    = "cuasi_antes_del_peak"
        print(
            f"⚠️  ADVERTENCIA [{nombre_carpeta}]: idx_quasi <= idx_peak. "
            f"Corregido a idx_peak + {margen_post_peak}."
        )

    print(f"  [{nombre_carpeta}]  "
          f"peak={tiempos[idx_peak]:.2f}s  "
          f"cuasi={tiempos[idx_quasi]:.2f}s  "
          f"frames_cuasi={n - idx_quasi}  "
          f"w_gamma={w_g}  "
          f"convergio={convergio}  "
          f"fallback={fallback_usado}")

    return {
        'tiempos'  : tiempos,
        'v_smooth' : v_smooth,
        'idx_peak' : idx_peak,
        'idx_quasi': idx_quasi,
        't_peak'   : float(tiempos[idx_peak]),
        't_quasi'  : float(tiempos[idx_quasi]),
        'fallback' : fallback_usado,
        # convergio=False  =>  t_quasi NO es una medición. No usar en tablas
        # sin declararlo. 'motivo' dice por qué; 'eps_minimo' es el umbral
        # que sí habría convergido; 'v_norm' es el denominador de Γ usado.
        'convergio': bool(convergio),
        'motivo'   : motivo,
        'eps_minimo': float(eps_min) if eps_min == eps_min else np.nan,
        'v_norm'   : float(v_norm),
        'w_gamma'  : int(w_g),
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

    ax.axvline(t[ip], color='royalblue',   lw=1.5, linestyle='--')
    ax.axvline(t[iq], color='forestgreen', lw=1.5, linestyle='--')

    if resultado['fallback']:
        ax.axvline(t[iq], color='red', lw=2.5, linestyle=':',
                   label='⚠ Fallback — revisar manualmente')

    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('V media (mm/s)')
    # Título omitido: descripción en el caption de la memoria (figura tipo).
    # El identificador de corrida/zona (parámetro `titulo`) migra al caption.
    # ax.set_title(titulo or "Detección de etapas")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=round(t[-1]))
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.close(fig)