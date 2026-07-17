"""
calcular_etapas_zonas.py
========================
Detecta etapas (inicio / transicion / cuasi) POR ZONA (Z1, Z2, Z3, ...)
usando la velocidad media de cada zona en función del tiempo, leída desde
los cachés .npz generados por construir_caches_zonas.py.

Es el análogo de calcular_etapas_polilinea.py, pero:
  - En vez de una matriz [frames × puntos_linea], lee el caché de zona
    (puntos dispersos con frame_idx) y construye v_media(t) agrupando por
    frame_idx dentro de cada zona.
  - Reutiliza TAL CUAL detectar_etapas() / graficar_etapas() de
    detectar_etapas.py (mismo criterio V3 de pendiente normalizada).
  - Descubre automáticamente qué zonas hay en cada caché, de modo que si
    cambias el número/geometría de zonas en definir_zonas.py y reconstruyes
    los cachés, este script recalcula para las zonas que existan.

Genera etapas_zonas.json con claves del tipo:
  "{carpeta}_Z1", "{carpeta}_Z2", "{carpeta}_Z3", ...

Uso:
  python calcular_etapas_zonas.py
"""

import os
import re
import json
import glob
import numpy as np

from detectar_etapas import detectar_etapas, graficar_etapas, natural_sort_key
from construir_caches_zonas import cargar_cache_zonas as _cargar_cache_zonas_disco
from construir_caches_zonas import _nombre_cache

try:
    from win10toast import ToastNotifier
    USAR_NOTIFICACION = True
except ImportError:
    USAR_NOTIFICACION = False
    print("⚠ win10toast no está instalado. Las notificaciones están desactivadas.")

# ============================================================
# CONFIGURACIÓN
# ============================================================

CACHE_DIR   = "cache_zonas"
OUTPUT_JSON = "etapas_zonas.json"
OUTPUT_FIGS = "Etapas_zonas"
GUARDAR_FIGURAS = True

# True  → recalcula todas las etapas y reescribe el JSON desde cero.
# False → conserva las entradas ya presentes en OUTPUT_JSON y solo agrega
#         las que falten (útil al añadir cachés nuevos sin recomputar todo).
RECALCULO = True

# Zonas a procesar.
#   None  → auto-detecta las zonas presentes en cada caché (recomendado:
#           si cambias la cantidad de zonas, se ajusta solo).
#   lista → fuerza un subconjunto, p.ej. ["Z1", "Z2"].
ZONAS = None

ETAPAS_ORDEN = ["inicio", "transicion", "cuasi"]

# Tomas a excluir del cálculo de la base (promedio por reología).
TOMAS_EXCLUIDAS = set()

# Overrides manuales: { "carpeta_Zk": (idx_peak, idx_quasi) }
OVERRIDES_MANUALES = {}

# Mínimo de puntos en un frame para que su v_media cuente (evita ruido).
MIN_PTS_FRAME = 3


# ============================================================
# UTILIDADES
# ============================================================

# Caché en memoria (solo dentro de esta corrida del script). Sin esto, cada
# .npz de zona se lee y descomprime del disco repetidas veces en la misma
# ejecución:
#   1. una vez por carpeta durante la auto-detección de zonas (ZONAS=None),
#   2. una vez por carpeta base (n-0000) DENTRO de calcular_tiempos_base(),
#      para CADA una de las zonas que se procesan (hasta 8),
#   3. una vez por carpeta DENTRO del bucle de aplicación, de nuevo por
#      cada zona (hasta 8) -- las carpetas base están incluidas aquí también.
# Para una toma base con las 8 zonas presentes, el mismo archivo se leía
# hasta 17 veces por corrida completa; para una toma no-base, hasta 9. El
# contenido del caché (arrays de puntos PIV) no se muta en ningún punto de
# este script -- solo se indexa y agrega -- así que reutilizar el mismo
# objeto en memoria es seguro.
_CACHE_MEM = {}

def cargar_cache_zonas(carpeta, cache_dir=CACHE_DIR):
    key = (carpeta, cache_dir)
    if key not in _CACHE_MEM:
        _CACHE_MEM[key] = _cargar_cache_zonas_disco(carpeta, cache_dir)
    return _CACHE_MEM[key]


def carpetas_disponibles(cache_dir):
    """Lista las carpetas que tienen caché de zonas (*_zonas.npz)."""
    archivos = glob.glob(os.path.join(cache_dir, "*_zonas.npz"))
    carpetas = [os.path.basename(a).replace("_zonas.npz", "") for a in archivos]
    return sorted(carpetas, key=natural_sort_key)


def zonas_en_cache(cache):
    """Zonas presentes en un caché, en orden natural, excluyendo 'fuera'."""
    zs = [z for z in np.unique(cache['zona']) if z != "fuera"]
    return sorted(zs, key=natural_sort_key)


def serie_temporal_zona(cache, zona):
    """
    Construye v_media(t) de una zona a partir del caché de puntos dispersos.

    Agrupa los puntos por frame_idx (solo los de la zona pedida) y promedia
    v_mag por frame. Frames de la corrida sin puntos en la zona quedan como
    NaN, preservando el eje temporal completo (igual criterio que las
    polilíneas: la cola NaN se maneja dentro de detectar_etapas()).

    Retorna
    -------
    t_full   : array (n_frames,)  timestamps normalizados a t0=0
    v_media  : array (n_frames,)  velocidad media de la zona por frame (NaN si vacío)
    """
    n_frames = cache['n_frames']
    fi   = cache['frame_idx']
    zsel = cache['zona'] == zona

    # timestamp por frame (tomamos el primer t visto en cada frame_idx)
    t_full = np.full(n_frames, np.nan, dtype=np.float64)
    t_all  = cache['t']
    # asignación vectorizada: para cada punto, fija t de su frame
    # (si un frame no tiene puntos en NINGUNA zona, su t queda NaN y se
    #  interpola más abajo)
    orden = np.argsort(fi)
    fi_o, t_o = fi[orden], t_all[orden]
    primeros = np.unique(fi_o, return_index=True)[1]
    t_full[fi_o[primeros]] = t_o[primeros]

    # v_media por frame en la zona
    v_media = np.full(n_frames, np.nan, dtype=np.float64)
    vmag = cache['v_mag']
    fi_z = fi[zsel]
    vm_z = vmag[zsel]
    if len(fi_z) > 0:
        # suma y conteo por frame
        suma   = np.zeros(n_frames)
        conteo = np.zeros(n_frames)
        np.add.at(suma,   fi_z, vm_z)
        np.add.at(conteo, fi_z, 1.0)
        con_datos = conteo >= MIN_PTS_FRAME
        v_media[con_datos] = suma[con_datos] / conteo[con_datos]

    # Rellenar timestamps faltantes por interpolación de índice (eje regular)
    nan_t = np.isnan(t_full)
    if nan_t.any() and not nan_t.all():
        idx = np.arange(n_frames)
        t_full[nan_t] = np.interp(idx[nan_t], idx[~nan_t], t_full[~nan_t])
    elif nan_t.all():
        t_full = np.arange(n_frames, dtype=np.float64)

    # Normalizar a t0 = 0 (igual que en polilíneas)
    t_full = t_full - t_full[0]
    return t_full, v_media


# ============================================================
# CÁLCULO DE BASE POR REOLOGÍA  (igual filosofía que polilíneas)
# ============================================================

def calcular_tiempos_base(carpetas, zona, cache_dir):
    """
    Promedia t_peak y t_quasi sobre las tomas base (n-0000) de una reología
    para una zona dada. Retorna (t_peak_base, t_quasi_base) o (None, None).
    """
    t_peaks, t_quasis = [], []

    for carpeta in carpetas:
        if any(m in carpeta for m in TOMAS_EXCLUIDAS):
            continue
        cache = cargar_cache_zonas(carpeta, cache_dir)
        if cache is None:
            continue
        if zona not in zonas_en_cache(cache):
            continue

        t_full, v_media = serie_temporal_zona(cache, zona)
        nombre = f"{carpeta}_{zona}"

        if nombre in OVERRIDES_MANUALES:
            ip, iq = OVERRIDES_MANUALES[nombre]
            n = len(t_full)
            ip = min(ip, n - 2); iq = min(iq, n - 1)
            t_peaks.append(float(t_full[ip])); t_quasis.append(float(t_full[iq]))
            continue

        vent_suav = 15 if "car-05" in carpeta else 50
        res = detectar_etapas(t_full, v_media,
                              nombre_carpeta=nombre,
                              ventana_suavizado=vent_suav)
        if not res["fallback"]:
            t_peaks.append(res["t_peak"])
            t_quasis.append(res["t_quasi"])
        else:
            print(f"    ⚠ Fallback en base, ignorada: {nombre}")

    if not t_peaks:
        return None, None
    return float(np.mean(t_peaks)), float(np.mean(t_quasis))


# ============================================================
# MAIN
# ============================================================

def main():
    todas_carpetas = carpetas_disponibles(CACHE_DIR)
    if not todas_carpetas:
        print(f"⚠ No hay cachés *_zonas.npz en {CACHE_DIR}. "
              f"Corre primero construir_caches_zonas.py")
        return

    print(f"Carpetas con caché: {len(todas_carpetas)}\n")

    # Descubrir el universo de zonas presentes (si ZONAS es None)
    if ZONAS is None:
        zonas_set = set()
        for carpeta in todas_carpetas:
            cache = cargar_cache_zonas(carpeta, CACHE_DIR)
            if cache is not None:
                zonas_set.update(zonas_en_cache(cache))
        zonas_a_procesar = sorted(zonas_set, key=natural_sort_key)
    else:
        zonas_a_procesar = ZONAS
    print(f"Zonas a procesar: {zonas_a_procesar}\n")

    # Cargar JSON previo si no se recalcula todo
    resultados = {}
    if not RECALCULO and os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            resultados = json.load(f)
        print(f"Conservando {len(resultados)} entradas previas (RECALCULO=False)\n")

    # Reologías presentes en los nombres de carpeta
    reos = sorted({m.group(1) for c in todas_carpetas
                   if (m := re.search(r'car-(\d+)', c))})

    for zona in zonas_a_procesar:
        print(f"\n{'='*55}\n  ZONA: {zona}\n{'='*55}")

        for reo in reos:
            carpetas_reo  = [c for c in todas_carpetas if f"car-{reo}" in c]
            carpetas_base = [c for c in carpetas_reo if "n-0000" in c]

            print(f"\n  Base car-{reo} ({len(carpetas_base)} tomas)...")
            t_peak_base, t_quasi_base = calcular_tiempos_base(
                carpetas_base, zona, CACHE_DIR)
            usar_base = t_peak_base is not None

            for carpeta in carpetas_reo:
                nombre = f"{carpeta}_{zona}"

                if not RECALCULO and nombre in resultados:
                    continue
                if any(m in carpeta for m in TOMAS_EXCLUIDAS):
                    continue

                cache = cargar_cache_zonas(carpeta, CACHE_DIR)
                if cache is None or zona not in zonas_en_cache(cache):
                    continue

                t_full, v_media = serie_temporal_zona(cache, zona)
                n = len(t_full)

                # ── Selección de índices ───────────────────────
                if nombre in OVERRIDES_MANUALES:
                    idx_peak, idx_quasi = OVERRIDES_MANUALES[nombre]
                    idx_peak  = min(idx_peak,  n - 2)
                    idx_quasi = min(idx_quasi, n - 1)
                    t_peak_used  = float(t_full[idx_peak])
                    t_quasi_used = float(t_full[idx_quasi])
                    modo = "override"

                elif usar_base:
                    # índice más cercano al tiempo base de esta toma
                    idx_peak  = int(np.argmin(np.abs(t_full - t_peak_base)))
                    idx_quasi = int(np.argmin(np.abs(t_full - t_quasi_base)))
                    idx_peak  = min(max(idx_peak, 0), n - 2)
                    idx_quasi = min(max(idx_quasi, idx_peak + 1), n - 1)
                    t_peak_used, t_quasi_used = t_peak_base, t_quasi_base
                    modo = "base"

                else:
                    # sin base válida: detección individual
                    vent_suav = 15 if "car-05" in carpeta else 50
                    res = detectar_etapas(t_full, v_media,
                                          nombre_carpeta=nombre,
                                          ventana_suavizado=vent_suav)
                    idx_peak, idx_quasi = res["idx_peak"], res["idx_quasi"]
                    t_peak_used, t_quasi_used = res["t_peak"], res["t_quasi"]
                    modo = "individual"

                entrada = {
                    'idx_peak' : int(idx_peak),
                    'idx_quasi': int(idx_quasi),
                    't_peak'   : float(t_peak_used),
                    't_quasi'  : float(t_quasi_used),
                    'n_frames' : int(n),
                    'modo'     : modo,
                    'fallback' : False,
                    'etapas'   : {
                        'inicio'    : (0, int(idx_peak)),
                        'transicion': (int(idx_peak), int(idx_quasi)),
                        'cuasi'     : (int(idx_quasi), int(n)),
                    }
                }
                resultados[nombre] = entrada
                print(f"  {nombre}  peak={t_full[idx_peak]:.2f}s  "
                      f"cuasi={t_full[idx_quasi]:.2f}s  [{modo}]")

                if GUARDAR_FIGURAS:
                    res_fig = {
                        'tiempos': t_full, 'v_smooth': v_media,
                        'idx_peak': idx_peak, 'idx_quasi': idx_quasi,
                        'fallback': False,
                    }
                    fig_dir  = os.path.join(OUTPUT_FIGS, zona)
                    fig_path = os.path.join(fig_dir, f"etapas_{nombre}.png")
                    graficar_etapas(res_fig, tiempos_raw=t_full, v_raw=v_media,
                                    titulo=f"Etapas — {nombre}",
                                    output_path=fig_path)

    # ── Guardar JSON ────────────────────────────────────────
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n✅ Guardado: {OUTPUT_JSON}  ({len(resultados)} entradas)")

    if USAR_NOTIFICACION:
        try:
            toaster = ToastNotifier()
            toaster.show_toast("VSCode", "¡Tu código de Python terminó exitosamente!", duration=5, threaded=False)
        except Exception as e:
            print(f"⚠ No se pudo mostrar la notificación de Windows: {e}")


if __name__ == "__main__":
    main()