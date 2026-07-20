"""
calcular_etapas_polilinea.py
============================
Detecta etapas (inicio / transicion / cuasi) POR ZONA (L, viga175, viga250)
usando la velocidad media a lo largo de cada polilínea, leída desde los
cachés .npz generados por espectograma.py / espectograma_viga.py.

Genera etapas_polilinea.json con claves del tipo:
  "{carpeta}_L"
  "{carpeta}_viga175"
  "{carpeta}_viga250"

Uso:
  python calcular_etapas_polilinea.py
"""

import os
import re
import json
import glob
import warnings
import numpy as np
from detectar_etapas import detectar_etapas, graficar_etapas, natural_sort_key

try:
    from win10toast import ToastNotifier
    USAR_NOTIFICACION = True
except ImportError:
    USAR_NOTIFICACION = False
    print("⚠ win10toast no está instalado. Las notificaciones están desactivadas.")

# ============================================================
# CONFIGURACIÓN
# ============================================================

CACHE_DIR = "cache_completo"
OUTPUT_JSON = "etapas_polilinea.json"
OUTPUT_FIGS = "Semana8/Etapas_polilinea"
GUARDAR_FIGURAS = True

# Zonas: (prefijo_cache, zona_key)
# prefijo="" → L (formato "{carpeta}_{etapa}.npz")
# prefijo="viga175" → viga175 (formato "viga175_{carpeta}_{etapa}.npz")
ZONAS = [
    ("",        "L"),
    ("viga175", "viga175"),
    ("viga250", "viga250"),
]


ETAPAS_ORDEN = ["inicio", "transicion", "cuasi"]

TOMAS_EXCLUIDAS_VIGA = {'m93-toma-1-n-0000-car-05-piv'}

# Índices manuales para casos problemáticos (override del promedio base)
# formato: { "carpeta_zonakey": (idx_peak, idx_quasi) }
OVERRIDES_MANUALES = {
}

# ============================================================
# UTILIDADES
# ============================================================

# Caché en memoria (solo dentro de esta corrida del script). El flujo del
# programa lee cada .npz por lo menos dos veces con el patrón actual: una
# vez dentro de calcular_indices_base() para las tomas n-0000, y otra vez
# en el bucle de aplicación, que vuelve a recorrer TODAS las carpetas de la
# reología (las tomas base incluidas, porque carpetas_base ⊆ carpetas_reo).
# Sin este caché, cada toma base se lee y descomprime del disco 2 veces por
# cada una de las 3 zonas procesadas (L, viga175, viga250) = 6 lecturas
# redundantes por toma base en una corrida completa.
_CACHE_MEM = {}

def cargar_cache_completo(carpeta, prefijo, cache_dir):
    fname = f"{prefijo}_{carpeta}_completo.npz" if prefijo else f"{carpeta}_completo.npz"
    path  = os.path.join(cache_dir, fname)
    if path in _CACHE_MEM:
        return _CACHE_MEM[path]
    if not os.path.exists(path):
        _CACHE_MEM[path] = (None, None)
        return None, None
    data = np.load(path)
    resultado = (data['matriz'], data['tiempos'])
    _CACHE_MEM[path] = resultado
    return resultado


def carpetas_disponibles(cache_dir):
    archivos = glob.glob(os.path.join(cache_dir, "*_completo.npz"))
    carpetas = set()
    for a in archivos:
        nombre = os.path.basename(a)
        if nombre.startswith("viga"):
            continue
        carpeta = nombre.replace("_completo.npz", "")
        carpetas.add(carpeta)
    return sorted(carpetas, key=natural_sort_key)


def calcular_indices_base(carpetas, prefijo, zona_key, cache_dir):
    resultados_peak  = []
    resultados_quasi = []
    t_peaks_lista    = []
    t_quasis_lista   = []

    for carpeta in carpetas:
        if prefijo != "" and any(m in carpeta for m in TOMAS_EXCLUIDAS_VIGA):
            print(f"    ⚠ Excluida de base: {carpeta}")
            continue

        nombre = f"{carpeta}_{zona_key}"

        mat_full, t_full = cargar_cache_completo(carpeta, prefijo, cache_dir)
        if mat_full is None:
            continue

        idx_sort = np.argsort(t_full)
        t_full   = t_full[idx_sort]
        mat_full = mat_full[idx_sort]
        t_full   = t_full - t_full[0]
        # Antes de que el frente de avance llegue a esta polilinea, ningun
        # punto del corte tiene vector PIV valido: la fila completa de
        # mat_full es NaN para ese frame. np.nanmean() avisa "Mean of empty
        # slice" -- correctamente devuelve NaN, no hay dato que promediar.
        # Se silencia solo ese mensaje puntual, sin alterar el resultado.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            v_media = np.nanmean(mat_full, axis=1)
        n        = len(t_full)

        # Si tiene override manual, usarlo directamente
        if nombre in OVERRIDES_MANUALES:
            ip, iq = OVERRIDES_MANUALES[nombre]
            ip = min(ip, n - 2)
            iq = min(iq, n - 1)
            resultados_peak.append(ip)
            resultados_quasi.append(iq)
            t_peaks_lista.append(float(t_full[ip]))
            t_quasis_lista.append(float(t_full[iq]))
            print(f"    [override] {nombre}: peak={ip}  cuasi={iq}")
            continue

        # Si no, detectar normalmente
        vent_suav = 15 if "car-05" in carpeta else 50
        res = detectar_etapas(t_full, v_media,
                              nombre_carpeta=nombre,
                              ventana_suavizado=vent_suav)

        if not res["fallback"]:
            resultados_peak.append(res["idx_peak"])
            resultados_quasi.append(res["idx_quasi"])
            t_peaks_lista.append(res["t_peak"])
            t_quasis_lista.append(res["t_quasi"])
        else:
            print(f"    ⚠ Fallback en base, ignorada: {nombre}")

    if not resultados_peak:
        return None, None

    idx_peak_base  = None   # ya no se usa
    idx_quasi_base = None   # ya no se usa
    t_peak_base    = float(np.mean(t_peaks_lista))
    t_quasi_base   = float(np.mean(t_quasis_lista))
    return t_peak_base, t_quasi_base  # ← solo 2 valores ahora

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_FIGS, exist_ok=True)
    dir_L       = os.path.join(OUTPUT_FIGS, "L")
    dir_viga175 = os.path.join(OUTPUT_FIGS, "viga175")
    dir_viga250 = os.path.join(OUTPUT_FIGS, "viga250")
    os.makedirs(dir_L,       exist_ok=True)
    os.makedirs(dir_viga175, exist_ok=True)
    os.makedirs(dir_viga250, exist_ok=True)

    todas_carpetas = carpetas_disponibles(CACHE_DIR)
    print(f"Carpetas encontradas: {len(todas_carpetas)}\n")

    resultados = {}
    fallbacks  = []

    for prefijo, zona_key in ZONAS:
        print(f"\n{'='*55}")
        print(f"  ZONA: {zona_key}")
        print(f"{'='*55}")

        for reo in ['02', '05']:
            # Carpetas base (n-0000) de esta reo
            carpetas_base = [c for c in todas_carpetas
                             if f"car-{reo}" in c and "n-0000" in c]
            print(f"\n  Calculando base car-{reo} ({len(carpetas_base)} tomas)...")
            t_peak_base, t_quasi_base = calcular_indices_base(
                carpetas_base, prefijo, zona_key, CACHE_DIR)

            if t_peak_base is None:
                # print(f"  ⚠ Sin base válida para car-{reo} / {zona_key}, usando detección individual")
                usar_base = False
            else:
                usar_base = True

            # Aplicar a TODAS las carpetas de esta reo
            carpetas_reo = [c for c in todas_carpetas if f"car-{reo}" in c]
            for carpeta in carpetas_reo:
                mat_full, t_full = cargar_cache_completo(carpeta, prefijo, CACHE_DIR)
                if mat_full is None:
                    print(f"  ⚠ Sin caché: {carpeta} / {zona_key}")
                    continue

                idx_sort = np.argsort(t_full)
                mat_full = mat_full[idx_sort]
                t_full   = t_full[idx_sort]
                t_full   = t_full - t_full[0]        # ← normalizar a t=0
                # Mismo caso benigno que en la recoleccion de base: frames
                # anteriores a la llegada del frente de avance no tienen
                # ningun vector PIV valido a lo largo de la polilinea.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Mean of empty slice")
                    v_media = np.nanmean(mat_full, axis=1)
                n        = len(t_full)
                nombre   = f"{carpeta}_{zona_key}"
                # Excluir m93 completamente en zonas de viga
                if prefijo != "" and any(m in carpeta for m in TOMAS_EXCLUIDAS_VIGA):
                    print(f"  ⚠ Excluida de resultados: {nombre}")
                    continue

                # Override manual — tiene prioridad absoluta
                if nombre in OVERRIDES_MANUALES:
                    idx_peak, idx_quasi = OVERRIDES_MANUALES[nombre]
                    idx_peak  = min(idx_peak,  n - 2)
                    idx_quasi = min(idx_quasi, n - 1)
                    print(f"  {nombre}  → override manual: peak={idx_peak}  cuasi={idx_quasi}")

                elif not usar_base:
                    print(f"  ⚠ Saltando {nombre}: sin índices base para car-{reo}/{zona_key}")
                    continue

                else:
                    # Clampear por si la toma es más corta que la base
                    # En vez de usar idx_peak_base e idx_quasi_base directamente,
# buscar el índice más cercano al tiempo base en t_full de esta toma
                    idx_peak  = int(np.argmin(np.abs(t_full - t_peak_base)))
                    idx_quasi = int(np.argmin(np.abs(t_full - t_quasi_base)))

                    # Validaciones
                    idx_peak  = min(max(idx_peak,  0), n - 2)
                    idx_quasi = min(max(idx_quasi, idx_peak + 1), n - 1)
                    if idx_quasi <= idx_peak:
                        idx_quasi = idx_peak + 1

                res = {
                    'tiempos'  : t_full,
                    'v_smooth' : v_media,
                    'idx_peak' : idx_peak,
                    'idx_quasi': idx_quasi,
                    't_peak'   : t_peak_base,    # ← fijo para todas
                    't_quasi'  : t_quasi_base,   # ← fijo para todas
                    'fallback' : False,
                    'etapas'   : {
                        'inicio'    : (0, idx_peak),
                        'transicion': (idx_peak, idx_quasi),
                        'cuasi'     : (idx_quasi, n),
                    }
                }
                print(f"  {nombre}  peak={t_full[idx_peak]:.2f}s  cuasi={t_full[idx_quasi]:.2f}s  [base]")
                resultados[nombre] = res

                if GUARDAR_FIGURAS:
                    if zona_key == "L":
                        fig_path = os.path.join(dir_L, f"etapas_{nombre}.png")
                    elif zona_key == "viga175":
                        fig_path = os.path.join(dir_viga175, f"etapas_{nombre}.png")
                    else:
                        fig_path = os.path.join(dir_viga250, f"etapas_{nombre}.png")

                    graficar_etapas(
                        res,
                        tiempos_raw=t_full,
                        v_raw=v_media,
                        titulo=f"Etapas — {nombre}",
                        output_path=fig_path
                    )

    # ── Guardar JSON ──────────────────────────────────────────
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n✅ Guardado: {OUTPUT_JSON}")
    print(f"   {len(resultados)} entradas")

    if fallbacks:
        print(f"\n⚠️  Fallbacks:")
        for c in fallbacks:
            print(f"   - {c}")
    else:
        print("\n✅ Sin fallbacks.")
    
    if USAR_NOTIFICACION:
        try:
            toaster = ToastNotifier()
            toaster.show_toast("VSCode", "¡Tu código de Python terminó exitosamente!", duration=5, threaded=False)
        except Exception as e:
            print(f"⚠ No se pudo mostrar la notificación de Windows: {e}")