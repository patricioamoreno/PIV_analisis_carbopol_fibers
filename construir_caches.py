"""
construir_caches.py
===================
Construye los cachés completos (.npz) de velocidad a lo largo de cada
polilínea para TODAS las carpetas PIV disponibles, incluyendo las que
están excluidas del análisis.

Genera un archivo por carpeta y zona:
  cache_completo/{carpeta}_completo.npz          → zona L
  cache_completo/viga175_{carpeta}_completo.npz  → viga 175 mm
  cache_completo/viga250_{carpeta}_completo.npz  → viga 250 mm

Uso:
  python construir_caches.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from win10toast import ToastNotifier
    USAR_NOTIFICACION = True
except ImportError:
    USAR_NOTIFICACION = False
    print("⚠ win10toast no está instalado. Las notificaciones están desactivadas.")

# ============================================================
# CONFIGURACIÓN — editar aquí
# ============================================================

BASE_PATH  = BASE_PATH = Path("../PIV_INTERPOLADO")
CACHE_DIR  = "cache_completo"

# True → reconstruye aunque el caché ya exista
RECALCULO  = True

# ── Geometría de la L ─────────────────────────────────────────
# N_PUNTOS_LINEA se fija a la resolución espacial REAL del PIV de la Cam 4
# (Δy ≈ 0.748 mm) sobre el FOV de ~36 mm: ~48 vectores independientes.
# Sobremuestrear (p.ej. 200 pts) NO añade información: el IDW interpola
# entre vectores ya interpolados y los puntos extra están correlacionados
# artificialmente, suavizando γ̇=dv/ds. Mantener 1 punto por Δy_PIV.
LINEA_NPY      = "Polilinea_L/linea_salida.npy"
N_PUNTOS_LINEA = 48           # antes 200 (4× sobremuestreado vs PIV)
ANGULO_L_DEG   = -30.0

# ── Geometría de las vigas ────────────────────────────────────
# Δy_PIV en vigas: car-0.5% → ~1.0 mm (W=16px), car-0.2% → ~2.0 mm (W=32px).
# Sobre los 75 mm de perfil, el caso más fino (car-0.5%) admite ~75 puntos
# independientes. 75 mantiene 1 punto por Δy_PIV en el caso más exigente y
# evita sobremuestrear; el suavizado posterior maneja el caso car-0.2%.
X_VIGA        = [175, 250]
Y_VIGA_MIN    = -75.0
Y_VIGA_MAX    =  0.0
N_PUNTOS_VIGA =  75           # antes 80 (≈ 1 punto por Δy_PIV de car-0.5%)

# ── Relleno de huecos internos (zona L) ───────────────────────
# Máximo nº de puntos NaN CONSECUTIVOS que se permite rellenar por
# interpolación lineal 1D a lo largo de la polilínea. Un hueco de 1-3 puntos
# (≲ 2.2 mm sobre la L, con Δs ≈ 0.75 mm) es un fallo local de correlación
# PIV y su relleno es defendible. Un hueco mayor implica extrapolar sobre una
# región sin vectores fiables: el perfil resultante sería inventado y
# contaminaría v_∥(s,t) y sobre todo γ̇ = dv/ds, que amplifica cualquier
# rampa artificial. Esos tramos se conservan como NaN.
# Poner 0 desactiva por completo el relleno 1D.
MAX_HUECO_INTERP = 3

# Zonas a procesar: (prefijo, etiqueta)
ZONAS = [
    ("",        "L"),
    ("viga175", "viga175"),
    ("viga250", "viga250"),
]

# ============================================================
# UTILIDADES
# ============================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


def carpetas_disponibles(base_path):
    """Lista todas las carpetas PIV disponibles en BASE_PATH."""
    carpetas = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    return sorted(carpetas, key=natural_sort_key)

# ============================================================
# GEOMETRÍA
# ============================================================

def geometria_zona(prefijo):
    if prefijo == "":
        pts       = np.load(LINEA_NPY)
        p0, p1    = pts[0], pts[1]
        t         = np.linspace(0, 1, N_PUNTOS_LINEA)
        xs        = p0[0] + t * (p1[0] - p0[0])
        ys        = p0[1] + t * (p1[1] - p0[1])
        ang_rad   = np.radians(ANGULO_L_DEG)
        dir_flujo = np.array([np.cos(ang_rad), np.sin(ang_rad)])
    else:
        x_viga    = float(prefijo.replace("viga", ""))
        ys        = np.linspace(Y_VIGA_MIN, Y_VIGA_MAX, N_PUNTOS_VIGA)
        xs        = np.full_like(ys, x_viga)
        dir_flujo = np.array([1.0, 0.0])
    return xs, ys, dir_flujo


# ============================================================
# LECTURA PIV
# ============================================================

def extraer_timestamp(fpath):
    try:
        with open(fpath, 'r') as f:
            for line in f:
                if "timestamp_s:" in line:
                    return float(line.split(":")[1].strip())
    except Exception:
        pass
    return 0.0


def cargar_y_corregir(fpath):
    columns = ['x', 'y', 'u', 'v', 'mag', 'valid']
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=columns)
    df['y'] *= -1
    df['v'] *= -1
    df['v_mag'] = np.sqrt(df['u']**2 + df['v']**2)
    return df


def _tramos_nan(nans):
    """Genera (i_ini, i_fin) de cada racha contigua de True en `nans`.

    i_fin es exclusivo, de modo que la longitud del tramo es i_fin - i_ini.
    """
    n   = len(nans)
    i   = 0
    while i < n:
        if not nans[i]:
            i += 1
            continue
        j = i
        while j < n and nans[j]:
            j += 1
        yield i, j
        i = j


def rellenar_huecos_cortos(result, max_hueco=MAX_HUECO_INTERP):
    """Interpolación lineal 1D SOLO en huecos internos cortos.

    Un tramo de NaN se rellena únicamente si cumple las dos condiciones:
      (a) es interno — tiene datos válidos a ambos lados (no es un borde,
          donde interpolar equivaldría a extrapolar); y
      (b) su longitud es <= max_hueco puntos consecutivos.

    Cualquier tramo más largo se aborta y conserva sus NaN originales. La
    forma del array NO cambia: siempre se devuelve len(result) puntos, con
    NaN donde no hay información fiable.
    """
    if max_hueco <= 0:
        return result

    nans = np.isnan(result)
    if not nans.any() or nans.all():
        return result

    idx_arr   = np.arange(len(result))
    validos   = ~nans
    idx_val   = idx_arr[validos]
    val_val   = result[validos]

    for i_ini, i_fin in _tramos_nan(nans):
        # (a) Bordes: sin dato válido a un lado → no se toca (no extrapolar).
        if i_ini == 0 or i_fin == len(result):
            continue
        # (b) Hueco demasiado largo → abortar, conservar NaN.
        if (i_fin - i_ini) > max_hueco:
            continue
        hueco = idx_arr[i_ini:i_fin]
        result[hueco] = np.interp(hueco, idx_val, val_val)

    return result


def contaminados(nan_mask, radio):
    """Máscara de puntos NO fiables tras suavizar + derivar.

    Un filtro de ventana `w` y una derivada centrada propagan cualquier valor
    rellenado a sus vecinos. Marcar solo los NaN originales es insuficiente:
    los puntos válidos contiguos a un hueco ya arrastran el valor inventado.
    Se dilata la máscara de NaN en `radio` puntos a cada lado. Para
    uniform_filter1d(size=w) seguido de np.gradient, radio = w//2 + 1.
    """
    m = np.asarray(nan_mask, dtype=bool)
    out = m.copy()
    for d in range(1, radio + 1):
        out[d:]  |= m[:-d]      # propaga hacia adelante
        out[:-d] |= m[d:]       # propaga hacia atrás
    return out


def interpolar_campo(df_frame, xs, ys, col, dist_max_mm=None,
                     max_hueco=MAX_HUECO_INTERP):
    """IDW con 4 vecinos más cercanos.

    SIEMPRE devuelve un array de len(xs): valores interpolados donde hay
    datos PIV cercanos, NaN donde no los hay. Nunca devuelve None, para
    que el frame (y su timestamp) se conserve aunque el material no llegue
    a esta polilínea — así el eje temporal no se comprime y el
    espectrograma muestra blanco en esos instantes en vez de acortarse.

    El relleno 1D posterior está acotado a huecos internos de como máximo
    `max_hueco` puntos consecutivos (ver rellenar_huecos_cortos).
    """
    result = np.full(len(xs), np.nan)

    mask = df_frame['valid'] == 1
    df_v = df_frame[mask]
    if len(df_v) < 10:
        # Frame PIV degradado: no hay vectores fiables → fila de NaN.
        return result

    px, py   = df_v['x'].values, df_v['y'].values
    vals     = df_v[col].values
    dist_max2 = (dist_max_mm ** 2) if dist_max_mm is not None else None

    for i, (xi, yi) in enumerate(zip(xs, ys)):
        dist = (px - xi)**2 + (py - yi)**2
        if dist_max2 is not None and dist.min() > dist_max2:
            continue
        idx       = np.argpartition(dist, min(4, len(dist) - 1))[:4]
        w         = 1.0 / (dist[idx] + 1e-10)
        result[i] = np.sum(w * vals[idx]) / np.sum(w)

    # Solo rellenamos huecos internos por interpolación 1D cuando NO hay
    # límite de distancia (zona L). Con dist_max (vigas) los NaN son
    # físicos —material ausente— y se respetan.
    # En la L el relleno está ahora ACOTADO: huecos > max_hueco puntos
    # consecutivos se dejan como NaN en vez de inventar velocidad.
    if dist_max_mm is None:
        result = rellenar_huecos_cortos(result, max_hueco=max_hueco)
    return result


# ============================================================
# CACHÉ
# ============================================================

def nombre_base_carpeta(clave):
    for suf in ['_viga175', '_viga250', '_L']:
        if clave.endswith(suf):
            return clave[:-len(suf)]
    return clave

def _nombre_cache(carpeta, prefijo, usar_magnitud=False):
    sufijo = "_mag" if usar_magnitud else ""
    return (f"{prefijo}_{carpeta}_completo{sufijo}.npz" if prefijo
            else f"{carpeta}_completo{sufijo}.npz")

def cargar_cache_completo(carpeta, prefijo, cache_dir, usar_magnitud=False):
    path = os.path.join(cache_dir, _nombre_cache(carpeta, prefijo, usar_magnitud))
    if os.path.exists(path):
        data = np.load(path)
        return data['matriz'], data['tiempos']
    return None, None

def ss_linea_L():
    pts    = np.load(LINEA_NPY)
    p0, p1 = pts[0], pts[1]
    largo  = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
    return np.linspace(0, largo, N_PUNTOS_LINEA)

def ss_viga():
    return np.linspace(0, Y_VIGA_MAX - Y_VIGA_MIN, N_PUNTOS_VIGA)

def cache_existe(carpeta, prefijo, usar_magnitud=False):
    return os.path.exists(os.path.join(CACHE_DIR, _nombre_cache(carpeta, prefijo, usar_magnitud)))


def construir_cache(carpeta, prefijo, usar_magnitud=False):
    """Construye y guarda el caché completo para una carpeta y zona.
    
    usar_magnitud=False → guarda proyección sobre dir_flujo (comportamiento original)
    usar_magnitud=True  → guarda magnitud √(u²+v²), archivo con sufijo _mag
    """
    fname = _nombre_cache(carpeta, prefijo, usar_magnitud)
    path  = os.path.join(CACHE_DIR, fname)

    if cache_existe(carpeta, prefijo, usar_magnitud) and not RECALCULO:
        print(f"  ✓ Ya existe: {fname}")
        return

    xs, ys, dir_flujo = geometria_zona(prefijo)
    dist_max = 8.0 if prefijo != "" else None

    files = sorted(
        glob.glob(os.path.join(BASE_PATH, carpeta, "*.txt")),
        key=natural_sort_key
    )
    if not files:
        print(f"  ⚠ Sin archivos PIV: {carpeta}")
        return

    zona_str = prefijo or 'L'
    modo_str = "magnitud" if usar_magnitud else "proyección"
    print(f"  🔨 {zona_str}/{carpeta}  ({len(files)} frames) [{modo_str}]", flush=True)

    filas, tiempos = [], []
    n_vacios = 0
    for i, fpath in enumerate(files):
        if i % 100 == 0:
            print(f"      frame {i}/{len(files)}...", flush=True)
        df  = cargar_y_corregir(fpath)
        u_l = interpolar_campo(df, xs, ys, 'u', dist_max_mm=dist_max)
        v_l = interpolar_campo(df, xs, ys, 'v', dist_max_mm=dist_max)
        # interpolar_campo nunca devuelve None: siempre hay array (con NaN
        # donde no hay material). Guardamos TODOS los frames para preservar
        # el eje temporal completo de la corrida.
        if usar_magnitud:
            val = np.sqrt(u_l**2 + v_l**2).astype(np.float32)
        else:
            val = (u_l * dir_flujo[0] + v_l * dir_flujo[1]).astype(np.float32)
        if np.isnan(val).all():
            n_vacios += 1
        filas.append(val)
        tiempos.append(extraer_timestamp(fpath))

    if not filas:
        print(f"  ⚠ Sin archivos procesables: {carpeta}/{zona_str}")
        return

    if n_vacios:
        print(f"  ℹ {n_vacios}/{len(filas)} frames sin material en la línea "
              f"(guardados como NaN, eje temporal completo)", flush=True)

    matriz = np.array(filas, dtype=np.float32)

    # Diagnóstico: fracción de NaN sobre los frames que SÍ tienen material.
    # Con el relleno acotado esta cifra debe subir respecto a la versión
    # anterior; ese aumento es exactamente el dato que antes se inventaba.
    if prefijo == "":
        con_material = ~np.isnan(matriz).all(axis=1)
        if con_material.any():
            frac_nan = float(np.isnan(matriz[con_material]).mean())
            print(f"  ℹ NaN residual en frames con material: {frac_nan:5.1%} "
                  f"(huecos > {MAX_HUECO_INTERP} pts no rellenados)", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        path,
        matriz  = matriz,
        tiempos = np.array(tiempos, dtype=np.float64),
    )
    print(f"  💾 Guardado: {fname}", flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    carpetas = carpetas_disponibles(BASE_PATH)
    print(f"Carpetas encontradas: {len(carpetas)}\n")

    for carpeta in carpetas:
        print(f"\n{'='*55}")
        print(f"  {carpeta}")
        for prefijo, zona_label in ZONAS:
            # Caché original (proyección sobre dir_flujo) — no tocar
            construir_cache(carpeta, prefijo, usar_magnitud=False)
            # Caché de magnitud para todas las zonas (para comparar con PTV)
            construir_cache(carpeta, prefijo, usar_magnitud=True)

    print(f"\n✅ Listo. Cachés en: {CACHE_DIR}")

    if USAR_NOTIFICACION:
        try:
            toaster = ToastNotifier()
            toaster.show_toast("VSCode", "¡Tu código de Python terminó exitosamente!", duration=5, threaded=False)
        except Exception as e:
            print(f"⚠ No se pudo mostrar la notificación de Windows: {e}")