"""
construir_caches_zonas.py
=========================
Construye un caché .npz por carpeta PIV que guarda, frame por frame, los
puntos válidos YA PROCESADOS (zona asignada + γ̇ + vorticidad + plug).

Misma filosofía que construir_caches.py (polilíneas):
  - un archivo .npz por carpeta
  - np.savez_compressed
  - bandera RECALCULO para reconstruir o saltar
  - loader gemelo cargar_cache_zonas() que devuelve None si no existe

Diferencia de contenido respecto a las polilíneas:
  En polilíneas guardas una matriz [n_frames × n_puntos_linea] de UN escalar.
  Aquí cada frame tiene un nº variable de puntos PIV dispersos, y lo caro es
  el KNN de γ̇/vorticidad. Por eso se cachea el RESULTADO de ese paso pesado:
  todos los puntos válidos con sus campos derivados. Así, para analizar una
  zona, cambiar el umbral de plug o recomputar estadísticas NO hay que releer
  ni recalcular nada — basta filtrar el caché.

Estructura del .npz  (arrays paralelos, todos de largo N = nº total de
puntos válidos acumulados sobre todos los frames de la carpeta):
    frame_idx   int32    índice de frame (0..n_frames-1) de cada punto
    t           float64  timestamp del frame de cada punto
    x, y        float32  coords corregidas (y ya invertida)
    u, v        float32  componentes corregidas (v ya invertida)
    v_mag       float32  magnitud
    gamma_dot   float32  tasa de deformación  [s⁻¹]
    vort        float32  vorticidad dvdx-dudy [s⁻¹]
    zona        '<U12'   "Z1"/"Z2"/"Z3"/"Vf1c1".../"fuera"
  + metadatos:
    carpeta, reo, n_frames, k_vecinos

NOTA sobre plug: NO se guarda la etiqueta plug/no-plug, solo γ̇. Así puedes
cambiar UMBRAL_GAMMA_PLUG al analizar sin reconstruir el caché. La etiqueta
se deriva al vuelo en el loader o en el análisis.

Uso:
    python construir_caches_zonas.py
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from definir_zonas import asignar_zona

# ============================================================
# CONFIGURACIÓN — editar aquí
# ============================================================

BASE_PATH = Path("../PIV_INTERPOLADO")
CACHE_DIR = "cache_zonas"

# True → reconstruye aunque el caché ya exista
RECALCULO = False

# Vecinos para el ajuste lineal local (γ̇ y vorticidad). Igual que gamma_fields.
K_VECINOS = 6

# Filtros opcionales de carpetas (None = todas)
CAR_OBJETIVO = None     # p.ej. "02"
FIB_OBJETIVO = None     # p.ej. "1500"

# Solo cachear puntos que caen en alguna zona (descarta "fuera").
# Reduce mucho el tamaño del .npz. Pon False si quieres conservarlo todo.
SOLO_DENTRO_DE_ZONA = True

# ============================================================
# UTILIDADES (idénticas a tus otros scripts)
# ============================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


def carpetas_disponibles(base_path):
    carpetas = [d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))]
    return sorted(carpetas, key=natural_sort_key)


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
    """Lee frame PIV, corrige ejes (y*=-1, v*=-1) y filtra valid==1."""
    columns = ['x', 'y', 'u', 'v', 'mag', 'valid']
    df = pd.read_csv(fpath, sep=r'\s+', comment='#', names=columns)
    df['y'] *= -1
    df['v'] *= -1
    df['v_mag'] = np.sqrt(df['u']**2 + df['v']**2)
    df = df[df['valid'] == 1].reset_index(drop=True)
    return df


# ============================================================
# GRADIENTES POR KNN  →  γ̇  y  vorticidad  (de gamma_fields.py, extendido)
# ============================================================

def calcular_gradientes(df, k=K_VECINOS):
    x = df['x'].values; y = df['y'].values
    u = df['u'].values; v = df['v'].values
    n = len(x)
    gamma_dot = np.full(n, np.nan, dtype=np.float32)
    vort      = np.full(n, np.nan, dtype=np.float32)
    if n < 4:
        return gamma_dot, vort

    tree = cKDTree(np.column_stack([x, y]))
    for i in range(n):
        _, idxs = tree.query([x[i], y[i]], k=min(k + 1, n))
        idxs = np.atleast_1d(idxs)[1:]
        if len(idxs) < 3:
            continue
        dx = x[idxs] - x[i]; dy = y[idxs] - y[i]
        A  = np.column_stack([np.ones(len(idxs)), dx, dy])
        try:
            cu, *_ = np.linalg.lstsq(A, u[idxs], rcond=None)
            cv, *_ = np.linalg.lstsq(A, v[idxs], rcond=None)
            dudx, dudy = cu[1], cu[2]
            dvdx, dvdy = cv[1], cv[2]
            gamma_dot[i] = np.sqrt(2.0*dudx**2 + 2.0*dvdy**2 + (dudy + dvdx)**2)
            vort[i]      = dvdx - dudy
        except Exception:
            continue
    return gamma_dot, vort


# ============================================================
# CACHÉ  (mismo patrón que construir_caches.py)
# ============================================================

def _nombre_cache(carpeta):
    return f"{carpeta}_zonas.npz"


def cache_existe(carpeta):
    return os.path.exists(os.path.join(CACHE_DIR, _nombre_cache(carpeta)))


def cargar_cache_zonas(carpeta, cache_dir=CACHE_DIR):
    """
    Loader gemelo de cargar_cache_completo().
    Retorna un dict de arrays paralelos, o None si no existe el caché.

    dict keys: frame_idx, t, x, y, u, v, v_mag, gamma_dot, vort, zona
               (+ atributos sueltos: carpeta, reo, n_frames, k_vecinos)
    """
    path = os.path.join(cache_dir, _nombre_cache(carpeta))
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {
        'frame_idx': data['frame_idx'],
        't':         data['t'],
        'x':         data['x'],
        'y':         data['y'],
        'u':         data['u'],
        'v':         data['v'],
        'v_mag':     data['v_mag'],
        'gamma_dot': data['gamma_dot'],
        'vort':      data['vort'],
        'zona':      data['zona'],
        'carpeta':   str(data['carpeta']),
        'reo':       str(data['reo']),
        'n_frames':  int(data['n_frames']),
        'k_vecinos': int(data['k_vecinos']),
    }


def cache_a_dataframe(carpeta, cache_dir=CACHE_DIR):
    """Conveniencia: devuelve el caché como DataFrame (o None)."""
    c = cargar_cache_zonas(carpeta, cache_dir)
    if c is None:
        return None
    return pd.DataFrame({
        'frame_idx': c['frame_idx'], 't': c['t'],
        'x': c['x'], 'y': c['y'], 'u': c['u'], 'v': c['v'],
        'v_mag': c['v_mag'], 'gamma_dot': c['gamma_dot'],
        'vort': c['vort'], 'zona': c['zona'],
    })


def construir_cache(carpeta):
    """Construye y guarda el caché de zonas para una carpeta."""
    fname = _nombre_cache(carpeta)
    path  = os.path.join(CACHE_DIR, fname)

    if cache_existe(carpeta) and not RECALCULO:
        print(f"  ✓ Ya existe: {fname}")
        return

    files = sorted(glob.glob(os.path.join(BASE_PATH, carpeta, "*.txt")),
                   key=natural_sort_key)
    if not files:
        print(f"  ⚠ Sin archivos PIV: {carpeta}")
        return

    m_reo = re.search(r'car-(\d+)', carpeta)
    reo   = m_reo.group(1) if m_reo else ""

    print(f"  🔨 {carpeta}  ({len(files)} frames)", flush=True)

    # Acumuladores (listas de arrays por frame; se concatenan al final)
    A_frame, A_t = [], []
    A_x, A_y, A_u, A_v, A_mag = [], [], [], [], []
    A_gd, A_vo, A_zona = [], [], []

    for i, fpath in enumerate(files):
        if i % 100 == 0:
            print(f"      frame {i}/{len(files)}...", flush=True)
        df = cargar_y_corregir(fpath)
        if df.empty:
            continue

        gamma_dot, vort = calcular_gradientes(df, k=K_VECINOS)
        zona = asignar_zona(df['x'].values, df['y'].values)

        if SOLO_DENTRO_DE_ZONA:
            keep = zona != "fuera"
            if not keep.any():
                continue
        else:
            keep = np.ones(len(df), dtype=bool)

        n_keep = int(keep.sum())
        t_frame = extraer_timestamp(fpath)

        A_frame.append(np.full(n_keep, i, dtype=np.int32))
        A_t.append(np.full(n_keep, t_frame, dtype=np.float64))
        A_x.append(df['x'].values[keep].astype(np.float32))
        A_y.append(df['y'].values[keep].astype(np.float32))
        A_u.append(df['u'].values[keep].astype(np.float32))
        A_v.append(df['v'].values[keep].astype(np.float32))
        A_mag.append(df['v_mag'].values[keep].astype(np.float32))
        A_gd.append(gamma_dot[keep])
        A_vo.append(vort[keep])
        A_zona.append(zona[keep])

    if not A_frame:
        print(f"  ⚠ Sin puntos en zona para {carpeta}")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx = np.concatenate(A_frame),
        t         = np.concatenate(A_t),
        x         = np.concatenate(A_x),
        y         = np.concatenate(A_y),
        u         = np.concatenate(A_u),
        v         = np.concatenate(A_v),
        v_mag     = np.concatenate(A_mag),
        gamma_dot = np.concatenate(A_gd),
        vort      = np.concatenate(A_vo),
        zona      = np.concatenate(A_zona),
        carpeta   = carpeta,
        reo       = reo,
        n_frames  = len(files),
        k_vecinos = K_VECINOS,
    )
    n_pts = sum(len(a) for a in A_frame)
    print(f"  💾 Guardado: {fname}  ({n_pts:,} puntos)", flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    carpetas = carpetas_disponibles(BASE_PATH)
    if CAR_OBJETIVO:
        carpetas = [c for c in carpetas if f"car-{CAR_OBJETIVO}" in c]
    if FIB_OBJETIVO:
        carpetas = [c for c in carpetas if f"n-{FIB_OBJETIVO}" in c]

    print(f"Carpetas encontradas: {len(carpetas)}\n")
    for carpeta in carpetas:
        print(f"\n{'='*55}\n  {carpeta}")
        construir_cache(carpeta)

    print(f"\n✅ Listo. Cachés en: {CACHE_DIR}")