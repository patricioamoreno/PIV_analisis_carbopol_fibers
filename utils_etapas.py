"""
utils_etapas.py
===============
PASO 2 del pipeline: módulo auxiliar que cualquier script importa
para obtener las etapas precalculadas sin releer los archivos PIV.

Uso típico en cualquier script (boxplots, compare, stats, etc.):

    from utils_etapas import cargar_etapas, buscar_carpetas, promediar_grupo_cached

    etapas = cargar_etapas()   # lee el JSON una vez

    # Obtener archivos de una etapa específica
    files_cuasi = get_files_por_etapa(etapas, "m89-toma-1-n-0000-car-05-piv", "cuasi")

    # Promediar campo en una etapa (para compare.py, stats.py, etc.)
    df_prom = promediar_etapa(etapas, "m89-toma-1-n-0000-car-05-piv", "cuasi")
"""

import os
import glob
import json
import re
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURACIÓN CENTRAL — editar solo aquí
# ==========================================
BASE_PATH   = "C:/Users/elisa/Desktop/PIV_INTERPOLADO"
ETAPAS_JSON = "C:/Users/elisa/Desktop/Memoria PIV/etapas.json"

# Carpeta donde se guardan los campos promediados en disco
CACHE_PATH  = "C:/Users/elisa/Desktop/Memoria PIV/cache_campos"

# ==========================================
# FUNCIONES DE APOYO (igual a tus scripts)
# ==========================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]

def extraer_timestamp(file_path):
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "timestamp_s:" in line:
                    return float(line.split(":")[1].strip())
    except Exception:
        return 0.0
    return 0.0

def cargar_y_corregir(file_path):
    columns = ['x', 'y', 'u', 'v', 'mag', 'valid']
    df = pd.read_csv(file_path, sep=r'\s+', comment='#', names=columns)
    df['y'] *= -1
    df['v'] *= -1
    df['v_mag'] = np.sqrt(df['u']**2 + df['v']**2)
    return df

def get_files(carpeta):
    ruta = os.path.join(BASE_PATH, carpeta)
    return sorted(glob.glob(os.path.join(ruta, "*.txt")), key=natural_sort_key)

# ==========================================
# CARGA DEL JSON
# ==========================================

def cargar_etapas(json_path=ETAPAS_JSON):
    """
    Carga el JSON generado por calcular_etapas.py.
    Llama esto UNA vez al inicio de tu script y pasa
    el resultado a las demás funciones.

    Retorna dict: { nombre_carpeta: { etapas, t_peak, t_quasi, ... } }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"No se encontró {json_path}.\n"
            f"Primero corre calcular_etapas_polilinea.py para generarlo."
        )
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# ACCESO A ARCHIVOS POR ETAPA
# ==========================================

def get_files_por_etapa(etapas, carpeta, etapa="cuasi"):
    """
    Retorna la lista de archivos .txt correspondientes a una etapa.

    Parámetros
    ----------
    etapas  : dict  — resultado de cargar_etapas()
    carpeta : str   — nombre de la carpeta (clave en el JSON)
    etapa   : str   — "inicio", "transicion" o "cuasi"

    Retorna
    -------
    list[str] — rutas completas de los archivos de esa etapa
    """
    if carpeta not in etapas:
        print(f"⚠ '{carpeta}' no está en etapas.json")
        return []

    i_ini, i_fin = etapas[carpeta]['etapas'][etapa]
    files = get_files(carpeta)

    if not files:
        print(f"⚠ Sin archivos en {carpeta}")
        return []

    return files[i_ini:i_fin]

# ==========================================
# SINCRONIZACIÓN ENTRE DOS CARPETAS
# ==========================================

def t_sinc(etapas, carpeta1, carpeta2, etapa="cuasi"):
    """
    Retorna el tiempo de corte sincronizado (el máximo entre las dos carpetas)
    para que AMBAS estén en la etapa solicitada antes de promediar.

    Parámetros
    ----------
    etapas   : dict — resultado de cargar_etapas()
    carpeta1 : str
    carpeta2 : str
    etapa    : str  — "inicio", "transicion" o "cuasi"

    Retorna
    -------
    float — timestamp de sincronización en segundos
    """
    def t_inicio_etapa(carpeta):
        if carpeta not in etapas:
            raise KeyError(f"'{carpeta}' no está en etapas.json")
        i_ini, _ = etapas[carpeta]['etapas'][etapa]
        files = get_files(carpeta)
        if not files or i_ini >= len(files):
            return 0.0
        return extraer_timestamp(files[i_ini])

    t1 = t_inicio_etapa(carpeta1)
    t2 = t_inicio_etapa(carpeta2)
    return max(t1, t2)

# ==========================================
# PROMEDIO DE CAMPO POR ETAPA
# ==========================================

def promediar_etapa(etapas, carpeta, etapa="cuasi"):
    """
    Promedia 'celdita a celdita' todos los frames de una etapa.
    Usando los índices del JSON en vez de buscar el tiempo de corte.

    Retorna DataFrame con columnas x, y, u, v, v_mag (promediadas).
    Retorna None si no hay datos.
    """
    files = get_files_por_etapa(etapas, carpeta, etapa)

    if not files:
        print(f"⚠ Sin archivos para {carpeta} / etapa '{etapa}'")
        return None

    data = [cargar_y_corregir(f) for f in files]
    return pd.concat(data).groupby(['x', 'y']).mean().reset_index()

# ==========================================
# PROMEDIO DESDE TIEMPO ABSOLUTO
# ==========================================

def promediar_desde_t(carpeta, t_corte):
    """
    Promedia todos los frames de una carpeta desde t_corte en adelante.

    Parámetros
    ----------
    carpeta : str   — nombre de la carpeta (sin BASE_PATH)
    t_corte : float — timestamp mínimo en segundos (inclusive)

    Retorna DataFrame con x, y, u, v, v_mag promediados, o None si no hay datos.
    """
    ruta  = os.path.join(BASE_PATH, carpeta)
    files = sorted(glob.glob(os.path.join(ruta, "*.txt")), key=natural_sort_key)
    data  = [cargar_y_corregir(f) for f in files if extraer_timestamp(f) >= t_corte]
    if not data:
        print(f"⚠ Sin frames >= {t_corte:.2f}s en {carpeta}")
        return None
    return pd.concat(data).groupby(['x', 'y']).mean().reset_index()

# ==========================================
# BÚSQUEDA DE CARPETAS
# ==========================================

def buscar_carpetas(etapas, reo=None, conc=None, toma=None):
    """
    Filtra el JSON por reología, concentración y/o toma.

    Parámetros (todos opcionales)
    ----------
    reo  : str — "02" o "05"
    conc : str — "0000", "0750", "1500", "3000"
    toma : str — "1" o "2"

    Retorna list[str] — nombres de carpetas que coinciden.
    """
    resultado = list(etapas.keys())
    if reo:
        resultado = [c for c in resultado if f"car-{reo}" in c]
    if conc:
        resultado = [c for c in resultado if f"n-{conc}" in c]
    if toma:
        resultado = [c for c in resultado if f"-toma-{toma}-" in c]
    return sorted(resultado, key=natural_sort_key)


def promediar_grupo(etapas, carpetas, etapa="cuasi", t_corte_global=None):
    """
    Si t_corte_global es None  → cada carpeta usa su propio t_inicio de etapa
    Si t_corte_global es float → todas las carpetas promedian desde ese t
    Si t_corte_global es 'auto'→ calcula automáticamente el máximo t_inicio
    """
    if t_corte_global == 'auto':
        t_inicios = []
        for carpeta in carpetas:
            if carpeta not in etapas:
                continue
            i_ini, _ = etapas[carpeta]['etapas'][etapa]
            files = get_files(carpeta)
            if files and i_ini < len(files):
                t_inicios.append(extraer_timestamp(files[i_ini]))
        t_corte_global = max(t_inicios) if t_inicios else None
        print(f"    t_corte_global automático: {t_corte_global:.2f}s")

    campos = []
    for carpeta in carpetas:
        if carpeta not in etapas:
            continue

        if t_corte_global is not None:
            df = promediar_desde_t(carpeta, t_corte_global)
            modo = f"desde t={t_corte_global:.2f}s"
        else:
            df = promediar_etapa(etapas, carpeta, etapa)
            i_ini, i_fin = etapas[carpeta]['etapas'][etapa]
            modo = f"{i_fin - i_ini} frames propios"

        if df is not None:
            campos.append(df)
            print(f"    {carpeta}: {modo} | {len(df)} pts")

    if not campos:
        print(f"  ⚠ Sin datos para el grupo en etapa '{etapa}'")
        return None, []

    df_mean = pd.concat(campos).groupby(['x', 'y']).mean().reset_index()
    return df_mean, campos


# ==========================================
# CACHÉ EN DISCO — CAMPOS PROMEDIADOS
# ==========================================
# Permite guardar los resultados de promediar_grupo en disco (parquet)
# para no releer los archivos PIV en cada ejecución.
#
# Uso en cualquier script:
#   from utils_etapas import promediar_grupo_cached, limpiar_cache_disco
#
#   df_mean, df_list = promediar_grupo_cached(etapas, carpetas, etapa="cuasi")
#
# Primera vez  → calcula con promediar_grupo y guarda en disco
# Siguientes   → carga desde disco directo (< 1 segundo por campo)
# Recalcular   → promediar_grupo_cached(..., forzar=True)
# Borrar todo  → limpiar_cache_disco()

def _cache_nombre(carpetas, etapa, t_corte_global=None):
    """Hash MD5 corto como nombre único para esta combinación."""
    import hashlib
    key_str = "_".join(sorted(carpetas)) + f"_{etapa}"
    if t_corte_global is not None:
        key_str += f"_t{t_corte_global:.3f}"
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def promediar_grupo_cached(etapas, carpetas, etapa="cuasi",
                           t_corte_global=None, forzar=False):
    """
    Versión de promediar_grupo con caché en disco.
    Misma firma y mismo resultado que promediar_grupo — solo agrega
    la capa de persistencia en disco.

    Parámetros
    ----------
    etapas         : dict — resultado de cargar_etapas()
    carpetas       : list[str]
    etapa          : str
    t_corte_global : float o None (igual que promediar_grupo)
    forzar         : bool — si True, ignora caché y recalcula desde cero

    Retorna
    -------
    (df_mean, df_list) — igual que promediar_grupo
    """
    os.makedirs(CACHE_PATH, exist_ok=True)
    key       = _cache_nombre(carpetas, etapa, t_corte_global)
    ruta_mean = os.path.join(CACHE_PATH, f"{key}_mean.parquet")
    ruta_meta = os.path.join(CACHE_PATH, f"{key}_meta.json")

    # ── Cargar desde disco si existe ───────────────────────
    if not forzar and os.path.exists(ruta_mean) and os.path.exists(ruta_meta):
        with open(ruta_meta, 'r') as f:
            meta = json.load(f)
        df_mean = pd.read_parquet(ruta_mean)
        df_list = []
        for i in range(meta['n_list']):
            ruta_i = os.path.join(CACHE_PATH, f"{key}_list{i}.parquet")
            if os.path.exists(ruta_i):
                df_list.append(pd.read_parquet(ruta_i))
        print(f"    ⚡ Caché cargado: {key} ({meta['n_list']} toma(s))")
        return df_mean, df_list

    # ── Calcular y guardar ─────────────────────────────────
    print(f"    🔄 Calculando y guardando en caché: {key}")
    df_mean, df_list = promediar_grupo(
        etapas, carpetas, etapa=etapa, t_corte_global=t_corte_global)

    if df_mean is not None:
        df_mean.to_parquet(ruta_mean, index=False)
        for i, df in enumerate(df_list):
            df.to_parquet(
                os.path.join(CACHE_PATH, f"{key}_list{i}.parquet"), index=False)
        meta = {
            'carpetas'       : list(carpetas),
            'etapa'          : etapa,
            't_corte_global' : t_corte_global,
            'n_list'         : len(df_list),
        }
        with open(ruta_meta, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"    💾 Guardado: {key} "
              f"({len(df_list)} toma(s), {len(df_mean)} pts en df_mean)")

    return df_mean, df_list


def limpiar_cache_disco():
    """
    Elimina todos los archivos de caché en disco.
    Úsalo cuando cambien los datos PIV o los criterios de etapas.
    """
    import shutil
    if os.path.exists(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)
        print(f"  🧹 Caché eliminado: {CACHE_PATH}")
    else:
        print(f"  ℹ️ No había caché en disco.")