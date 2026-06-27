"""
carga_real.py
=============
Carga los datos REALES del experimento y los convierte al DataFrame largo que
consumen las 4 capas (capas.py).

Entradas
--------
1. PIV (campo de fluido), como carpeta de arrays .npy:
     x, y, t, v_mag, vort, gamma_dot, zona  (cada uno (N_puntos,))
   La columna 'zona' YA viene calculada por el pipeline PIV con etiquetas:
     Vf1c1, Vf1c2, Vf2c1, Vf2c2, Vf2c3  (celdas de la viga)
     Z1, Z2, Z3                          (zonas de la L)

2. Fibras (foto final / ultimo frame), CSV con columnas:
     track_id, timestamp_s, x_mm, y_mm, angle_deg, cam_name

3. Etapas (transicion / cuasi), JSON externo. Formato flexible (ver
   cargar_etapas). Si no se entrega, se usa un fallback por umbral de v_mag.

Decisiones de adaptacion (ver informe)
--------------------------------------
- PREDICTORES: V=v_mag, omega=|vort|, gamma_dot=gamma_dot.
- Outliers extremos (gamma_dot/vort llegan a 1e6 por gradientes en bordes):
  se agregan por MEDIANA y se hace clip al percentil 99 antes de agregar.
- Asignacion de fibras a zona: por BOUNDING BOX real de cada zona del PIV
  (no por grilla teorica), porque las zonas reales no llenan toda la grilla.
- theta de la fibra = angle_deg.
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd

# Predictores estandar que esperan las capas
PREDICTORES = ["V", "omega", "gamma_dot"]
ETAPAS = ["transicion", "cuasi"]

# Mapeo de columnas PIV -> predictores
PIV_VARS = {"V": "v_mag", "omega": "vort", "gamma_dot": "gamma_dot"}

CLIP_PERCENTIL = 99.0   # recorte de outliers extremos antes de agregar


# ----------------------------------------------------------------------
# 1. CARGA DEL PIV
# ----------------------------------------------------------------------
# Mapeo: nombre de columna interna -> posibles nombres en el archivo PIV.
# Si tu .npz usa otras claves, añade los alias aqui.
ALIAS = {
    "x": ["x"],
    "y": ["y"],
    "t": ["t", "time", "tiempo"],
    "v_mag": ["v_mag", "vmag", "V", "speed"],
    "vort": ["vort", "vorticidad", "omega", "w"],
    "gamma_dot": ["gamma_dot", "gammadot", "gamma", "shear_rate"],
    "zona": ["zona", "zone", "region"],
}


def _resolver_fuente_piv(piv_path):
    """
    Acepta:
      - ruta a un archivo .npz                -> lo usa directo
      - carpeta con UN .npz dentro            -> usa ese .npz
      - carpeta con varios .npz               -> usa el primero (avisa)
      - carpeta con .npy sueltos (formato viejo) -> los lee como antes
    Devuelve ('npz', ruta_npz) o ('npy', carpeta).
    """
    if os.path.isfile(piv_path) and piv_path.endswith(".npz"):
        return "npz", piv_path
    if os.path.isdir(piv_path):
        npzs = sorted(glob.glob(os.path.join(piv_path, "*.npz")))
        if len(npzs) == 1:
            return "npz", npzs[0]
        if len(npzs) > 1:
            print(f"[aviso] {len(npzs)} .npz en la carpeta; uso el primero: "
                  f"{os.path.basename(npzs[0])}. Para varias tomas usa "
                  f"cargar_varias_tomas().")
            return "npz", npzs[0]
        # sin .npz -> intentar .npy sueltos
        if glob.glob(os.path.join(piv_path, "*.npy")):
            return "npy", piv_path
    raise FileNotFoundError(
        f"No encontre .npz ni .npy en '{piv_path}'. Pasa la ruta a un .npz "
        f"o a la carpeta caches_zonas que lo contiene.")


def _buscar_clave(disponibles, candidatos, archivo=""):
    for c in candidatos:
        if c in disponibles:
            return c
    raise KeyError(
        f"No encontre ninguna de {candidatos} en el PIV {archivo}. "
        f"Claves disponibles: {sorted(disponibles)}. "
        f"Edita el diccionario ALIAS en carga_real.py para mapear tu nombre.")


def cargar_piv(piv_path):
    """
    Lee el PIV (de .npz o de .npy sueltos) y devuelve un DataFrame por punto:
      x, y, t, V, omega, gamma_dot, zona
    'omega' se guarda como |vort| (magnitud de rotacion; el signo es sentido
    de giro y no aporta a 'cuanta' rotacion hay).
    Detecta el formato automaticamente y resuelve nombres de clave via ALIAS.
    """
    tipo, fuente = _resolver_fuente_piv(piv_path)

    if tipo == "npz":
        z = np.load(fuente, allow_pickle=True)
        disp = set(z.files)
        get = lambda k: z[_buscar_clave(disp, ALIAS[k], os.path.basename(fuente))]
    else:  # npy sueltos
        disp = {os.path.basename(f)[:-4]
                for f in glob.glob(os.path.join(fuente, "*.npy"))}
        get = lambda k: np.load(
            os.path.join(fuente, _buscar_clave(disp, ALIAS[k]) + ".npy"),
            allow_pickle=True)

    df = pd.DataFrame({
        "x": get("x").astype(float),
        "y": get("y").astype(float),
        "t": get("t").astype(float),
        "V": get("v_mag").astype(float),
        "omega": np.abs(get("vort").astype(float)),
        "gamma_dot": get("gamma_dot").astype(float),
        "zona": get("zona").astype(str),
    })
    return df


def bounding_boxes(df_piv):
    """Bounding box (x0,x1,y0,y1) de cada zona real del PIV."""
    boxes = {}
    for z, g in df_piv.groupby("zona"):
        boxes[z] = (g.x.min(), g.x.max(), g.y.min(), g.y.max())
    return boxes


# ----------------------------------------------------------------------
# 2. DEFINICION DE ETAPAS
# ----------------------------------------------------------------------
def cargar_etapas(path_json=None, df_piv=None, zonas=None, toma=None,
                  campo_corte="t_quasi"):
    """
    Devuelve un dict {zona: t_corte} donde t <= t_corte es 'transicion' y
    t > t_corte es 'cuasi'. Soporta varios formatos de JSON:

      D) {"m70-...-piv_Vf1c1": {"t_quasi": 7.7, "t_peak": 1.4, ...}, ...}
         (FORMATO REAL: clave = {nombre_toma}_{zona}, valor = dict con tiempos.
          'toma' filtra las claves de esa toma; campo_corte elige que tiempo
          marca el corte, por defecto t_quasi = inicio del cuasi-estacionario.)
      A) {"Vf1c1": 6.2, "Vf2c1": 5.8, ...}              # corte por zona
      B) {"global": 6.0}                                # un corte para todas
      C) {"Vf1c1": {"transicion":[0,6.2],"cuasi":[6.2,20]}, ...}  # rangos

    Si path_json es None, usa un fallback: detecta el corte por el decaimiento
    de la mediana de v_mag de cada zona (instante donde v_mag cae bajo el 30%
    de su maximo suavizado).
    """
    zonas = zonas if zonas is not None else (
        sorted(df_piv.zona.unique()) if df_piv is not None else [])

    if path_json and os.path.exists(path_json):
        with open(path_json) as f:
            data = json.load(f)

        # --- Deteccion de formato D (el real) ---
        es_formato_D = any(
            isinstance(v, dict) and campo_corte in v for v in data.values())

        if es_formato_D:
            cortes = _cortes_formato_D(data, zonas, toma, campo_corte)
        elif "global" in data and len(data) == 1:
            cortes = {z: float(data["global"]) for z in zonas}
        else:
            cortes = {}
            for z in zonas:
                if z in data:
                    v = data[z]
                    cortes[z] = float(v["transicion"][1]) if isinstance(v, dict) \
                        else float(v)
                else:
                    cortes[z] = None

        faltan = [z for z in zonas if cortes.get(z) is None]
        if faltan and df_piv is not None:
            cortes.update(_fallback_cortes(df_piv, faltan))
        return cortes

    return _fallback_cortes(df_piv, zonas)


def _cortes_formato_D(data, zonas, toma, campo_corte):
    """
    Extrae {zona: t_corte} del formato real para una toma dada.
    La clave del JSON es '{nombre_toma}_{zona}'. Se identifica la zona como
    el sufijo tras el ultimo '_', y la toma como el resto. Si 'toma' se da,
    solo se usan las claves de esa toma; si no, se toma la unica toma presente
    (o se avisa si hay varias).
    """
    registros = []
    for k, v in data.items():
        if not isinstance(v, dict) or campo_corte not in v:
            continue
        toma_key, _, zona = k.rpartition("_")
        registros.append((toma_key, zona, v[campo_corte]))

    if not registros:
        return {z: None for z in zonas}

    tomas_disp = sorted({r[0] for r in registros})

    if toma is not None:
        cod = _codigo_toma(toma)
        candidatas = [t for t in tomas_disp
                      if t == toma or (cod and _codigo_toma(t) == cod
                                       and _mismo_perfil(t, toma))]
        if not candidatas:
            candidatas = [t for t in tomas_disp
                          if cod and _codigo_toma(t) == cod]
        toma_sel = candidatas[0] if candidatas else None
        if toma_sel is None:
            print(f"[aviso] No encontre la toma '{toma}' en el JSON de etapas. "
                  f"Tomas disponibles: {tomas_disp[:5]}... Usare fallback.")
    else:
        toma_sel = tomas_disp[0] if len(tomas_disp) == 1 else None
        if toma_sel is None:
            print(f"[aviso] El JSON tiene {len(tomas_disp)} tomas y no se "
                  f"especifico cual. Usando la primera: {tomas_disp[0]}")
            toma_sel = tomas_disp[0]

    cortes = {z: None for z in zonas}
    for tk, zona, tcut in registros:
        if tk == toma_sel and zona in cortes:
            cortes[zona] = float(tcut)
    return cortes


def _codigo_toma(nombre):
    """Extrae el codigo mNN del nombre (ej. 'm74-toma-1...' -> 'm74')."""
    m = re.search(r"m(\d+)", os.path.basename(str(nombre)).lower())
    return f"m{m.group(1)}" if m else None


def _mismo_perfil(a, b):
    """Compara reologia/concentracion entre dos nombres (car-0X, n-XXXX)."""
    def perfil(s):
        car = re.search(r"car-?(\d+)", str(s).lower())
        n = re.search(r"n-?(\d+)", str(s).lower())
        return (car.group(1) if car else None, n.group(1) if n else None)
    return perfil(a) == perfil(b)


def _fallback_cortes(df_piv, zonas):
    """Corte automatico por decaimiento de v_mag (mediana movil por zona)."""
    cortes = {}
    for z in zonas:
        g = df_piv[df_piv.zona == z]
        if g.empty:
            cortes[z] = None
            continue
        # mediana de v_mag en bins de tiempo
        tb = np.linspace(g.t.min(), g.t.max(), 40)
        idx = np.clip(np.digitize(g.t, tb) - 1, 0, len(tb) - 2)
        med = pd.Series(g.V.values).groupby(idx).median()
        centros = 0.5 * (tb[:-1] + tb[1:])
        vmax = med.max()
        umbral = 0.30 * vmax
        bajo = med[med < umbral]
        if len(bajo):
            cortes[z] = float(centros[bajo.index[0]])
        else:
            cortes[z] = float(np.median(g.t))   # sin decaimiento claro
    return cortes


def etiquetar_etapa(df_piv, cortes):
    """Agrega columna 'etapa' (transicion/cuasi) segun corte por zona."""
    df = df_piv.copy()
    tc = df["zona"].map(cortes)
    df["etapa"] = np.where(df["t"] <= tc, "transicion", "cuasi")
    return df


# ----------------------------------------------------------------------
# 3. AGREGACION DE PREDICTORES POR ZONA Y ETAPA
# ----------------------------------------------------------------------
def agregar_fluido(df_piv_etapas, clip_p=CLIP_PERCENTIL):
    """
    Mediana robusta de cada predictor por (zona, etapa), con clip de outliers
    al percentil clip_p calculado dentro de cada (zona, etapa).
    Devuelve DataFrame: zona, etapa, V, omega, gamma_dot, n_puntos.
    """
    filas = []
    for (z, e), g in df_piv_etapas.groupby(["zona", "etapa"]):
        fila = {"zona": z, "etapa": e, "n_puntos": len(g)}
        for p in PREDICTORES:
            v = g[p].to_numpy(float)
            hi = np.nanpercentile(v, clip_p)
            v = np.clip(v, None, hi)
            fila[p] = float(np.nanmedian(v))
        filas.append(fila)
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# 4. CARGA DE FIBRAS Y ASIGNACION A ZONA
# ----------------------------------------------------------------------
def cargar_fibras(path_csv):
    """Lee un CSV de fibras (ultimo frame) -> DataFrame con theta, x_mm, y_mm."""
    df = pd.read_csv(path_csv)
    df = df.rename(columns={"angle_deg": "theta"})
    df["fibra_id"] = np.arange(len(df))
    return df[["fibra_id", "x_mm", "y_mm", "theta", "track_id"]]


def asignar_fibras_zona(df_fibras, boxes):
    """
    Asigna cada fibra a la zona cuya bounding box (del PIV) la contiene.
    Si cae en varias (cajas solapadas), elige la de menor area (mas especifica).
    Si no cae en ninguna, zona = NaN (se descarta del analisis).
    """
    areas = {z: (x1 - x0) * (y1 - y0) for z, (x0, x1, y0, y1) in boxes.items()}
    zonas = []
    for px, py in zip(df_fibras.x_mm, df_fibras.y_mm):
        candidatas = [z for z, (x0, x1, y0, y1) in boxes.items()
                      if x0 <= px <= x1 and y0 <= py <= y1]
        if not candidatas:
            zonas.append(np.nan)
        else:
            zonas.append(min(candidatas, key=lambda z: areas[z]))
    out = df_fibras.copy()
    out["zona"] = zonas
    return out


# ----------------------------------------------------------------------
# 5. ENSAMBLE FINAL -> DataFrame largo (fibra x etapa)
# ----------------------------------------------------------------------
def construir_largo(df_fibras_zona, df_fluido):
    """
    Cruza cada fibra (con su zona) con los predictores de fluido de esa zona
    en cada etapa. Resultado: una fila por (fibra, etapa).
    Descarta fibras sin zona asignada.
    """
    fib = df_fibras_zona.dropna(subset=["zona"]).copy()
    # producto fibra x etapa via merge con la tabla de fluido (que tiene 2
    # filas por zona: transicion y cuasi)
    largo = fib.merge(df_fluido, on="zona", how="inner",
                      suffixes=("", "_fluido"))
    # columnas finales que esperan las capas
    largo["zona_id"] = pd.factorize(largo["zona"])[0]
    cols = ["fibra_id", "zona_id", "zona", "x_mm", "y_mm", "theta", "etapa",
            "V", "omega", "gamma_dot", "track_id", "n_puntos"]
    return largo[cols].reset_index(drop=True)


def cargar_todo(dir_piv, path_fibras_csv, path_etapas_json=None,
                verbose=True):
    """
    Pipeline completo de carga real. Devuelve (df_largo, diagnostico).
    diagnostico es un dict con info util (cortes de etapa, n fibras fuera, etc.)
    """
    df_piv = cargar_piv(dir_piv)
    boxes = bounding_boxes(df_piv)
    zonas = sorted(df_piv.zona.unique())

    # nombre de la toma derivado del archivo PIV, para emparejar con el JSON
    nombre_toma = os.path.splitext(os.path.basename(
        _resolver_fuente_piv(dir_piv)[1]))[0]
    cortes = cargar_etapas(path_etapas_json, df_piv, zonas, toma=nombre_toma)
    df_piv_e = etiquetar_etapa(df_piv, cortes)
    df_fluido = agregar_fluido(df_piv_e)

    df_fib = cargar_fibras(path_fibras_csv)
    df_fib_z = asignar_fibras_zona(df_fib, boxes)
    n_fuera = df_fib_z["zona"].isna().sum()

    df_largo = construir_largo(df_fib_z, df_fluido)

    diag = {
        "cortes_etapa": cortes,
        "zonas_piv": zonas,
        "n_fibras_total": len(df_fib),
        "n_fibras_asignadas": len(df_fib) - n_fuera,
        "n_fibras_fuera": int(n_fuera),
        "fibras_por_zona": df_fib_z["zona"].value_counts(dropna=False).to_dict(),
        "tabla_fluido": df_fluido,
        "usó_json": bool(path_etapas_json and os.path.exists(path_etapas_json)),
    }
    if verbose:
        print(f"PIV: {len(df_piv):,} puntos | zonas: {zonas}")
        print(f"Etapas {'(JSON)' if diag['usó_json'] else '(fallback v_mag)'}: "
              + ", ".join(f"{z}@{c:.1f}s" for z, c in cortes.items()
                          if c is not None))
        print(f"Fibras: {diag['n_fibras_asignadas']}/{diag['n_fibras_total']} "
              f"asignadas ({n_fuera} fuera de toda zona)")
        print(f"Filas largo (fibra x etapa): {len(df_largo)}")
    return df_largo, diag