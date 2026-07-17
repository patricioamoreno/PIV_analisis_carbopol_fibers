"""
reconstruccion_lagrangiana.py
==============================
Sistema de atribucion de zona de TRANSICION por fibra con TRIPLE VALIDACION
CRUZADA (E1/E2/E3), replicando la metodologia de
github.com/LukasWolff2002/Exp_Func_PIV_PTV (carpeta reconstruccion_lagrangiana/),
adaptada a este proyecto (cache_zonas .npz, definir_zonas.asignar_zona,
etapas_zonas.json).

Problema que resuelve
---------------------
El Metodo 1 (zona final) y el Metodo 2 original (adveccion pura) no tienen
forma de VALIDARSE entre si. Este modulo agrega esa validacion construyendo
la misma atribucion por TRES caminos independientes y midiendo su
concordancia:

  E1 -- Track real (oro): moda de la zona observada DIRECTAMENTE en el track
        PTV medido, durante el regimen de transicion. Requiere que el track
        tenga frames dentro de la ventana de transicion.
  E2 -- Stitching simplificado: igual que E1, pero antes se re-enlazan
        segmentos de track que se cortan y retoman cerca en espacio/tiempo
        (gap-closing), para recuperar tracks fragmentados por perdida de
        deteccion. Version simplificada del E2 del repo de referencia (sin el
        chequeo de continuidad rotacional via omega).
  E3 -- Pathline PIV: integracion hacia atras de la posicion FINAL de la
        fibra sobre el campo (u,v) de cache_zonas, clasificando la zona MODAL
        visitada durante el regimen de transicion (no fraccion de tiempo como
        en la version anterior de construir_caches_adveccion.py). Inmune a la
        fragmentacion del track (no depende de deteccion continua), pero
        hereda el supuesto de particula pasiva (ver discusion previa).

Salidas
-------
  atribucion_E1_E2_E3.csv   -- una fila por (toma, track_id) con las 3
                               atribuciones + zona_final + concordancia
  resumen_concordancia.csv  -- tasa de acuerdo E1-E2, E1-E3, E2-E3 por toma

Requiere: cache_zonas/<carpeta>_zonas.npz (PIV, con u,v,frame_idx),
carpetas con ptv_merged.json (PTV completo, no solo ultimo frame),
etapas_zonas.json.

Uso:
    python reconstruccion_lagrangiana.py
"""

import os
import re
import glob
import time
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

try:
    from win10toast import ToastNotifier
    USAR_NOTIFICACION = True
except ImportError:
    USAR_NOTIFICACION = False
    print("⚠ win10toast no está instalado. Las notificaciones están desactivadas.")

from trayectoria_comun import (cargar_ptv_completo, clasificar_zona_y_etapa,
                               zona_modal, resumen_fragmentacion)
from definir_zonas import asignar_zona
from carga_real import cargar_etapas

# ============================================================
# CONFIGURACION — editar aqui
# ============================================================

CACHE_ZONAS = "cache_zonas"          # .npz de PIV (con u,v,frame_idx)
PTV_DIR = "."                        # carpeta raiz con subcarpetas <ensayo>/ptv_merged.json
ETAPAS_JSON = "etapas_zonas.json"

SALIDA_CSV = "atribucion_E1_E2_E3.csv"
SALIDA_CONCORDANCIA = "resumen_concordancia.csv"

MIN_FRAMES_E1 = 3     # frames minimos totales para considerar el track "estable"

# Parametros de stitching (E2) -- version simplificada del repo de referencia
GAP_T_MAX = 0.30      # ventana temporal maxima para enlazar segmentos [s]
GAP_R_MAX = 8.0       # radio espacial maximo tras prediccion por velocidad [mm]

# Adveccion (E3)
METODO_ADVECCION = "rk2"
SUBMUESTREO = 1

# Límite del relleno por vecino más cercano en E3. LinearNDInterpolator
# devuelve NaN fuera de la envolvente convexa del PIV del frame (donde NO hay
# material medido). Rellenar sin límite con NearestNDInterpolator equivale a
# extrapolar arbitrariamente lejos, y en un integrador ese error se compone:
# la posición se actualiza con una velocidad inventada y no hay forma de
# recuperar la trayectoria después. Ver la misma discusión y el mismo fix en
# construir_caches_adveccion.py (DIST_MAX_NN_MM).
DIST_MAX_NN_MM = 3.0


# ============================================================
# Emparejamiento PIV <-> carpeta PTV completa, por codigo mNN
# ============================================================
def _codigo_toma(nombre):
    m = re.search(r"m(\d+)", os.path.basename(str(nombre)).lower())
    return f"m{m.group(1)}" if m else None


def emparejar_piv_ptv():
    npzs = {_codigo_toma(f): f
            for f in glob.glob(os.path.join(CACHE_ZONAS, "*.npz"))}
    carpetas_ptv = {}
    for root, dirs, files in os.walk(PTV_DIR):
        if "ptv_merged.json" in files:
            cod = _codigo_toma(os.path.basename(root))
            if cod:
                carpetas_ptv[cod] = root
    comunes = sorted(set(npzs) & set(carpetas_ptv))
    pares = [{"cod": c, "npz": npzs[c], "carpeta_ptv": carpetas_ptv[c]}
             for c in comunes]
    faltan_piv = sorted(set(carpetas_ptv) - set(npzs))
    faltan_ptv = sorted(set(npzs) - set(carpetas_ptv))
    if faltan_piv:
        print(f"[aviso] PTV sin PIV: {faltan_piv}")
    if faltan_ptv:
        print(f"[aviso] PIV sin carpeta PTV completa: {faltan_ptv}")
    return pares


# ============================================================
# E1 -- zona modal directa del track real
# ============================================================
def calcular_E1(ptv_clasificado):
    """Por track_id: zona modal durante 'transicion', y n_frames totales."""
    filas = []
    for tid, g in ptv_clasificado.groupby("track_id"):
        n_frames = g["frame"].nunique()
        zona_trans = zona_modal(g, "transicion")
        zona_final = g.sort_values("frame")["zona"].iloc[-1]
        filas.append({"track_id": tid, "n_frames": n_frames,
                      "zona_transicion_E1": zona_trans,
                      "zona_final": zona_final})
    return pd.DataFrame(filas)


# ============================================================
# E2 -- stitching simplificado (gap-closing) + zona modal
# ============================================================
def _velocidad_track(g):
    """
    Velocidad al FINAL del segmento, para predecir donde continua tras un gap.
    Usa vx_mm_s/vy_mm_s YA MEDIDOS por el algoritmo de PTV en el ultimo frame
    del segmento (mas preciso que estimar por diferencias finitas sobre todo
    el segmento). Si vinieran NaN, cae a la diferencia finita como respaldo.
    """
    g = g.sort_values("frame")
    vx, vy = g["vx_mm_s"].iloc[-1], g["vy_mm_s"].iloc[-1]
    if pd.notna(vx) and pd.notna(vy):
        return float(vx), float(vy)
    if len(g) < 2:
        return 0.0, 0.0
    dt = g["t"].iloc[-1] - g["t"].iloc[0]
    if dt <= 0:
        return 0.0, 0.0
    vx = (g["x_mm"].iloc[-1] - g["x_mm"].iloc[0]) / dt
    vy = (g["y_mm"].iloc[-1] - g["y_mm"].iloc[0]) / dt
    return vx, vy


def stitch_tracks(ptv_clasificado, gap_t_max=GAP_T_MAX, gap_r_max=GAP_R_MAX):
    """
    Re-enlaza segmentos de track que terminan y otro que empieza cerca en
    espacio (tras predecir la posicion con la velocidad del segmento que
    termina) y tiempo. Simplificacion del E2 del repo de referencia: no se
    chequea continuidad rotacional (omega), solo posicion+tiempo.

    Devuelve el DataFrame con una columna nueva 'track_id_stitched' que
    agrupa los track_id originales que se consideran la misma fibra fisica.
    """
    df = ptv_clasificado.copy()
    segmentos = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        segmentos.append({
            "tid": tid, "t_ini": g["t"].iloc[0], "t_fin": g["t"].iloc[-1],
            "x_ini": g["x_mm"].iloc[0], "y_ini": g["y_mm"].iloc[0],
            "x_fin": g["x_mm"].iloc[-1], "y_fin": g["y_mm"].iloc[-1],
            "vx": _velocidad_track(g)[0], "vy": _velocidad_track(g)[1],
        })
    seg = pd.DataFrame(segmentos).sort_values("t_ini").reset_index(drop=True)

    padre = {s: s for s in seg["tid"]}

    def find(a):
        while padre[a] != a:
            a = padre[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            padre[ra] = rb

    # para cada segmento que TERMINA, buscar el que EMPIEZA mas cerca despues
    for i, s in seg.iterrows():
        candidatos = seg[(seg["t_ini"] > s["t_fin"]) &
                        (seg["t_ini"] <= s["t_fin"] + gap_t_max)]
        if candidatos.empty:
            continue
        dt_pred = candidatos["t_ini"] - s["t_fin"]
        x_pred = s["x_fin"] + s["vx"] * dt_pred
        y_pred = s["y_fin"] + s["vy"] * dt_pred
        dist = np.hypot(candidatos["x_ini"] - x_pred,
                        candidatos["y_ini"] - y_pred)
        if (dist <= gap_r_max).any():
            j = dist.idxmin()
            union(s["tid"], seg.loc[j, "tid"])

    df["track_id_stitched"] = df["track_id"].map(lambda t: find(t))
    return df


def calcular_E2(ptv_clasificado):
    """Igual que E1 pero sobre los tracks re-enlazados (stitched)."""
    stitched = stitch_tracks(ptv_clasificado)
    filas = []
    for tid_orig, g in stitched.groupby("track_id"):
        tid_stitch = g["track_id_stitched"].iloc[0]
        grupo_completo = stitched[stitched["track_id_stitched"] == tid_stitch]
        zona_trans = zona_modal(grupo_completo, "transicion")
        filas.append({"track_id": tid_orig, "zona_transicion_E2": zona_trans})
    return pd.DataFrame(filas)


# ============================================================
# E3 -- pathline PIV (adveccion hacia atras) + zona modal por regimen
# ============================================================
def cargar_campo_piv_e3(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({
        "x": z["x"].astype(float), "y": z["y"].astype(float),
        "t": z["t"].astype(float), "frame": z["frame_idx"].astype(int),
        "u": z["u"].astype(float), "v": z["v"].astype(float),
    })
    frames = np.sort(df["frame"].unique())
    t_de_frame = df.groupby("frame")["t"].first().to_dict()
    por_frame = {f: sub[["x", "y", "u", "v"]].reset_index(drop=True)
                 for f, sub in df.groupby("frame")}
    ts = np.array([t_de_frame[f] for f in frames])
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 30
    return {"frames": frames, "t_de_frame": t_de_frame,
            "por_frame": por_frame, "dt": dt}


def _construir_interpoladores(sub):
    """
    Construye la triangulacion de Delaunay UNA SOLA VEZ por frame, e
    interpola u y v JUNTOS (LinearNDInterpolator acepta valores vectoriales).

    Antes se llamaba griddata por separado para 'u' y 'v', lo que triangulaba
    los MISMOS puntos (x,y) dos veces por evaluacion -- con RK2 (posicion +
    punto medio) eso eran 4 triangulaciones por frame en vez de 1. Con
    cientos de frames y varias tomas, esto es el cuello de botella principal
    del tiempo de ejecucion. Ahora: 1 triangulacion por frame, reutilizada
    para todas las evaluaciones de ese frame.

    Se construye también un cKDTree sobre los mismos puntos: permite acotar
    el respaldo por vecino más cercano a DIST_MAX_NN_MM (ver _interp_uv_rapido).
    """
    xy = sub[["x", "y"]].to_numpy()
    uv = sub[["u", "v"]].to_numpy()
    lin = LinearNDInterpolator(xy, uv)
    nn = NearestNDInterpolator(xy, uv)   # respaldo fuera del casco convexo
    tree = cKDTree(xy)
    return lin, nn, tree


def _interp_uv_rapido(lin, nn, tree, puntos, dist_max=DIST_MAX_NN_MM):
    """Evalua los interpoladores YA CONSTRUIDOS (ver _construir_interpoladores).

    Fuera de la envolvente convexa se admite el respaldo por vecino más
    cercano SOLO si el punto está a menos de `dist_max` mm de un vector PIV
    real. Más allá, el valor queda NaN: es una posición fuera del material
    medido, no un hueco de correlación a rellenar.

    Devuelve (dict campo -> array, mask_valido).
    """
    val = np.asarray(lin(puntos), dtype=float)
    nanmask = np.isnan(val).any(axis=1)
    if nanmask.any():
        cand = puntos[nanmask]
        dist, _ = tree.query(cand, k=1)
        cerca = dist <= dist_max
        relleno = np.full((len(cand), val.shape[1]), np.nan)
        if cerca.any():
            relleno[cerca] = nn(cand[cerca])
        val[nanmask] = relleno
    valido = ~np.isnan(val).any(axis=1)
    return {"u": val[:, 0], "v": val[:, 1]}, valido


def _fmt_seg(s):
    """Formatea segundos a Xm YYs o Xh YYm ZZs para progreso legible."""
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def calcular_E3(campo, fib_finales, cortes_etapa, metodo=METODO_ADVECCION,
                submuestreo=SUBMUESTREO, verbose=True):
    """
    Integra hacia atras la posicion final de cada fibra sobre el campo PIV,
    clasifica zona+regimen en cada paso, y devuelve la zona MODAL durante
    transicion y la zona de ORIGEN (punto mas temprano alcanzado).

    fib_finales: DataFrame con columnas track_id, x_mm, y_mm (posicion final).
    cortes_etapa: dict {zona: t_quasi} (mismo formato que cargar_etapas).

    Optimizaciones respecto a la version anterior:
      - 1 sola triangulacion de Delaunay por frame (antes se repetia para u
        y v por separado, y de nuevo en el punto medio de RK2: hasta 4x mas
        trabajo del necesario).
      - Progreso con tiempo transcurrido y ETA (antes corria en silencio).
      - Agregacion final por groupby en vez de filtrar el DataFrame completo
        una vez POR FIBRA (evita un costo cuadratico con muchas fibras).
    """
    frames = campo["frames"][::-1][::submuestreo]
    dt = campo["dt"] * submuestreo
    pos = fib_finales[["x_mm", "y_mm"]].to_numpy(dtype=float).copy()
    F = len(pos)
    n = len(frames)
    intervalo = max(1, n // 20)   # progreso cada ~5%

    # Una fibra "viva" es aquella cuya trayectoria retrocedida sigue dentro
    # del material medido. Al salir (interpolación no fiable más allá de
    # DIST_MAX_NN_MM) se marca perdida y deja de aportar registros: seguir
    # integrando con una velocidad extrapolada arbitrariamente lejos
    # contaminaría el resto de su historia hacia atrás.
    vivo = np.ones(F, dtype=bool)

    registros = []
    t0 = time.time()
    for i, f in enumerate(frames):
        sub = campo["por_frame"][f]
        t_f = campo["t_de_frame"][f]
        if len(sub) < 4:
            continue

        lin, nn, tree = _construir_interpoladores(sub)   # 1 triangulacion, no 2-4
        uv, ok = _interp_uv_rapido(lin, nn, tree, pos)
        vivo &= ok
        zonas = asignar_zona(pos[:, 0], pos[:, 1])

        validos = vivo & (zonas != None) & (zonas != "fuera")  # noqa: E711
        idx_validos = np.nonzero(validos)[0]
        for k in idx_validos:
            z = zonas[k]
            corte = cortes_etapa.get(z)
            regimen = ("transicion" if (corte is not None and t_f <= corte)
                      else "cuasi" if corte is not None else "n/a")
            registros.append((k, z, regimen))

        if not vivo.any():
            if verbose:
                print(f"      ⚠ todas las fibras salieron del material en "
                      f"el frame {i+1}/{n}; se trunca E3 aquí.", flush=True)
            break

        if i < n - 1:
            if metodo == "euler":
                pos[vivo, 0] -= uv["u"][vivo] * dt
                pos[vivo, 1] -= uv["v"][vivo] * dt
            else:
                xm = pos[:, 0] - 0.5 * uv["u"] * dt
                ym = pos[:, 1] - 0.5 * uv["v"] * dt
                mid, ok_mid = _interp_uv_rapido(lin, nn, tree,
                                                np.column_stack([xm, ym]))
                # Punto medio del RK2 fuera del radio de confianza -> el paso
                # no es fiable; la fibra se da por perdida en vez de degradar
                # a Euler silenciosamente.
                vivo &= ok_mid
                pos[vivo, 0] -= mid["u"][vivo] * dt
                pos[vivo, 1] -= mid["v"][vivo] * dt

        if verbose and ((i + 1) % intervalo == 0 or (i + 1) == n):
            transcurrido = time.time() - t0
            restante = transcurrido / (i + 1) * (n - (i + 1))
            print(f"      E3 {100*(i+1)/n:5.1f}%  frame {i+1}/{n}  "
                  f"transcurrido={_fmt_seg(transcurrido)}  "
                  f"restante≈{_fmt_seg(restante)}", flush=True)

    if verbose:
        n_vivas = int(vivo.sum())
        print(f"      E3: fibras con trayectoria completa hasta el primer "
              f"frame disponible: {n_vivas}/{F} ({100*n_vivas/F:.0f}%)",
              flush=True)

    reg_df = pd.DataFrame(registros, columns=["idx", "zona", "regimen"])

    # agregacion vectorizada (groupby) en vez de un filtro completo por fibra
    def _moda(s):
        m = s.mode()
        return m.iloc[0] if not m.empty else None

    zona_trans_por_idx = (reg_df[reg_df.regimen == "transicion"]
                          .groupby("idx")["zona"].agg(_moda))
    zona_origen_por_idx = reg_df.groupby("idx")["zona"].last()

    out = []
    for k in range(F):
        out.append({"track_id": fib_finales["track_id"].iloc[k],
                    "zona_transicion_E3": zona_trans_por_idx.get(k),
                    "zona_origen_E3": zona_origen_por_idx.get(k)})
    return pd.DataFrame(out)


# ============================================================
# Concordancia entre metodos
# ============================================================
def calcular_concordancia(tabla):
    """
    Tasa de acuerdo par-a-par entre E1, E2, E3 (solo sobre tracks donde las
    tres atribuciones existen). Devuelve dict con las 3 tasas + n comparado.
    """
    cols = ["zona_transicion_E1", "zona_transicion_E2", "zona_transicion_E3"]
    t = tabla.dropna(subset=cols)
    n = len(t)
    if n == 0:
        return {"n_comparado": 0, "acuerdo_E1_E2": np.nan,
                "acuerdo_E1_E3": np.nan, "acuerdo_E2_E3": np.nan}
    return {
        "n_comparado": n,
        "acuerdo_E1_E2": float((t.zona_transicion_E1 == t.zona_transicion_E2).mean()),
        "acuerdo_E1_E3": float((t.zona_transicion_E1 == t.zona_transicion_E3).mean()),
        "acuerdo_E2_E3": float((t.zona_transicion_E2 == t.zona_transicion_E3).mean()),
    }


# ============================================================
# MAIN
# ============================================================
def procesar_toma(par):
    cod = par["cod"]
    print(f"\n{'='*55}\n  {cod}")
    t0_toma = time.time()

    ptv = cargar_ptv_completo(par["carpeta_ptv"])
    if ptv.empty:
        print("  ⚠ sin datos PTV")
        return None
    frag = resumen_fragmentacion(ptv)
    print(f"  PTV: {frag['n_tracks']} tracks, frames/track "
          f"mediana={frag['frames_mediana']:.1f} "
          f"(min={frag['frames_min']}, max={frag['frames_max']})")

    ptv_c = clasificar_zona_y_etapa(ptv, ETAPAS_JSON, cod)

    t0 = time.time()
    e1 = calcular_E1(ptv_c)
    e1 = e1[e1["n_frames"] >= MIN_FRAMES_E1]
    print(f"  E1 listo en {_fmt_seg(time.time()-t0)}")

    t0 = time.time()
    e2 = calcular_E2(ptv_c)
    print(f"  E2 listo en {_fmt_seg(time.time()-t0)}")

    t0 = time.time()
    campo = cargar_campo_piv_e3(par["npz"])
    print(f"  campo PIV cargado en {_fmt_seg(time.time()-t0)} "
          f"({len(campo['frames'])} frames)")
    zonas_presentes = sorted(z for z in ptv_c["zona"].unique()
                             if z and z != "fuera")
    cortes = cargar_etapas(ETAPAS_JSON, df_piv=None, zonas=zonas_presentes,
                           toma=cod)
    fib_finales = (ptv_c.sort_values("frame")
                   .groupby("track_id").last().reset_index()
                   [["track_id", "x_mm", "y_mm"]])

    t0 = time.time()
    e3 = calcular_E3(campo, fib_finales, cortes)
    print(f"  E3 (adveccion) listo en {_fmt_seg(time.time()-t0)}")

    tabla = e1.merge(e2, on="track_id", how="left") \
              .merge(e3, on="track_id", how="left")
    tabla.insert(0, "toma", cod)

    conc = calcular_concordancia(tabla)
    conc["toma"] = cod
    print(f"  Concordancia (n={conc['n_comparado']}): "
          f"E1-E2={conc['acuerdo_E1_E2']:.2f}  "
          f"E1-E3={conc['acuerdo_E1_E3']:.2f}  "
          f"E2-E3={conc['acuerdo_E2_E3']:.2f}"
          if conc["n_comparado"] > 0 else "  Concordancia: sin datos suficientes")
    print(f"  → toma {cod} completa en {_fmt_seg(time.time()-t0_toma)}")
    return tabla, conc


if __name__ == "__main__":
    pares = emparejar_piv_ptv()
    print(f"Tomas emparejadas (PIV + PTV completo): {len(pares)}")
    if not pares:
        print(f"[ERROR] Revisa que existan .npz en '{CACHE_ZONAS}/' y "
              f"carpetas con ptv_merged.json bajo '{PTV_DIR}'.")
        raise SystemExit

    tablas, concs = [], []
    for par in pares:
        res = procesar_toma(par)
        if res is not None:
            tablas.append(res[0])
            concs.append(res[1])

    if tablas:
        pd.concat(tablas, ignore_index=True).to_csv(SALIDA_CSV, index=False)
        pd.DataFrame(concs).to_csv(SALIDA_CONCORDANCIA, index=False)
        print(f"\n{'='*55}\nGuardado: {SALIDA_CSV}, {SALIDA_CONCORDANCIA}")
    toaster = ToastNotifier()
    toaster.show_toast("VSCode", "¡Tu código de Python terminó exitosamente!", duration=5)