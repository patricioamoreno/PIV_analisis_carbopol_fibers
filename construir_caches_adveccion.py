"""
construir_caches_adveccion.py
=============================
COMPLEMENTO al analisis de zona final. Reconstruye la TRAYECTORIA PROBABLE de
cada fibra retrocediendo en el tiempo sobre el campo (u,v) del PIV (adveccion
hacia atras), y guarda un cache .npz por toma con la influencia que cada zona
ejercio sobre cada fibra (cuanto tiempo estuvo y bajo que V, omega, gamma_dot).

Por que existe (reunion 10-07): el analisis de zona final asocia cada fibra al
fluido de la zona donde QUEDO. Esto no dice si la fibra se oriento ahi o solo
llego ahi. Con el campo (u,v) resuelto en tiempo se estima por donde paso cada
fibra y que flujo experimento en su recorrido.

SUPUESTO (declararlo en la memoria): la fibra se advecta como particula pasiva
del fluido (se mueve con la velocidad local). Las fibras reales tienen inercia
y tamano, asi que la trayectoria es una ESTIMACION, no la verdad exacta.

Mismo estilo que construir_caches_zonas.py: config arriba, un .npz por toma,
np.savez_compressed, bandera RECALCULO, loader gemelo cargar_cache_adveccion().

Fuentes:
  - CAMPO PIV (u,v,frame_idx por punto): cache_zonas/<carpeta>_zonas.npz.
  - FIBRAS (foto final): fibras_ultimo_frame/<...>-ptv.csv (CSV sueltos).

Salida .npz (una fila por (fibra, zona) atravesada):
    fibra_id, zona, n_pasos, t_en_zona, frac_tiempo,
    V_med, omega_med, gamma_dot_med, theta   (+ meta)

Uso:
    python construir_caches_adveccion.py
"""

import os
import re
import glob
import time
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from win10toast import ToastNotifier

# ============================================================
# CONFIGURACION — editar aqui
# ============================================================

CACHE_ZONAS = "cache_zonas"
FIBRAS_DIR = "fibras_ultimo_frame"
CACHE_ADV_DIR = "cache_adveccion"

RECALCULO = True

CAR_OBJETIVO = None
FIB_OBJETIVO = None

METODO = "rk2"           # "euler" o "rk2"
SUBMUESTREO = 1          # 1 de cada k frames
N_PASOS = None           # nº max de pasos (None = todos)

PREDICTORES = ["V", "omega", "gamma_dot"]


# ============================================================
# UTILIDADES
# ============================================================

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', s)]


def _codigo_toma(nombre):
    m = re.search(r"m(\d+)", os.path.basename(str(nombre)).lower())
    return f"m{m.group(1)}" if m else None


def _meta_toma(nombre):
    car = re.search(r"car-?(\d+)", os.path.basename(str(nombre)).lower())
    fib = re.search(r"n-?(\d+)", os.path.basename(str(nombre)).lower())
    return (f"car-{car.group(1)}" if car else None,
            int(fib.group(1)) if fib else None)


def _nombre_cache(cod):
    return f"{cod}__adveccion.npz"


def cache_existe(cod, cache_dir=CACHE_ADV_DIR):
    return os.path.exists(os.path.join(cache_dir, _nombre_cache(cod)))


def cargar_cache_adveccion(cod, cache_dir=CACHE_ADV_DIR):
    """Loader gemelo: influencia por (fibra,zona) como DataFrame, o None."""
    path = os.path.join(cache_dir, _nombre_cache(cod))
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return pd.DataFrame({
        'fibra_id': d['fibra_id'], 'zona': d['zona'],
        'n_pasos': d['n_pasos'], 't_en_zona': d['t_en_zona'],
        'frac_tiempo': d['frac_tiempo'], 'V_med': d['V_med'],
        'omega_med': d['omega_med'], 'gamma_dot_med': d['gamma_dot_med'],
        'theta': d['theta'],
    })


def emparejar_piv_fibras():
    npzs = {}
    for f in glob.glob(os.path.join(CACHE_ZONAS, "*.npz")):
        cod = _codigo_toma(f)
        if cod:
            npzs[cod] = f
    csvs = {}
    for f in glob.glob(os.path.join(FIBRAS_DIR, "*.csv")):
        if os.path.basename(f).startswith("_"):
            continue
        cod = _codigo_toma(f)
        if cod:
            csvs[cod] = f
    comunes = sorted(set(npzs) & set(csvs), key=natural_sort_key)
    pares = []
    for cod in comunes:
        reo, fib = _meta_toma(csvs[cod])
        if CAR_OBJETIVO and reo != f"car-{CAR_OBJETIVO}":
            continue
        if FIB_OBJETIVO and fib != int(FIB_OBJETIVO):
            continue
        pares.append({"cod": cod, "npz": npzs[cod], "csv": csvs[cod]})
    return pares


# ============================================================
# CARGA DEL CAMPO PIV (del cache_zonas, con u,v por frame)
# ============================================================

def cargar_campo_piv(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    faltan = [k for k in ['x', 'y', 't', 'frame_idx', 'u', 'v', 'v_mag',
                          'gamma_dot', 'vort', 'zona'] if k not in z.files]
    if faltan:
        raise KeyError(f"faltan campos {faltan}. Tiene: {sorted(z.files)}")

    df = pd.DataFrame({
        "x": z["x"].astype(float), "y": z["y"].astype(float),
        "t": z["t"].astype(float), "frame": z["frame_idx"].astype(int),
        "u": z["u"].astype(float), "v": z["v"].astype(float),
        "V": z["v_mag"].astype(float),
        "omega": np.abs(z["vort"].astype(float)),
        "gamma_dot": z["gamma_dot"].astype(float),
        "zona": z["zona"].astype(str),
    })
    boxes = {zz: (g.x.min(), g.x.max(), g.y.min(), g.y.max())
             for zz, g in df.groupby("zona") if zz != "fuera"}
    frames = np.sort(df["frame"].unique())
    t_de_frame = df.groupby("frame")["t"].first().to_dict()
    por_frame = {f: sub[["x", "y", "u", "v", "V", "omega", "gamma_dot"]]
                    .reset_index(drop=True)
                 for f, sub in df.groupby("frame")}
    ts = np.array([t_de_frame[f] for f in frames])
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 30
    return {"frames": frames, "t_de_frame": t_de_frame,
            "por_frame": por_frame, "boxes": boxes, "dt": dt}


def cargar_fibras(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"angle_deg": "theta"})
    df["fibra_id"] = np.arange(len(df))
    return df[["fibra_id", "x_mm", "y_mm", "theta"]]


# ============================================================
# INTERPOLACION Y ADVECCION
# ============================================================

def _construir_interpoladores(por_frame):
    """
    Construye UNA vez el interpolador de cada frame (triangulacion de Delaunay)
    y lo cachea. griddata retriangulaba en cada llamada; esto lo hace una sola
    vez por frame y acelera el proceso varias veces.
    Devuelve dict frame -> (LinearNDInterpolator, NearestNDInterpolator) sobre
    las columnas [u, v, V, omega, gamma_dot] juntas.
    """
    campos = ["u", "v", "V", "omega", "gamma_dot"]
    interp = {}
    for f, sub in por_frame.items():
        xy = sub[["x", "y"]].to_numpy()
        vals = sub[campos].to_numpy()
        # un solo interpolador multi-columna (comparte la triangulacion)
        lin = LinearNDInterpolator(xy, vals)
        nn = NearestNDInterpolator(xy, vals)
        interp[f] = (lin, nn)
    return interp, campos


def _interp_en(interp_frame, campos, puntos):
    """Interpola todos los campos a la vez en 'puntos' usando el interpolador
    ya construido del frame. Rellena huecos con vecino mas cercano."""
    lin, nn = interp_frame
    M = lin(puntos)                       # (P x 5)
    nanrows = np.isnan(M).any(axis=1)
    if nanrows.any():
        M[nanrows] = nn(puntos[nanrows])
    return {c: M[:, j] for j, c in enumerate(campos)}


def _zona_de_puntos(puntos, boxes):
    areas = {z: (x1 - x0) * (y1 - y0) for z, (x0, x1, y0, y1) in boxes.items()}
    zonas = []
    for px, py in puntos:
        cand = [z for z, (x0, x1, y0, y1) in boxes.items()
                if x0 <= px <= x1 and y0 <= py <= y1]
        zonas.append(min(cand, key=lambda z: areas[z]) if cand else "fuera")
    return np.array(zonas, dtype=object)


def _retroceder(campo, fibras_xy, cod=""):
    frames = campo["frames"]
    dt = campo["dt"] * SUBMUESTREO
    orden = frames[::-1][::SUBMUESTREO]
    if N_PASOS is not None:
        orden = orden[:N_PASOS + 1]

    print(f"    construyendo interpoladores de {len(orden)} frames...",
          flush=True)
    interp, campos = _construir_interpoladores(
        {f: campo["por_frame"][f] for f in orden})

    F = len(fibras_xy)
    pos = fibras_xy.astype(float).copy()
    reg = []
    n = len(orden)
    print(f"    advectando {F} fibras por {n} pasos...", flush=True)
    t0 = time.time()
    # aviso cada 5% (o cada paso si hay pocos) para que se note el avance
    intervalo = max(1, n // 20)
    for i, f in enumerate(orden):
        camp = _interp_en(interp[f], campos, pos)
        zonas = _zona_de_puntos(pos, campo["boxes"])
        for k in range(F):
            reg.append((k, int(f), camp["V"][k], camp["omega"][k],
                        camp["gamma_dot"][k], zonas[k]))
        if i < n - 1:
            if METODO == "euler":
                pos[:, 0] -= camp["u"] * dt
                pos[:, 1] -= camp["v"] * dt
            else:
                xm = pos[:, 0] - 0.5 * camp["u"] * dt
                ym = pos[:, 1] - 0.5 * camp["v"] * dt
                mid = _interp_en(interp[f], campos, np.column_stack([xm, ym]))
                pos[:, 0] -= mid["u"] * dt
                pos[:, 1] -= mid["v"] * dt
        # progreso: cada 'intervalo' pasos, con tiempo transcurrido y ETA
        if (i + 1) % intervalo == 0 or (i + 1) == n:
            transcurrido = time.time() - t0
            ritmo = transcurrido / (i + 1)          # segundos por paso
            restante = ritmo * (n - (i + 1))
            print(f"      {100*(i+1)/n:5.1f}%  paso {i+1}/{n}  "
                  f"transcurrido={_fmt_seg(transcurrido)}  "
                  f"restante≈{_fmt_seg(restante)}", flush=True)
    cols = ["fibra_id", "frame", "V", "omega", "gamma_dot", "zona"]
    return pd.DataFrame.from_records(reg, columns=cols), dt


def _fmt_seg(s):
    """Formatea segundos a m:ss o h:mm:ss para que sea legible."""
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _influencia(traj, dt):
    t = traj[traj.zona != "fuera"].copy()
    agg = (t.groupby(["fibra_id", "zona"])
             .agg(n_pasos=("frame", "size"), V_med=("V", "mean"),
                  omega_med=("omega", "mean"),
                  gamma_dot_med=("gamma_dot", "mean")).reset_index())
    agg["t_en_zona"] = agg["n_pasos"] * dt
    tot = agg.groupby("fibra_id")["n_pasos"].transform("sum")
    agg["frac_tiempo"] = agg["n_pasos"] / tot
    return agg


# ============================================================
# CONSTRUIR CACHE DE UNA TOMA
# ============================================================

def construir_cache(par):
    cod = par["cod"]
    path = os.path.join(CACHE_ADV_DIR, _nombre_cache(cod))

    if cache_existe(cod) and not RECALCULO:
        print(f"  ✓ Ya existe: {_nombre_cache(cod)}")
        return

    try:
        campo = cargar_campo_piv(par["npz"])
    except KeyError as e:
        print(f"  ⚠ {cod}: {e}")
        return
    fib = cargar_fibras(par["csv"])
    if len(fib) == 0:
        print(f"  ⚠ Sin fibras: {cod}")
        return

    reo, nfib_conc = _meta_toma(par["csv"])
    print(f"  🔨 {cod}  ({len(campo['frames'])} frames, {len(fib)} fibras)",
          flush=True)
    t0_toma = time.time()

    traj, dt = _retroceder(campo, fib[["x_mm", "y_mm"]].to_numpy())
    infl = _influencia(traj, dt)
    infl = infl.merge(fib[["fibra_id", "theta"]], on="fibra_id", how="left")

    os.makedirs(CACHE_ADV_DIR, exist_ok=True)
    np.savez_compressed(
        path,
        fibra_id=infl["fibra_id"].to_numpy(),
        zona=infl["zona"].to_numpy().astype("<U12"),
        n_pasos=infl["n_pasos"].to_numpy(),
        t_en_zona=infl["t_en_zona"].to_numpy(),
        frac_tiempo=infl["frac_tiempo"].to_numpy(),
        V_med=infl["V_med"].to_numpy(),
        omega_med=infl["omega_med"].to_numpy(),
        gamma_dot_med=infl["gamma_dot_med"].to_numpy(),
        theta=infl["theta"].to_numpy(),
        toma=cod, n_fibras=len(fib),
        reologia=(reo or ""), conc_fibras=(nfib_conc or 0),
        submuestreo=SUBMUESTREO, metodo=METODO,
    )
    print(f"  💾 Guardado: {_nombre_cache(cod)}  "
          f"({len(infl)} filas fibra×zona)  "
          f"— toma completa en {_fmt_seg(time.time()-t0_toma)}", flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    pares = emparejar_piv_fibras()
    print(f"Tomas emparejadas: {len(pares)}\n")
    if not pares:
        print(f"[ERROR] No se emparejo nada. Revisa .npz en '{CACHE_ZONAS}/' "
              f"y CSV en '{FIBRAS_DIR}/' con codigo mNN comun.")
    t0_total = time.time()
    for i, par in enumerate(pares, 1):
        print(f"\n{'='*55}\n  [{i}/{len(pares)}]  {par['cod']}  "
              f"(transcurrido total: {_fmt_seg(time.time()-t0_total)})")
        construir_cache(par)
    if pares:
        print(f"\n{'='*55}\nTODO TERMINADO en {_fmt_seg(time.time()-t0_total)} "
              f"({len(pares)} tomas)")
    toaster = ToastNotifier()
    toaster.show_toast("VSCode", "¡Tu código de Python terminó exitosamente!", duration=5)
