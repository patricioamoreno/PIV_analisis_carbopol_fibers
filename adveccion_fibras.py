"""
adveccion_fibras.py
===================
Reconstruye la TRAYECTORIA PROBABLE de cada fibra retrocediendo en el tiempo
sobre el campo de velocidad del PIV (backward particle tracking / adveccion
hacia atras), y registra que fluido (V, omega, gamma_dot) y que ZONAS
atraveso en su camino.

Motivacion (reunion 10-07 con el profesor)
------------------------------------------
La foto final del PTV solo dice DONDE quedo cada fibra, no por donde paso. Hoy
el analisis asocia cada fibra al fluido de su zona FINAL, lo cual es debil: una
fibra pudo ser orientada mientras cruzaba otras zonas (incluida la L) y solo
acabar donde acabo. Con el campo (u,v) del PIV resuelto en tiempo (665 frames,
30 fps) se puede estimar de donde vino cada fibra y que condiciones de flujo
experimento realmente. Asi se detecta la INFLUENCIA de cada zona sobre la fibra
segun la velocidad, el giro (vorticidad) y gamma_dot que le impuso.

Metodo
------
1. Se parte de la posicion final (x,y) de la fibra en el ultimo frame.
2. En cada paso hacia atras se interpola el campo (u,v) del PIV en la posicion
   actual y en el frame actual, y se retrocede: x <- x - u*dt, y <- y - v*dt.
   (integracion de Euler hacia atras; opcion RK2 para mas precision.)
3. En cada punto de la trayectoria se interpola tambien V, omega, gamma_dot y
   se registra la zona (por bounding box). Asi cada fibra acumula el fluido que
   vivio y el tiempo que paso en cada zona.
4. Se agrega por (fibra, zona) -> cuanto tiempo estuvo y que condiciones medias
   de flujo experimento en cada zona. Eso permite medir la influencia de cada
   zona sobre la orientacion final.

SUPUESTO (declararlo en la memoria): la fibra se advecta como una particula
pasiva del fluido (se mueve con la velocidad local). Las fibras reales tienen
inercia y tamano finito, asi que la trayectoria es una ESTIMACION, no la
verdad exacta. Es el enfoque estandar y es mucho mejor que 'la zona final'.

Requisitos del PIV (arrays por-punto, uno por medicion, apilados por frame):
    x, y, t, frame_idx, u, v, v_mag, vort, gamma_dot, zona
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata, NearestNDInterpolator

PREDICTORES = ["V", "omega", "gamma_dot"]


# ----------------------------------------------------------------------
# Carga del campo PIV con series temporales (u, v por frame)
# ----------------------------------------------------------------------
def cargar_campo_piv(npz_path):
    """
    Carga el PIV como estructura por-frame para interpolar en espacio y tiempo.
    Devuelve un dict:
      frames : array de indices de frame ordenados
      t_de_frame : dict frame -> tiempo (s)
      por_frame  : dict frame -> DataFrame(x,y,u,v,V,omega,gamma_dot)
      boxes      : dict zona -> (x0,x1,y0,y1)  (bounding box real)
      dt         : paso de tiempo entre frames (s)
    """
    z = np.load(npz_path, allow_pickle=True)

    def g(*names):
        for n in names:
            if n in z.files:
                return z[n]
        raise KeyError(f"Falta {names} en {npz_path}. Claves: {sorted(z.files)}")

    df = pd.DataFrame({
        "x": g("x").astype(float),
        "y": g("y").astype(float),
        "t": g("t").astype(float),
        "frame": g("frame_idx", "frame").astype(int),
        "u": g("u").astype(float),
        "v": g("v").astype(float),
        "V": g("v_mag").astype(float),
        "omega": np.abs(g("vort").astype(float)),
        "gamma_dot": g("gamma_dot").astype(float),
        "zona": g("zona").astype(str),
    })

    # bounding boxes por zona (sobre todo el registro)
    boxes = {zz: (gg.x.min(), gg.x.max(), gg.y.min(), gg.y.max())
             for zz, gg in df.groupby("zona")}

    frames = np.sort(df["frame"].unique())
    t_de_frame = df.groupby("frame")["t"].first().to_dict()
    por_frame = {f: sub[["x", "y", "u", "v", "V", "omega", "gamma_dot"]]
                    .reset_index(drop=True)
                 for f, sub in df.groupby("frame")}

    # dt tipico
    ts = np.array([t_de_frame[f] for f in frames])
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 30

    return {"frames": frames, "t_de_frame": t_de_frame,
            "por_frame": por_frame, "boxes": boxes, "dt": dt}


# ----------------------------------------------------------------------
# Interpolacion del campo en un frame dado
# ----------------------------------------------------------------------
def _interp_frame(sub, puntos, campos):
    """
    Interpola 'campos' (lista de columnas) del DataFrame sub (malla dispersa
    de un frame) en las posiciones 'puntos' (M x 2). Usa interpolacion lineal
    y rellena huecos con vecino mas cercano. Devuelve dict campo->array(M).
    """
    xy = sub[["x", "y"]].to_numpy()
    out = {}
    for c in campos:
        vals = sub[c].to_numpy()
        lin = griddata(xy, vals, puntos, method="linear")
        # rellenar NaN (fuera del convex hull) con vecino mas cercano
        nan = np.isnan(lin)
        if nan.any():
            nn = NearestNDInterpolator(xy, vals)
            lin[nan] = nn(puntos[nan])
        out[c] = lin
    return out


def asignar_zona_puntos(puntos, boxes):
    """Zona (por bbox de menor area) de cada punto Mx2; NaN si fuera de todo."""
    areas = {z: (x1 - x0) * (y1 - y0) for z, (x0, x1, y0, y1) in boxes.items()}
    zonas = []
    for px, py in puntos:
        cand = [z for z, (x0, x1, y0, y1) in boxes.items()
                if x0 <= px <= x1 and y0 <= py <= y1]
        zonas.append(min(cand, key=lambda z: areas[z]) if cand else np.nan)
    return np.array(zonas, dtype=object)


# ----------------------------------------------------------------------
# Adveccion hacia atras de un conjunto de fibras
# ----------------------------------------------------------------------
def retroceder_fibras(campo, fibras_xy, n_pasos=None, metodo="rk2",
                      submuestreo_frames=1, verbose=True):
    """
    Retrocede TODAS las fibras a la vez sobre el campo PIV.

    campo       : salida de cargar_campo_piv.
    fibras_xy   : array (F x 2) con la posicion final de cada fibra.
    n_pasos     : cuantos frames retroceder (por defecto: todos los disponibles).
    metodo      : 'euler' (rapido) o 'rk2' (mas preciso).
    submuestreo_frames : usar 1 de cada k frames para acelerar (k>=1).

    Devuelve un DataFrame largo con una fila por (fibra, paso):
      fibra_id, paso, frame, t, x, y, u, v, V, omega, gamma_dot, zona
    que es la trayectoria reconstruida con el fluido vivido en cada punto.
    """
    frames = campo["frames"]
    dt = campo["dt"] * submuestreo_frames
    # frames de mas reciente a mas antiguo, submuestreados
    orden = frames[::-1][::submuestreo_frames]
    if n_pasos is not None:
        orden = orden[:n_pasos + 1]

    F = len(fibras_xy)
    pos = fibras_xy.astype(float).copy()
    registros = []

    for i, f in enumerate(orden):
        sub = campo["por_frame"][f]
        t = campo["t_de_frame"][f]

        # interpolar todo el campo en las posiciones actuales
        camp = _interp_frame(sub, pos, ["u", "v", "V", "omega", "gamma_dot"])
        zonas = asignar_zona_puntos(pos, campo["boxes"])

        for k in range(F):
            registros.append((k, i, int(f), float(t),
                              pos[k, 0], pos[k, 1],
                              camp["u"][k], camp["v"][k],
                              camp["V"][k], camp["omega"][k],
                              camp["gamma_dot"][k], zonas[k]))

        # retroceder un paso (no en el ultimo)
        if i < len(orden) - 1:
            if metodo == "euler":
                pos[:, 0] -= camp["u"] * dt
                pos[:, 1] -= camp["v"] * dt
            else:  # rk2 (punto medio) hacia atras
                xm = pos[:, 0] - 0.5 * camp["u"] * dt
                ym = pos[:, 1] - 0.5 * camp["v"] * dt
                mid = _interp_frame(sub, np.column_stack([xm, ym]), ["u", "v"])
                pos[:, 0] -= mid["u"] * dt
                pos[:, 1] -= mid["v"] * dt

        if verbose and i % 50 == 0:
            print(f"  paso {i}/{len(orden)-1}  (frame {f}, t={t:.2f}s)")

    cols = ["fibra_id", "paso", "frame", "t", "x", "y", "u", "v",
            "V", "omega", "gamma_dot", "zona"]
    return pd.DataFrame.from_records(registros, columns=cols)


# ----------------------------------------------------------------------
# Agregacion: influencia de cada zona sobre cada fibra
# ----------------------------------------------------------------------
def influencia_por_zona(traj, dt):
    """
    A partir de la trayectoria reconstruida, resume para cada (fibra, zona):
      - t_en_zona : tiempo total que la fibra paso en esa zona (s)
      - frac_tiempo : fraccion de su recorrido en esa zona
      - V/omega/gamma_dot medios experimentados en esa zona
    Esto cuantifica cuanto y bajo que condiciones cada zona 'trabajo' la fibra.
    """
    t = traj.dropna(subset=["zona"]).copy()
    agg = (t.groupby(["fibra_id", "zona"])
             .agg(n_pasos=("paso", "size"),
                  V_med=("V", "mean"),
                  omega_med=("omega", "mean"),
                  gamma_dot_med=("gamma_dot", "mean"))
             .reset_index())
    agg["t_en_zona"] = agg["n_pasos"] * dt
    tot = agg.groupby("fibra_id")["n_pasos"].transform("sum")
    agg["frac_tiempo"] = agg["n_pasos"] / tot
    return agg


def resumen_global_zonas(influencia):
    """
    Promedia la influencia entre todas las fibras -> tabla por zona:
      cuanto tiempo (en promedio) una fibra pasa en esa zona y bajo que flujo.
    Da una vision de que zonas son las que mas 'exponen' a las fibras.
    """
    return (influencia.groupby("zona")
            .agg(fibras=("fibra_id", "nunique"),
                 frac_tiempo_med=("frac_tiempo", "mean"),
                 V_med=("V_med", "mean"),
                 omega_med=("omega_med", "mean"),
                 gamma_dot_med=("gamma_dot_med", "mean"))
            .reset_index()
            .sort_values("frac_tiempo_med", ascending=False))
