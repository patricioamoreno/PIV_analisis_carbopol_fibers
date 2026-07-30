"""
adveccion_nucleo.py
====================
Nucleo compartido de integracion lagrangiana hacia atras sobre el campo PIV
(interpolacion + retroceso en el tiempo). Antes esta fisica vivia por
duplicado, con el MISMO NOMBRE de funcion pero firma distinta, en dos
archivos:

  - construir_caches_adveccion.py  (Analisis 5: TODAS las fibras,
    interpola u,v,V,omega,gamma_dot -- necesita el historial fisico
    completo para correlacionar contra la orientacion)
  - reconstruccion_lagrangiana.py  (E3: SOLO el subconjunto de fibras con
    track PTV suficientemente largo, interpola solo u,v -- solo necesita
    clasificar zona, para validar el metodo contra E1)

Tener dos copias es un riesgo real, no solo prolijidad: un fix aplicado a
una copia (por ejemplo, el bug de coordenadas NaN propagandose a
cKDTree.query en el paso medio del RK2 para particulas ya "muertas") puede
no propagarse a la otra, y ambas dejan de calcular exactamente lo mismo
sin que nadie lo note -- justo lo que la validacion E1/E3 pretende
descartar.

Ambos scripts importan de aqui.
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

DIST_MAX_NN_MM_DEFAULT = 3.0


def construir_interpoladores(sub, campos=("u", "v")):
    """
    Triangulacion de Delaunay UNA sola vez por frame, interpolando todas
    las columnas de 'campos' juntas (LinearNDInterpolator acepta valores
    vectoriales) -- evita repetir la triangulacion una vez por columna y
    de nuevo en el punto medio del RK2.

    sub: DataFrame de un frame con columnas 'x','y' + las de 'campos'.
    Devuelve (lin, nn, tree) para ese frame.
    """
    xy = sub[["x", "y"]].to_numpy()
    vals = sub[list(campos)].to_numpy()
    lin = LinearNDInterpolator(xy, vals)
    nn = NearestNDInterpolator(xy, vals)     # respaldo fuera del casco convexo
    tree = cKDTree(xy)                        # acota el respaldo a dist_max
    return lin, nn, tree


def interp_campos(interp_frame, campos, puntos, dist_max=DIST_MAX_NN_MM_DEFAULT):
    """
    Evalua los interpoladores YA CONSTRUIDOS (ver construir_interpoladores)
    en 'puntos'.

    GUARDIA DE COORDENADAS NO FINITAS: los puntos con x o y no finitos
    (NaN/inf) nunca se le pasan a lin() ni a cKDTree.query, que revienta
    con ValueError ante un input no finito. Una coordenada no finita solo
    puede venir de una particula ya perdida en un paso anterior de la
    integracion hacia atras (velocidad NaN -> posicion NaN al restar);
    se marca invalida de inmediato, sin intentar interpolar ni buscar
    vecino.

    Fuera de la envolvente convexa se admite el respaldo por vecino mas
    cercano SOLO si el punto esta a menos de dist_max mm de un vector PIV
    real; mas alla, el valor queda NaN -- posicion fuera del material
    medido, no un hueco de correlacion a rellenar.

    Devuelve (dict campo -> array, mask_valido).
    """
    lin, nn, tree = interp_frame
    campos = list(campos)
    puntos = np.asarray(puntos, dtype=float)

    finito = np.all(np.isfinite(puntos), axis=1)

    M = np.full((len(puntos), len(campos)), np.nan, dtype=float)
    if finito.any():
        ev = np.asarray(lin(puntos[finito]), dtype=float)
        if ev.ndim == 1:          # un solo campo: LinearNDInterpolator aplana
            ev = ev.reshape(-1, 1)
        M[finito] = ev

    nanrows = finito & np.isnan(M).any(axis=1)
    if nanrows.any():
        cand = puntos[nanrows]
        dist, _ = tree.query(cand, k=1)
        cerca = dist <= dist_max
        relleno = np.full((len(cand), M.shape[1]), np.nan)
        if cerca.any():
            ev2 = np.asarray(nn(cand[cerca]), dtype=float)
            if ev2.ndim == 1:
                ev2 = ev2.reshape(-1, 1)
            relleno[cerca] = ev2
        M[nanrows] = relleno

    valido = finito & ~np.isnan(M).any(axis=1)
    return {c: M[:, j] for j, c in enumerate(campos)}, valido
