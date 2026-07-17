"""
indice_dispersion_acotado.py
=============================
Indice de dispersion espacial ACOTADO EN [0,1] para centroides de fibras,
basado en el estadistico de vecino mas cercano de Clark & Evans (1954),
ampliamente usado en ecologia espacial y adoptado en literatura de FRC/UHPC
para cuantificar uniformidad de distribucion de fibras.

Definicion
----------
    R = r_obs_medio / r_esperado_bajo_CSR
    r_esperado = 1 / (2 * sqrt(densidad))

    R = 1   -> aleatoriedad espacial completa (CSR, referencia neutra)
    R -> 0  -> agrupamiento maximo (clumping)
    R_max = 2.1491 -> empaquetamiento hexagonal perfecto (maxima uniformidad)

Indice acotado (1 = uniforme perfecto, 0 = maximo agrupamiento):
    Iu = min(R / R_max, 1)

Referencia: Clark, P.J. & Evans, F.C. (1954). "Distance to Nearest Neighbor
as a Measure of Spatial Relationships in Populations." Ecology, 35(4).
"""

import numpy as np
from scipy.spatial import cKDTree

R_MAX_HEXAGONAL = 2.1491


def indice_dispersion_clark_evans(x, y, area=None, bbox=None):
    """
    Calcula el indice de dispersion acotado [0,1] de un conjunto de
    centroides de fibras (x, y) en una zona.

    Parametros
    ----------
    x, y : arrays de coordenadas de los centroides [mm].
    area : area de la zona [mm^2]. Si None, se estima con bbox.
    bbox : (x0,x1,y0,y1) para estimar el area si no se entrega 'area'
           directamente (usa el area del rectangulo/zona real).

    Devuelve
    --------
    dict con:
      R          : indice de Clark-Evans crudo (0 a ~2.15)
      Iu         : indice acotado [0,1] (1=uniforme, 0=agrupado)
      r_obs      : distancia media observada al vecino mas cercano [mm]
      r_esperado : distancia esperada bajo CSR [mm]
      n          : numero de fibras usadas
    Devuelve NaN en los campos si N<2 o el area no es valida.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)

    if n < 2:
        return {"R": np.nan, "Iu": np.nan, "r_obs": np.nan,
                "r_esperado": np.nan, "n": n}

    if area is None:
        if bbox is not None:
            x0, x1, y0, y1 = bbox
            area = (x1 - x0) * (y1 - y0)
        else:
            # area de la caja envolvente de los propios puntos (fallback)
            area = (x.max() - x.min()) * (y.max() - y.min())

    if area <= 0:
        return {"R": np.nan, "Iu": np.nan, "r_obs": np.nan,
                "r_esperado": np.nan, "n": n}

    # distancia al vecino mas cercano (k=2 porque el punto 0 es el mismo)
    puntos = np.column_stack([x, y])
    arbol = cKDTree(puntos)
    dist, _ = arbol.query(puntos, k=2)
    r_obs = float(np.mean(dist[:, 1]))

    densidad = n / area
    r_esp = 1.0 / (2.0 * np.sqrt(densidad))

    R = r_obs / r_esp
    Iu = float(min(R / R_MAX_HEXAGONAL, 1.0))

    return {"R": R, "Iu": Iu, "r_obs": r_obs, "r_esperado": r_esp, "n": n}


# ----------------------------------------------------------------------
# NOTA sobre efecto de borde (edge effect)
# ----------------------------------------------------------------------
# Las fibras cerca del BORDE de la zona tienen su vecino mas cercano
# potencialmente FUERA de la zona (no observado), lo que sesga r_obs hacia
# arriba y por tanto Iu hacia valores mas "uniformes" de lo real. Para zonas
# pequenas (pocas fibras) esto puede ser significativo. Correccion estandar:
# Donnelly (1978) ajusta r_esperado segun el perimetro de la zona. Si tus
# zonas son grandes respecto al espaciado entre fibras, el sesgo es menor;
# si no, considera implementar la correccion de Donnelly o usar solo fibras
# con un margen de borde (buffer) antes de calcular vecinos.
