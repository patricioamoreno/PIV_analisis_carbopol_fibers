"""
nucleo.py
=========
Utilidades compartidas por las 4 capas de analisis de orientacion/dispersion
de fibras (tesis viga L + Carbopol con fibras).

Contiene:
  - Convencion de zonas (grilla 2x3 de la viga, de definir_zonas.py).
  - Asignacion de fibras a zona por posicion del centroide.
  - Manejo de theta CIRCULAR (orientacion de fibras: 0 y 180 equivalen).
  - Metrica de dispersion de centroides por zona.
  - VIF (factor de inflacion de varianza) para diagnosticar colinealidad.
  - Simulador de datos realista (variacion intra-zona) para pruebas.

Convenciones fijas
------------------
PREDICTORES = ["V", "omega", "gamma_dot"]
ETAPAS      = ["transicion", "cuasi"]
zona_id     = fila*N_COLS + col   (fila,col base 0)  ->  {0..5}
etiqueta    = "f{fila+1}c{col+1}"
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CONFIGURACION GLOBAL
# ----------------------------------------------------------------------
N_FILAS, N_COLS = 2, 3
N_ZONAS = N_FILAS * N_COLS
PREDICTORES = ["V", "omega", "gamma_dot"]
ETAPAS = ["transicion", "cuasi"]

# Geometria de la viga (de definir_zonas.py)
X_REF, Y_REF = 151.0, 0.0
ANCHO_VIGA, ALTURA_VIGA = 300.0, 75.0
DX_CELDA = ANCHO_VIGA / N_COLS      # 100 mm
DY_CELDA = ALTURA_VIGA / N_FILAS    # 37.5 mm


def etiqueta_zona(zona_id):
    fila, col = divmod(int(zona_id), N_COLS)
    return f"f{fila + 1}c{col + 1}"


def asignar_zona(x_mm, y_mm):
    """
    Asigna zona_id a un centroide (x_mm, y_mm) usando la grilla de la viga.
    Devuelve np.nan si el punto cae fuera de la viga.

    La viga ocupa x in [X_REF, X_REF+ANCHO], y in [Y_REF-ALTURA, Y_REF].
    fila 0 = arriba (y cercano a 0); col 0 = izquierda (x cercano a X_REF).
    """
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_mm, dtype=float)
    col = np.floor((x - X_REF) / DX_CELDA)
    fila = np.floor((Y_REF - y) / DY_CELDA)
    dentro = (col >= 0) & (col < N_COLS) & (fila >= 0) & (fila < N_FILAS)
    zona = fila * N_COLS + col
    zona = np.where(dentro, zona, np.nan)
    return zona


# ----------------------------------------------------------------------
# THETA CIRCULAR  (orientacion de fibras en [0, 180))
# ----------------------------------------------------------------------
# Una fibra a 10 grados y otra a 170 grados estan casi alineadas: la
# diferencia angular real es 20 grados, no 160. Por eso NO se puede tratar
# theta como un escalar lineal sin cuidado. Se usan dos herramientas:
#   1. Para correlacion/regresion: proyectar a (cos 2t, sin 2t) o usar el
#      "orden parametro" como respuesta escalar bien definida.
#   2. Para resumir orientacion de una zona: direccion media circular.

def theta_a_rad(theta_deg):
    return np.deg2rad(np.asarray(theta_deg, dtype=float))


def direccion_media_circular(theta_deg):
    """
    Direccion media de orientaciones en [0,180) (eje, no vector).
    Se duplica el angulo (2*theta) para mapear el eje a un circulo completo,
    se promedia en el plano complejo y se divide por 2.
    Devuelve (theta_media_deg, R) con R en [0,1] = coherencia (1=alineadas).
    """
    t = theta_a_rad(theta_deg)
    z = np.exp(2j * t)
    zbar = np.nanmean(z)
    theta_media = (np.angle(zbar) / 2.0) % np.pi
    R = np.abs(zbar)
    return np.rad2deg(theta_media), float(R)


def orden_parametro(theta_deg):
    """
    Orden parametro de orientacion S = <cos(2(theta - theta_media))> = R.
    Escalar en [0,1]; 0 = isotropico (sin orientacion preferente),
    1 = todas las fibras perfectamente alineadas. Buen resumen por zona.
    """
    _, R = direccion_media_circular(theta_deg)
    return R


def features_circular(theta_deg):
    """
    Proyeccion de theta a dos componentes lineales (cos2t, sin2t) para usar
    theta como VARIABLE DEPENDIENTE en regresion/correlacion sin el problema
    del wrap. Se modela cada componente y se combinan.
    Devuelve DataFrame con columnas theta_cos, theta_sin.
    """
    t = theta_a_rad(theta_deg)
    return pd.DataFrame({"theta_cos": np.cos(2 * t),
                         "theta_sin": np.sin(2 * t)})


# ----------------------------------------------------------------------
# DISPERSION DE CENTROIDES POR ZONA
# ----------------------------------------------------------------------
def dispersion_centroides(x_mm, y_mm):
    """
    Mide que tan dispersos estan los centroides de las fibras de una zona.
    Devuelve dict con varias metricas:
      - sigma_iso : raiz de la suma de varianzas en x e y (RMS de dispersion)
      - sigma_x, sigma_y : desviaciones por eje
      - area_convex_aprox : 4*sigma_x*sigma_y (proxy de area ocupada)
      - n : numero de fibras
    Robusta a NaN; requiere >=2 fibras para varianza.
    """
    x = np.asarray(x_mm, dtype=float)
    y = np.asarray(y_mm, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    n = int(m.sum())
    if n < 2:
        return {"sigma_iso": np.nan, "sigma_x": np.nan, "sigma_y": np.nan,
                "area_aprox": np.nan, "n": n}
    sx = np.std(x[m], ddof=1)
    sy = np.std(y[m], ddof=1)
    return {"sigma_iso": float(np.hypot(sx, sy)),
            "sigma_x": float(sx), "sigma_y": float(sy),
            "area_aprox": float(4 * sx * sy), "n": n}


# ----------------------------------------------------------------------
# VIF  (diagnostico de colinealidad de los predictores)
# ----------------------------------------------------------------------
def calcular_vif(X):
    """
    VIF de cada columna de X (DataFrame de predictores ya numerico).
    VIF_j = 1 / (1 - R2_j), con R2_j de regresar x_j contra el resto.
    Regla practica: VIF>5 colinealidad notable; VIF>10 severa -> los beta
    individuales de la regresion no son interpretables (usar Capa 3).
    Devuelve Series {predictor: vif}.
    """
    from sklearn.linear_model import LinearRegression
    cols = list(X.columns)
    vifs = {}
    for c in cols:
        otros = [k for k in cols if k != c]
        if not otros:
            vifs[c] = 1.0
            continue
        r2 = LinearRegression().fit(X[otros], X[c]).score(X[otros], X[c])
        vifs[c] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs, name="VIF")


# ----------------------------------------------------------------------
# SIMULADOR DE DATOS  (para pruebas; reemplazar por datos reales)
# ----------------------------------------------------------------------
def simular_datos(n_fibras_por_zona=60, seed=42, colinealidad=0.6):
    """
    Genera datos sinteticos REALISTAS para probar las 4 capas:
      - Predictores con variacion intra-zona (cada fibra ve un valor local).
      - Colinealidad PARCIAL y controlable entre V, omega, gamma_dot
        (parametro 'colinealidad' in [0,1]) -> evita el caso degenerado de
        correlacion perfecta y permite que VIF/Capa 3 tengan sentido.
      - theta depende de gamma_dot local (+ ruido) -> señal recuperable.
      - dispersion de zona depende de V de la zona en transicion.

    Devuelve df_largo: una fila por (fibra, etapa) con columnas:
      fibra_id, zona_id, zona, fila, col, x_mm, y_mm, theta, etapa,
      V, omega, gamma_dot
    """
    rng = np.random.default_rng(seed)
    filas = []
    fid = 0

    # niveles base de fluido por zona y etapa (medianas "verdaderas")
    base = {}
    for etapa in ETAPAS:
        for z in range(N_ZONAS):
            f, c = divmod(z, N_COLS)
            v0 = (40 if etapa == "transicion" else 8) + c * 4 + f * 2
            g0 = (15 if etapa == "transicion" else 4) * rng.uniform(0.8, 1.2)
            w0 = rng.uniform(1, 5) + (2 if etapa == "transicion" else 0)
            base[(z, etapa)] = (v0, w0, g0)

    for z in range(N_ZONAS):
        n = n_fibras_por_zona + int(rng.integers(-12, 12))
        for _ in range(n):
            x = rng.uniform(X_REF, X_REF + ANCHO_VIGA)
            y = rng.uniform(Y_REF - ALTURA_VIGA, Y_REF)
            # theta es la FOTO FINAL: un solo valor por fibra, depende del
            # gamma_dot local de la etapa cuasi. Se calcula antes del loop
            # de etapas y se repite (la orientacion final no cambia segun
            # con que etapa la emparejemos en la tabla larga).
            _, _, g0_cuasi = base[(z, "cuasi")]
            g_local_cuasi = g0_cuasi * (1 + 0.25 * rng.normal(0, 1))
            theta = (4.0 * g_local_cuasi + rng.normal(0, 18)) % 180

            for etapa in ETAPAS:
                v0, w0, g0 = base[(z, etapa)]
                comun = rng.normal(0, 1)
                v = v0 * (1 + 0.25 * (colinealidad * comun +
                                      (1 - colinealidad) * rng.normal(0, 1)))
                if etapa == "cuasi":
                    g = g_local_cuasi
                else:
                    g = g0 * (1 + 0.25 * (colinealidad * comun +
                                          (1 - colinealidad) * rng.normal(0, 1)))
                w = w0 * (1 + 0.25 * (colinealidad * comun +
                                      (1 - colinealidad) * rng.normal(0, 1)))
                filas.append({
                    "fibra_id": fid, "zona_id": z, "zona": etiqueta_zona(z),
                    "fila": z // N_COLS, "col": z % N_COLS,
                    "x_mm": x, "y_mm": y, "theta": theta, "etapa": etapa,
                    "V": v, "omega": w, "gamma_dot": g,
                })
            fid += 1

    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = simular_datos()
    print("Filas:", len(df), "| fibras:", df.fibra_id.nunique())
    print(df.head())
    # chequeo VIF global
    print("\nVIF (cuasi):")
    print(calcular_vif(df[df.etapa == "cuasi"][PREDICTORES]))
    print("\nDireccion media zona 0:",
          direccion_media_circular(
              df[(df.zona_id == 0) & (df.etapa == "cuasi")].theta))