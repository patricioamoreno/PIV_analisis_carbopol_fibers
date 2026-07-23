"""
definir_zonas.py
================
Geometría de las 3 zonas de análisis (Z1 vertical, Z2 codo, Z3 salida)
de la L-beam 30° + viga horizontal.

REFACTOR respecto a la versión original:
  - La GEOMETRÍA (vértices de cada zona) se calcula siempre al importar.
  - Se expone POLIGONOS_ZONA  y  la función asignar_zona(x, y).
  - TODO el código de dibujo matplotlib quedó bajo  if __name__ == "__main__":
    para que importar este módulo NO genere figuras ni archivos PNG.

Uso desde el pipeline:
    from definir_zonas import asignar_zona, POLIGONOS_ZONA

    zona = asignar_zona(df['x'].values, df['y'].values)   # array de strings
    # zona[i] ∈ {"Z1", "Z2", "Z3", "fuera"}
"""

import numpy as np
from matplotlib.path import Path

# =====================================
# Parámetros geométricos
# =====================================
altura_viga       = 75
ancho_viga        = 300
altura_l_exterior = 294
altura_l_interior = 274
ancho_l_superior  = 85
ancho_l_interior  = 66
altura_l_lateral  = 20
angulo            = 30
separacion_l_viga = 6
x_ref             = 151
y_ref             = 0

# =====================================
# Parámetros de zonas
# =====================================
n_filas_viga    = 2
n_columnas_viga = 3
altura_codo     = 35
ancho_codo      = 100

assert altura_codo >= altura_l_lateral, "Error: altura_codo < altura_l_lateral"
assert ancho_codo  >= ancho_l_superior, "Error: ancho_codo < ancho_l_superior"

# =====================================
# Utilidades trigonométricas
# =====================================
s = np.sin(np.radians(angulo))
c = np.cos(np.radians(angulo))

# =====================================
# Vértices de la L
# =====================================
l1 = np.array([x_ref + separacion_l_viga * s,
               y_ref + separacion_l_viga * c])
l2 = np.array([l1[0] + altura_l_lateral * s,
               l1[1] + altura_l_lateral * c])
l3 = np.array([l2[0] - ancho_l_interior * c,
               l2[1] + ancho_l_interior * s])
l4 = np.array([l3[0] + altura_l_interior * s,
               l3[1] + altura_l_interior * c])
l5 = np.array([l4[0] - ancho_l_superior * c,
               l4[1] + ancho_l_superior * s])
l6 = np.array([l5[0] - altura_l_exterior * s,
               l5[1] - altura_l_exterior * c])

# =====================================
# Vértices de la viga
# =====================================
b1 = np.array([x_ref,              y_ref             ])
b2 = np.array([x_ref + ancho_viga, y_ref             ])
b3 = np.array([x_ref + ancho_viga, y_ref - altura_viga])
b4 = np.array([x_ref,              y_ref - altura_viga])

# =====================================
# Zonas de la L
# =====================================
# Z2: Codo
c1 = l6.copy()
c2 = np.array([c1[0] + ancho_codo * c,  c1[1] - ancho_codo * s])
c3 = np.array([c2[0] + altura_codo * s, c2[1] + altura_codo * c])
c4 = np.array([c3[0] - ancho_codo * c,  c3[1] + ancho_codo * s])

# Z1: Zona vertical
v1 = c4.copy()
v2 = np.array([v1[0] + ancho_l_superior * c, v1[1] - ancho_l_superior * s])
v3 = l4.copy()
v4 = l5.copy()

# Z3: Salida
resto_ancho = ancho_l_superior + ancho_l_interior - ancho_codo
s1 = l1.copy()
s2 = l2.copy()
s3 = np.array([l2[0] - resto_ancho * c,      l2[1] + resto_ancho * s])
s4 = np.array([s3[0] - altura_l_lateral * s, s3[1] - altura_l_lateral * c])

# =====================================
# Grilla de la viga
# =====================================
dx_celda = ancho_viga  / n_columnas_viga
dy_celda = altura_viga / n_filas_viga

def celda_viga(fila, col):
    xmin = x_ref + col       * dx_celda
    xmax = x_ref + (col + 1) * dx_celda
    ymin = y_ref - (fila + 1) * dy_celda
    ymax = y_ref -  fila      * dy_celda
    return np.array([[xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]])


# ============================================================
# >>> NÚCLEO DE ASIGNACIÓN DE ZONA  (lo que faltaba) <<<
# ============================================================
# Cada zona es un polígono de 4 vértices en orden (horario o antihorario).
# Se construye un matplotlib.path.Path por zona y se testea punto-en-polígono
# de forma vectorizada con contains_points().

POLIGONOS_ZONA = {
    "Z1": np.array([v1, v2, v3, v4]),   # Brazo vertical
    "Z2": np.array([c1, c2, c3, c4]),   # Codo
    "Z3": np.array([s1, s2, s3, s4]),   # Salida
}

# — Celdas de la viga como zonas independientes —
# Nombre "Vf{fila}c{col}" con la MISMA numeración 1-based del dibujo
# (f1c1 = celda superior-izquierda). Son n_filas_viga × n_columnas_viga zonas.
for _fila in range(n_filas_viga):
    for _col in range(n_columnas_viga):
        POLIGONOS_ZONA[f"Vf{_fila+1}c{_col+1}"] = celda_viga(_fila, _col)

# Orden de prioridad si un punto cayera en el solape de dos polígonos.
# Z2 (codo) primero por ser la región de interés físico central; las celdas
# de la viga no solapan entre sí ni con Z1/Z2/Z3, así que su orden es indiferente.
_ORDEN_PRIORIDAD = ["Z2", "Z1", "Z3"] + [
    f"Vf{f+1}c{c+1}" for f in range(n_filas_viga) for c in range(n_columnas_viga)
]

_PATHS = {nombre: Path(verts) for nombre, verts in POLIGONOS_ZONA.items()}


def asignar_zona(x, y, radius=0.0):
    """
    Asigna cada punto (x, y) a su zona: "Z1", "Z2", "Z3" o "fuera".

    Parámetros
    ----------
    x, y   : array-like (N,)  — coordenadas YA corregidas (y = y*-1 aplicada)
    radius : float            — tolerancia de borde para contains_points
                                (positivo ensancha levemente el polígono)

    Retorna
    -------
    zona : np.ndarray de dtype '<U5', shape (N,)
           Valores ∈ {"Z1", "Z2", "Z3", "fuera"}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pts = np.column_stack([x, y])

    zona = np.full(len(pts), "fuera", dtype="<U12")

    # Se recorre en orden de prioridad; solo se rellena lo aún "fuera"
    for nombre in _ORDEN_PRIORIDAD:
        dentro = _PATHS[nombre].contains_points(pts, radius=radius)
        nuevos = dentro & (zona == "fuera")
        zona[nuevos] = nombre

    return zona


def asignar_zona_df(df, col_x="x", col_y="y", col_out="zona", radius=0.0):
    """Versión cómoda que añade la columna 'zona' a un DataFrame y lo retorna."""
    df = df.copy()
    df[col_out] = asignar_zona(df[col_x].values, df[col_y].values, radius=radius)
    return df


# ============================================================
# DIBUJO (solo si se ejecuta directamente, NO al importar)
# ============================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f7fa')
    fig.patch.set_facecolor('#f5f7fa')

    AZUL_OSC  = '#1a3a6e'
    AZUL_MED  = '#4a7ab5'
    AZUL_CLAR = '#b7d4f0'
    AMARILLO  = '#e8b422'
    NARANJA   = '#d4703a'
    GRIS_VIGA = '#dce8f5'
    GRIS_L    = '#e4eaf4'

    ax.add_patch(plt.Polygon([b1, b2, b3, b4], closed=True,
                 facecolor=GRIS_VIGA, edgecolor=AZUL_OSC, linewidth=1.8, zorder=1))

    colores_grilla = ['#c8dff5', '#b0d0ee', '#98c1e7',
                      '#d8eaf8', '#c0daf2', '#a8caec']
    for fila in range(n_filas_viga):
        for col in range(n_columnas_viga):
            verts = celda_viga(fila, col)
            idx   = fila * n_columnas_viga + col
            ax.add_patch(plt.Polygon(verts, closed=True,
                         facecolor=colores_grilla[idx % len(colores_grilla)],
                         edgecolor=AZUL_MED, linewidth=1.0,
                         linestyle='--', alpha=0.7, zorder=2))

    ax.add_patch(plt.Polygon([l1, l2, l3, l4, l5, l6], closed=True,
                 facecolor=GRIS_L, edgecolor=AZUL_OSC, linewidth=1.8, zorder=3))

    ax.add_patch(plt.Polygon([v1, v2, v3, v4], closed=True,
                 facecolor=AZUL_CLAR, edgecolor=AZUL_OSC,
                 linewidth=1.2, alpha=0.65, zorder=4, label='Z1 – Brazo vertical'))
    ax.add_patch(plt.Polygon([c1, c2, c3, c4], closed=True,
                 facecolor=AMARILLO, edgecolor=AZUL_OSC,
                 linewidth=1.2, alpha=0.65, zorder=4, label='Z2 – Codo'))
    ax.add_patch(plt.Polygon([s1, s2, s3, s4], closed=True,
                 facecolor=NARANJA, edgecolor=AZUL_OSC,
                 linewidth=1.2, alpha=0.65, zorder=4, label='Z3 – Salida'))

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    # Título omitido: descripción en el caption de la memoria (figura tipo).
    # ax.set_title('Zonas de análisis — L-beam 30° + Viga horizontal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle=':', linewidth=0.35, alpha=0.5)

    all_pts = np.vstack([l1, l2, l3, l4, l5, l6, b1, b2, b3, b4])
    ax.set_xlim(all_pts[:, 0].min() - 50, all_pts[:, 0].max() + 40)
    ax.set_ylim(all_pts[:, 1].min() - 20, all_pts[:, 1].max() + 50)

    plt.tight_layout()
    plt.savefig('zonas_lbeam.png', dpi=150, bbox_inches='tight')
    print("Guardado: zonas_lbeam.png")