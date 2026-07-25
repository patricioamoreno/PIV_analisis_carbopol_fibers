r"""
esquema_camaras.py
==================
Genera el esquema de cobertura de las cuatro camaras sobre la geometria en L
y la viga, usando la MISMA geometria analitica que definir_zonas.py, de modo
que el esquema y la particion de zonas no puedan divergir.

Salida:  esquema_camaras.pdf  (vectorial, para \includegraphics)

>>> UNICO BLOQUE A EDITAR: FOV_CAMARAS <<<
Los tamanos de campo de visión provienen de la Tabla de parametros de
adquisicion (12.8, 13.1, 13.1 y 3.6 cm de lado). Los CENTROS son los que
debes ajustar a tu montaje real: cambia (cx, cy) en mm, en el mismo sistema
de coordenadas de definir_zonas.py (origen en la esquina superior izquierda
de la viga; el brazo de la L crece hacia +y).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle

from definir_zonas import POLIGONOS_ZONA

# ======================================================================
# CONFIGURACION DE CAMARAS
# Valores tomados directamente de CAM_PROFILES y EDITOR_CONFIG del editor.
#   x_px, y_px : esquina superior-izquierda en pixeles del canvas del editor
#   px_per_mm  : escala PROPIA de la camara
#   width_px   : lado del sensor en pixeles (FOV cuadrado)
#   rot        : rotacion en grados, segun el editor
# ======================================================================
CAMARAS = {
    1: dict(x_px=160.0,  y_px=-1140.0, px_per_mm=8.0,  width_px=1024, rot=-30.0, color="#1f77b4"),
    2: dict(x_px=1140.0, y_px=-140.0,  px_per_mm=7.8,  width_px=1024, rot=0.0,   color="#d62728"),
    3: dict(x_px=2080.0, y_px=-180.0,  px_per_mm=7.8,  width_px=1024, rot=-1.5,  color="#2ca02c"),
    4: dict(x_px=1230.0, y_px=-500.0,  px_per_mm=10.7, width_px=384,  rot=-32.0, color="#9467bd"),
}

# El editor usa y hacia abajo; definir_zonas.py usa y hacia arriba.
FLIP_Y = True
# Signo de la rotacion al pasar de un sistema al otro.
SIGNO_ROT = -1.0

# Zonas de la L y de la viga (para colorearlas distinto)
ZONAS_L = ["Z1", "Z2", "Z3"]

FIGSIZE = (7.2, 7.2)
DPI = 200


def dibujar_zonas(ax):
    """Dibuja los poligonos de las nueve zonas con su etiqueta."""
    for nombre, verts in POLIGONOS_ZONA.items():
        v = np.asarray(verts, dtype=float)
        es_L = nombre in ZONAS_L
        ax.add_patch(MplPolygon(
            v, closed=True,
            facecolor="#d9d9d9" if es_L else "#f0f0f0",
            edgecolor="#4d4d4d",
            linewidth=0.9, zorder=1,
        ))
        cx, cy = v[:, 0].mean(), v[:, 1].mean()
        ax.text(cx, cy, nombre, ha="center", va="center",
                fontsize=8.5, zorder=4,
                color="#1a1a1a" if es_L else "#555555")


def dibujar_fov(ax):
    """Dibuja el campo de vision de cada camara en coordenadas del montaje."""
    for cam, d in sorted(CAMARAS.items()):
        ppm = d["px_per_mm"]
        lado = d["width_px"] / ppm                    # lado del FOV [mm]
        x0 = d["x_px"] / ppm                          # esquina sup-izq [mm]
        y0 = d["y_px"] / ppm
        if FLIP_Y:
            y0 = -y0
        ang = SIGNO_ROT * d["rot"]

        # Rectangulo anclado en la esquina superior-izquierda, creciendo
        # hacia abajo; matplotlib rota en torno al ancla.
        ax.add_patch(Rectangle(
            (x0, y0 - lado), lado, lado,
            angle=ang, rotation_point=(x0, y0),
            fill=False, edgecolor=d["color"],
            linewidth=1.6, linestyle="--", zorder=5,
        ))

        # Etiqueta en el centro geometrico del FOV ya rotado.
        t = np.radians(ang)
        dx, dy = lado / 2.0, -lado / 2.0
        cx = x0 + dx * np.cos(t) - dy * np.sin(t)
        cy = y0 + dx * np.sin(t) + dy * np.cos(t)
        ax.text(cx, cy, f"Cam {cam}", ha="center", va="center",
                fontsize=9, fontweight="bold", color=d["color"], zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=d["color"], linewidth=0.7, alpha=0.9))


def dibujar_flujo(ax):
    """Flecha indicando el sentido del escurrimiento a lo largo de la viga."""
    ax.annotate("", xy=(430, -95), xytext=(170, -95),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.4,
                                color="#333333"), zorder=6)
    ax.text(300, -103, "sentido del escurrimiento",
            ha="center", va="top", fontsize=8.5, color="#333333")


def main(salida="esquema_camaras.pdf"):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    dibujar_zonas(ax)
    dibujar_fov(ax)
    dibujar_flujo(ax)

    ax.set_aspect("equal")
    ax.set_xlim(-40, 480)
    ax.set_ylim(-135, 375)
    ax.set_xlabel("$x$ [mm]", fontsize=10)
    ax.set_ylabel("$y$ [mm]", fontsize=10)
    ax.tick_params(labelsize=8.5)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(True, linewidth=0.3, alpha=0.35, zorder=0)

    fig.tight_layout()
    fig.savefig(salida, bbox_inches="tight")
    fig.savefig(salida.replace(".pdf", ".png"), dpi=DPI, bbox_inches="tight")
    print(f"escrito: {salida}")


if __name__ == "__main__":
    main()
