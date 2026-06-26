"""
linea_corte_auto.py
===================
Procesa UN solo archivo PIV (.txt), detecta automáticamente el borde de
salida del canal en "L" (respetando la rotación de la viga), construye una
línea de corte transversal (paralela al borde de salida) retrocedida 8 mm
aguas arriba, la grafica sobre el campo de velocidades y guarda línea+imagen.

Formato esperado de columnas (líneas con # se ignoran):
    x(mm)  y(mm)  u(mm/s)  v(mm/s)  magnitude(mm/s)  valid(0/1)

Requiere: pandas, numpy, matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO     = r"C:/Users/elisa/Desktop/PIV_INTERPOLADO/m70-toma-2-n-3000-car-02-piv/frame_000000000.txt"
OUTPUT_PATH = r"Polilinea_L"   # carpeta de salida

RETROCESO   = 8.0     # mm aguas arriba desde el borde de salida
BINS        = 40      # resolución del histograma 2D para densidad
HALF_FACT   = 1.7     # ancho de banda = HALF_FACT * std a lo largo del borde
HALF_MIN    = 0.0    # ancho de banda mínimo (mm)


# -- Lectura --------------------------------------------------
def cargar_piv(path):
    df = pd.read_csv(path, sep=r'\s+', comment='#',
                     names=['x', 'y', 'u', 'v', 'mag', 'valid'])
    df = df[df['valid'] == 1].copy()
    df['y'] *= -1          # inversion de ejes
    df['v'] *= -1
    return df


# -- Deteccion de la zona densa (referencia de la salida) ----
def detectar_centro(x, y):
    """Centro (cx, cy) de la celda del histograma 2D con mas puntos validos."""
    H, xe, ye = np.histogram2d(x, y, bins=BINS)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    cx = 0.5 * (xe[i] + xe[i + 1])
    cy = 0.5 * (ye[j] + ye[j + 1])
    mask = ((x >= xe[i]) & (x < xe[i + 1]) &
            (y >= ye[j]) & (y < ye[j + 1]))
    return cx, cy, mask


# -- Construccion de la linea de corte -----------------------
def construir_linea(df):
    x, y = df['x'].values, df['y'].values

    # 1. Aislar el borde de SALIDA (frontera aguas abajo).
    #    La salida esta hacia (+X, -Y) por la rotacion de ~30 grados hacia abajo.
    ang  = np.deg2rad(-30)
    diag = np.array([np.cos(ang), np.sin(ang)])
    proj = x * diag[0] + y * diag[1]
    umbral = np.percentile(proj, 90)        # 10% mas avanzado = salida
    borde = proj >= umbral
    xb, yb = x[borde], y[borde]

    # 2. Direccion del borde de salida (cara transversal del canal) por PCA.
    bx, by = xb.mean(), yb.mean()
    Xc = np.column_stack([xb - bx, yb - by])
    cov = Xc.T @ Xc / len(Xc)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tx, ty = eigvecs[:, np.argmax(eigvals)]  # tangente al borde (transversal)

    # 3. Eje del flujo = perpendicular al borde, orientado aguas abajo.
    ex, ey = -ty, tx
    if (bx - x.mean()) * ex + (by - y.mean()) * ey < 0:
        ex, ey = -ex, -ey

    # 4. Retroceso aguas arriba desde el borde de salida.
    centro_x = bx - RETROCESO * ex
    centro_y = by - RETROCESO * ey

    # 5. Extremos a lo largo de la tangente (paralelo al borde de salida).
    s_perp = (xb - bx) * tx + (yb - by) * ty
    half = max(np.std(s_perp) * HALF_FACT, HALF_MIN)
    p1 = (centro_x + tx * half, centro_y + ty * half)
    p2 = (centro_x - tx * half, centro_y - ty * half)

    return p1, p2, (bx, by), borde


# -- Guardado ------------------------------------------------
def guardar_linea(p1, p2):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    path = os.path.join(OUTPUT_PATH, "linea_salida.npy")
    np.save(path, np.array([p1, p2]))
    print(f"  OK Linea guardada: {path}")
    return path


def guardar_figura(fig, nombre="diagnostico_linea_corte.png"):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    out = os.path.join(OUTPUT_PATH, nombre)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  OK Figura guardada: {out}")


# -- Grafico de diagnostico ----------------------------------
def graficar(df, p1, p2, centro, mask):
    x, y, mag = df['x'].values, df['y'].values, df['mag'].values

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(x, y, c=mag, cmap='jet', s=8,
                    vmin=0, vmax=np.percentile(mag, 98), alpha=0.75)
    plt.colorbar(sc, ax=ax, label='Velocidad (mm/s)')

    # Borde de salida: marcadores discretos que no tapan los datos
    # ax.scatter(x[mask], y[mask], facecolors='none', edgecolors='white',
    #            s=18, lw=0.8, alpha=0.5, label='Borde salida', zorder=5)
    # ax.plot(*centro, 'w^', ms=10, label='Centro borde', zorder=6)

    # Linea de corte
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=3,
            label='Linea de corte', zorder=7)
    ax.plot(*p1, 'ko', ms=8, zorder=8)
    ax.plot(*p2, 'ko', ms=8, zorder=8)

    ax.annotate(f'P1\n({p1[0]:.1f}, {p1[1]:.1f})', xy=p1,
                xytext=(p1[0] + 2, p1[1] + 2), fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))
    ax.annotate(f'P2\n({p2[0]:.1f}, {p2[1]:.1f})', xy=p2,
                xytext=(p2[0] + 2, p2[1] - 4), fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85))

    ax.set_title('Linea de corte automatica sobre la L',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    guardar_figura(fig)        # guarda la imagen de la L
    plt.show()


# -- Main ----------------------------------------------------
def main():
    df = cargar_piv(ARCHIVO)
    if df.empty:
        print("X No hay puntos validos en el archivo.")
        return
    p1, p2, centro, mask = construir_linea(df)
    print(f"Centro borde: ({centro[0]:.1f}, {centro[1]:.1f}) mm")
    print(f"P1 = ({p1[0]:.1f}, {p1[1]:.1f})  P2 = ({p2[0]:.1f}, {p2[1]:.1f}) mm")
    print(f"Longitud linea de corte: {np.hypot(p2[0]-p1[0], p2[1]-p1[1]):.1f} mm")
    guardar_linea(p1, p2)      # guarda el .npy
    graficar(df, p1, p2, centro, mask)


if __name__ == "__main__":
    main()