import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from win10toast import ToastNotifier

# Importamos las funciones y vértices de tu archivo definir_zonas.py
from definir_zonas import (
    asignar_zona_df,
    b1, b2, b3, b4,
    l1, l2, l3, l4, l5, l6,
    v1, v2, v3, v4,
    c1, c2, c3, c4,
    s1, s2, s3, s4,
    n_filas_viga, n_columnas_viga, celda_viga
)

FIBRAS = "fibras_ultimo_frame"  # Cambia esto al nombre de tu archivo de fibras
FIGS_DIR = "figuras_fibras"  # Carpeta donde se guardarán las figuras

os.makedirs(FIGS_DIR, exist_ok=True)

def graficar_fibras_en_setup(ruta_datos, ruta_guardado="fibras_setup.png"):
    # 1. Cargar los datos (Ajusta esto según el formato real de tu archivo)
    # Por ejemplo, si es un CSV delimitado por comas:
    df_fibras = pd.read_csv(ruta_datos)
    
    # Asegúrate de que las columnas se llamen 'x' e 'y'. Si se llaman distinto,
    # renómbralas aquí o cambia los parámetros en asignar_zona_df.
    
    # 2. Asignar zonas a las fibras
    df_fibras = asignar_zona_df(df_fibras, col_x='x_mm', col_y='y_mm', col_out='zona')

    # Imprimir un pequeño resumen en consola
    print("Conteo de fibras por zona:")
    print(df_fibras['zona'].value_counts())

    # 3. Configurar la figura (Mismo estilo que tu código original)
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f7fa')
    fig.patch.set_facecolor('#f5f7fa')

    # -- DIBUJAR FONDO GEOMÉTRICO --
    AZUL_OSC  = '#1a3a6e'
    AZUL_MED  = '#4a7ab5'
    AZUL_CLAR = '#b7d4f0'
    AMARILLO  = '#e8b422'
    NARANJA   = '#d4703a'
    GRIS_VIGA = '#dce8f5'
    GRIS_L    = '#e4eaf4'

    # Viga
    ax.add_patch(plt.Polygon([b1, b2, b3, b4], closed=True,
                 facecolor=GRIS_VIGA, edgecolor=AZUL_OSC, linewidth=1.8, zorder=1))

    # Grilla de la viga
    colores_grilla = ['#c8dff5', '#b0d0ee', '#98c1e7', '#d8eaf8', '#c0daf2', '#a8caec']
    for fila in range(n_filas_viga):
        for col in range(n_columnas_viga):
            verts = celda_viga(fila, col)
            idx   = fila * n_columnas_viga + col
            ax.add_patch(plt.Polygon(verts, closed=True,
                         facecolor=colores_grilla[idx % len(colores_grilla)],
                         edgecolor=AZUL_MED, linewidth=1.0, linestyle='--', alpha=0.7, zorder=2))

    # L-beam
    ax.add_patch(plt.Polygon([l1, l2, l3, l4, l5, l6], closed=True,
                 facecolor=GRIS_L, edgecolor=AZUL_OSC, linewidth=1.8, zorder=3))

    # Zonas Z1, Z2, Z3
    ax.add_patch(plt.Polygon([v1, v2, v3, v4], closed=True, facecolor=AZUL_CLAR, edgecolor=AZUL_OSC, linewidth=1.2, alpha=0.5, zorder=4))
    ax.add_patch(plt.Polygon([c1, c2, c3, c4], closed=True, facecolor=AMARILLO, edgecolor=AZUL_OSC, linewidth=1.2, alpha=0.5, zorder=4))
    ax.add_patch(plt.Polygon([s1, s2, s3, s4], closed=True, facecolor=NARANJA, edgecolor=AZUL_OSC, linewidth=1.2, alpha=0.5, zorder=4))

    # -- DIBUJAR FIBRAS --
    # Diccionario de colores para el scatter según la zona
    color_map = {
        "Z1": "#004488",       # Azul oscuro para que resalte
        "Z2": "#880000",       # Rojo oscuro
        "Z3": "#006600",       # Verde oscuro
        "fuera": "#333333"     # Gris muy oscuro para las que caen fuera
    }

    # Dibujamos iterando por zona para generar correctamente la leyenda
    zonas_presentes = df_fibras['zona'].unique()
    for zona in zonas_presentes:
        df_z = df_fibras[df_fibras['zona'] == zona]
        
        # Si es una celda de la viga (empieza con V), le damos un color purpura, 
        # sino usamos el color mapeado
        color_pto = color_map.get(zona, "#800080") 
        label_pto = f"Fibras en {zona} (n={len(df_z)})"

        ax.scatter(df_z['x_mm'], df_z['y_mm'], 
                   color=color_pto, 
                   edgecolor='white', 
                   linewidth=0.5, 
                   s=30,           # Tamaño del punto
                   zorder=5,       # Por encima de las zonas
                   label=label_pto)

    # Configuraciones finales del gráfico
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Posición final de fibras por Zona — L-beam 30° + Viga')
    
    # Mover la leyenda afuera si hay muchas zonas
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, linestyle=':', linewidth=0.35, alpha=0.5)

    # Ajustar los límites igual que en tu código original
    all_pts = np.vstack([l1, l2, l3, l4, l5, l6, b1, b2, b3, b4])
    ax.set_xlim(all_pts[:, 0].min() - 50, all_pts[:, 0].max() + 40)
    ax.set_ylim(all_pts[:, 1].min() - 20, all_pts[:, 1].max() + 50)

    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado exitosamente como: {ruta_guardado}")

# ============================================================
# Ejecución del script
# ============================================================
if __name__ == "__main__":
    # AQUÍ PON EL NOMBRE DE TU ARCHIVO DE DATOS
    for archivo in glob.glob(os.path.join(FIBRAS, "*.csv")):
        if "m73" in archivo or "_resumen" in archivo:
            continue  # Omitir archivos que contengan "m73" o "_resumen"
        graficar_fibras_en_setup(archivo, ruta_guardado=f"{FIGS_DIR}/fibras_setup_{os.path.basename(archivo)}.png")
    toaster = ToastNotifier()
    toaster.show_toast("VSCode", "¡Tu código de Python terminó exitosamente!", duration=5)