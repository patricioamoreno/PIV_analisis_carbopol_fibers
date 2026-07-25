"""
quitar_titulos.py
=================
Elimina de las figuras los titulos de nivel-figura y los titulos de ejes
unicos, que en una memoria duplican el pie de figura y aportan un registro
coloquial impropio ("¿En que etapa se define la orientacion?").

CONSERVA los titulos de panel dentro de figuras multi-panel, porque ahi el
titulo no duplica el pie: identifica cual panel es cual.

Uso:
    python quitar_titulos.py            # parchea generar_mapas.py in-place
    python generar_mapas.py             # regenera las figuras sin titulos
"""

import re
import shutil

ARCHIVO = "generar_mapas.py"

# Bloque que desactiva SOLO los titulos de nivel-figura.
INYECCION = '''
# ----------------------------------------------------------------------
# Titulos de figura desactivados para la memoria: la informacion va en el
# pie de figura (\\caption), no impresa dentro de la imagen. Los titulos de
# panel en figuras multi-panel SI se conservan, porque identifican paneles.
# ----------------------------------------------------------------------
TITULOS_EN_FIGURA = False

if not TITULOS_EN_FIGURA:
    import matplotlib.figure as _mplfig
    _mplfig.Figure.suptitle = lambda self, *a, **k: None
# ----------------------------------------------------------------------
'''

# Titulos de ejes en figuras de UN SOLO panel: tambien duplican el pie.
# Se identifican por su texto de apertura.
TITULOS_PANEL_UNICO = [
    "Velocidad en transición vs",
    "¿En qué etapa se define la orientación?",
    "Interacción reología × fibras sobre el alineamiento",
    "La relación flujo→",
]


def main():
    shutil.copy(ARCHIVO, ARCHIVO + ".bak")
    src = open(ARCHIVO, encoding="utf-8").read()

    # 1) Inyectar el bloque tras el ultimo import de nivel superior.
    ms = list(re.finditer(r"^(?:import|from)\s+\S+.*$", src, flags=re.M))
    if not ms:
        raise SystemExit("No se encontraron imports en " + ARCHIVO)
    corte = ms[-1].end()
    src = src[:corte] + "\n" + INYECCION + src[corte:]

    # 2) Comentar los ax.set_title de figuras de un solo panel.
    n = 0
    for marca in TITULOS_PANEL_UNICO:
        pat = re.compile(
            r"^([ \t]*)(ax\.set_title\(\s*f?\"" + re.escape(marca) + r".*?\)\s*)$",
            flags=re.M | re.S,
        )

        def _rep(m):
            nonlocal n
            n += 1
            sangria = m.group(1)
            cuerpo = "\n".join(sangria + "# " + l.lstrip()
                               for l in m.group(2).splitlines())
            return cuerpo

        src, k = pat.subn(_rep, src)

    open(ARCHIVO, "w", encoding="utf-8").write(src)
    print(f"parcheado {ARCHIVO}  (respaldo en {ARCHIVO}.bak)")
    print(f"  suptitle desactivado globalmente")
    print(f"  {n} titulos de panel unico comentados")


if __name__ == "__main__":
    main()
