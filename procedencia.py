"""
procedencia.py
==============
Sella cada salida del pipeline con los parametros que la generaron.

Motivo: sensibilidad.py reescribe constantes en el codigo fuente, corre el
pipeline y restaura las constantes al terminar. Lo que NO restaura son los
CSV derivados, que quedan con los valores de la ultima combinacion barrida.
El resultado es que el codigo declara DIST_MAX_KNN_MM = 5.0 mientras
acum_tabla_zona.csv contiene numeros calculados con 3.0, sin ninguna senal
visible de que asi sea.

Este modulo hace visible esa procedencia y permite verificarla antes de
publicar cualquier cifra.

Uso tipico:

    from procedencia import sellar, verificar

    sellar("acum_tabla_zona.csv")          # tras escribir el CSV
    verificar("acum_tabla_zona.csv")       # antes de reportar cifras
"""

import json
import re
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# Constantes que definen una corrida. (archivo, nombre de la constante)
PARAMETROS = [
    ("construir_caches_zonas.py", "DIST_MAX_KNN_MM"),
    ("construir_caches_zonas.py", "K_VECINOS"),
    ("construir_caches.py",       "MAX_HUECO_INTERP"),
    ("criterio_exclusion.py", "UMBRAL_D"),
]


def leer_constante(archivo, nombre):
    """Extrae el valor literal de una constante de nivel superior."""
    ruta = RAIZ / archivo
    if not ruta.exists():
        return None
    pat = re.compile(rf"^{re.escape(nombre)}\s*=\s*([^\s#]+)", re.M)
    m = pat.search(ruta.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def parametros_actuales():
    """Diccionario con la configuracion vigente en el codigo fuente."""
    d = {n: leer_constante(f, n) for f, n in PARAMETROS}
    faltantes = [n for n, v in d.items() if v is None]
    if faltantes:
        raise RuntimeError(
            "No se pudieron leer estos parametros: " + ", ".join(faltantes) +
            ". Revisa PARAMETROS: la constante puede haberse movido de archivo "
            "o cambiado de nombre. Sellar con None produce una verificacion "
            "que pasa sin comprobar nada.")
    d["_sellado"] = datetime.now().isoformat(timespec="seconds")
    return d


def _ruta_sello(csv_path):
    p = Path(csv_path)
    return p.with_name(p.stem + ".params.json")


def sellar(csv_path):
    """Escribe el sello junto al CSV. Llamar despues de cada to_csv()."""
    params = parametros_actuales()
    _ruta_sello(csv_path).write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    return params


def verificar(*csv_paths, estricto=True):
    """
    Comprueba que todos los CSV indicados provengan de la misma
    configuracion, y que esa configuracion sea la vigente en el codigo.

    Devuelve True si todo concuerda. Con estricto=True, lanza si no.
    """
    actual = {k: v for k, v in parametros_actuales().items()
              if not k.startswith("_")}
    problemas = []
    sellos = {}

    for c in csv_paths:
        s = _ruta_sello(c)
        if not s.exists():
            problemas.append(f"{c}: sin sello de procedencia")
            continue
        d = json.loads(s.read_text(encoding="utf-8"))
        sellos[c] = {k: v for k, v in d.items() if not k.startswith("_")}

    # Todos los sellos entre si
    valores = list(sellos.values())
    if valores and any(v != valores[0] for v in valores):
        problemas.append("los CSV no provienen de la misma corrida:")
        for c, v in sellos.items():
            problemas.append(f"    {c}: {v}")

    # Sellos contra el codigo vigente
    for c, v in sellos.items():
        difs = {k: (v.get(k), actual.get(k))
                for k in actual if v.get(k) != actual.get(k)}
        if difs:
            problemas.append(
                f"{c}: generado con {({k: a for k, (a, _) in difs.items()})} "
                f"pero el codigo declara {({k: b for k, (_, b) in difs.items()})}")

    if problemas:
        msg = "PROCEDENCIA INCONSISTENTE\n  " + "\n  ".join(problemas)
        if estricto:
            raise RuntimeError(msg)
        print(msg)
        return False

    print(f"procedencia verificada: {actual}")
    return True


if __name__ == "__main__":
    import sys
    objetivos = sys.argv[1:] or [
        "acum_tabla_zona.csv", "acum_capa1_global.csv",
        "acum_capa2_global.csv", "acum_capa4_global.csv",
        "contraste_simpson.csv",
    ]
    existentes = [o for o in objetivos if (RAIZ / o).exists()]
    verificar(*existentes, estricto=False)
