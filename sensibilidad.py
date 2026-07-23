"""
analisis_sensibilidad.py
========================

Análisis de sensibilidad de los umbrales de reconstrucción de vacíos
(Sección "Reconstrucción acotada de vacíos" del Capítulo 3).

PREGUNTA QUE RESPONDE
---------------------
Los tres umbrales del pipeline se fijaron por argumento dimensional:

  1. MAX_HUECO_INTERP  = 3 puntos   (construir_caches.py)
  2. radio contaminación = w//2 + 1 (construir_caches.py, contaminados())
  3. DIST_MAX_KNN_MM   = 5.0 mm     (construir_caches_zonas.py)

¿Cambian las CONCLUSIONES del Capítulo 4 si se mueven a valores
razonables alternativos? Si no cambian, los umbrales son inocuos y la
limitación declarada en la memoria queda resuelta. Si cambian, hay que
reportarlo: la conclusión dependería de una decisión de procesamiento.

QUÉ SE COMPARA (y qué NO)
-------------------------
NO importa que los valores numéricos de rho cambien un poco: es esperable.
Lo que importa es si se conservan las tres afirmaciones que sostiene la
memoria:

  (A) el ORDEN de los predictores por |rho|      (cuál predice mejor)
  (B) la ETAPA dominante de la Capa 4            (transición vs cuasi)
  (C) el SIGNO de las correlaciones principales  (sentido físico)

USO
---
    python analisis_sensibilidad.py --dry-run   # verifica config, no ejecuta
    python analisis_sensibilidad.py --rapido    # barrido univariado (5 corridas)
    python analisis_sensibilidad.py             # malla completa (9 corridas)
    python analisis_sensibilidad.py --verbose   # muestra la salida del pipeline

Empieza SIEMPRE por --dry-run: comprueba que los scripts existen y muestra
las conclusiones actuales, sin gastar tiempo de cómputo.

Salida: sensibilidad/resumen_sensibilidad.csv  +  veredicto por consola.

NOTA: cada combinación reconstruye cachés y reejecuta el pipeline, así que
esto es LENTO (del orden de horas si la malla es grande). Empieza con
--rapido para verificar que el circuito funciona antes de lanzar todo.
"""

import os
import sys
import json
import shutil
import subprocess
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================

RAIZ = Path(__file__).resolve().parent
DIR_SALIDA = RAIZ / "sensibilidad"

# Valores a probar por umbral. El primero de cada lista DEBE ser el valor
# adoptado en la memoria (el caso base contra el que se compara).
MALLA = {
    "MAX_HUECO_INTERP": [3, 2, 5],      # puntos consecutivos rellenables
    "DIST_MAX_KNN_MM":  [5.0, 3.0, 8.0],  # radio de vecinos por zona [mm]
}
# El radio de contaminación (w//2 + 1) se deriva de la ventana de suavizado
# y no es un grado de libertad independiente: moverlo equivale a cambiar w,
# que es un parámetro de medición y no de reconstrucción. Se deja fijo y se
# documenta esa decisión en la memoria.

# Archivos del pipeline que hay que tocar
F_CACHES  = RAIZ / "construir_caches.py"
F_ZONAS   = RAIZ / "construir_caches_zonas.py"

# Resultado final del que se extraen las conclusiones
F_CAPA1   = RAIZ / "acum_capa1_global.csv"

# Pasos del pipeline a reejecutar tras cambiar un umbral.
# Ajusta esta lista si tu orden de ejecución difiere.
PIPELINE = [
    "construir_caches.py",
    "construir_caches_zonas.py",
    "calcular_etapas_polilinea.py",
    "calcular_etapas_zonas.py",
    "construir_tabla_zonas_todas.py",
    "criterio_exclusion.py",
    "analisis_global.py",
]


# ============================================================
# UTILIDADES
# ============================================================

def leer_constante(archivo: Path, nombre: str):
    """Devuelve el valor actual de una constante de módulo."""
    for linea in archivo.read_text(encoding="utf-8").splitlines():
        s = linea.strip()
        if s.startswith(f"{nombre}") and "=" in s:
            valor = s.split("=", 1)[1].split("#")[0].strip()
            return valor
    raise ValueError(f"No se encontró {nombre} en {archivo.name}")


def escribir_constante(archivo: Path, nombre: str, valor):
    """Reescribe en sitio la línea de una constante de módulo."""
    lineas = archivo.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, linea in enumerate(lineas):
        s = linea.strip()
        if s.startswith(f"{nombre}") and "=" in s:
            comentario = ""
            if "#" in linea:
                comentario = "  # " + linea.split("#", 1)[1].strip()
            indent = linea[: len(linea) - len(linea.lstrip())]
            lineas[i] = f"{indent}{nombre} = {valor}{comentario}\n"
            archivo.write_text("".join(lineas), encoding="utf-8")
            return
    raise ValueError(f"No se encontró {nombre} en {archivo.name}")


def correr_pipeline(verbose=False):
    """Ejecuta los pasos del pipeline en orden. Devuelve True si todo OK.

    Los scripts del pipeline imprimen emojis (✅, 🔨, ⚠). En Windows, la
    salida capturada por subprocess usa cp1252 por defecto y esos caracteres
    provocan UnicodeEncodeError DENTRO del subproceso, matándolo aunque el
    cálculo esté bien. Se fuerza UTF-8 vía PYTHONIOENCODING y se decodifica
    con errors='replace' para que ningún carácter raro aborte la corrida.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"   # el subproceso escribe en UTF-8
    env["PYTHONUTF8"] = "1"             # modo UTF-8 de Python 3.7+

    for script in PIPELINE:
        ruta = RAIZ / script
        if not ruta.exists():
            print(f"    ⚠ No existe {script}, se omite")
            continue
        print(f"    · {script}", flush=True)
        res = subprocess.run(
            [sys.executable, str(ruta)],
            cwd=RAIZ,
            capture_output=not verbose,
            text=True,
            encoding="utf-8",      # decodificar la salida como UTF-8
            errors="replace",      # nunca abortar por un carácter suelto
            env=env,
        )
        if res.returncode != 0:
            print(f"    ✗ Falló {script}  (código {res.returncode})")
            if not verbose:
                # Mostrar las últimas líneas reales del error, no solo una.
                cola = (res.stderr or res.stdout or "").strip().splitlines()
                for linea in cola[-8:]:
                    print(f"      | {linea}")
            return False
    return True


def extraer_conclusiones(f_capa1: Path):
    """
    Reduce el CSV de la Capa 1 a las AFIRMACIONES que sostiene la memoria.

    Devuelve un dict con, para cada respuesta (orden_S, sigma_iso):
      - ranking:   predictores ordenados por |rho| (el orden es la conclusión)
      - etapa_dom: etapa dominante por predictor (la pregunta de la Capa 4)
      - signos:    signo de cada rho (el sentido físico)
    """
    df = pd.read_csv(f_capa1)
    out = {}
    for resp, g in df.groupby("respuesta"):
        g = g.copy()
        g["abs_rho"] = g["rho"].abs()

        # (A) ranking global de predictores por |rho| máximo
        rank = (g.groupby("predictor")["abs_rho"].max()
                 .sort_values(ascending=False).index.tolist())

        # (B) etapa dominante por predictor (Capa 4)
        etapa_dom = {}
        for pred, gp in g.groupby("predictor"):
            fila = gp.loc[gp["abs_rho"].idxmax()]
            etapa_dom[pred] = fila["etapa"]

        # (C) signo de cada (predictor, etapa)
        signos = {f"{r.predictor}|{r.etapa}": int(np.sign(r.rho))
                  for r in g.itertuples()}

        out[resp] = {"ranking": rank, "etapa_dom": etapa_dom, "signos": signos}
    return out


def comparar(base: dict, alt: dict):
    """Compara dos conjuntos de conclusiones. Devuelve (idéntico?, detalles)."""
    difs = []
    for resp in base:
        if resp not in alt:
            difs.append(f"{resp}: ausente en la variante")
            continue
        b, a = base[resp], alt[resp]
        if b["ranking"] != a["ranking"]:
            difs.append(f"{resp}: ranking {b['ranking']} → {a['ranking']}")
        for pred, et in b["etapa_dom"].items():
            if a["etapa_dom"].get(pred) != et:
                difs.append(
                    f"{resp}/{pred}: etapa dominante {et} → "
                    f"{a['etapa_dom'].get(pred)}")
        for k, s in b["signos"].items():
            if k in a["signos"] and a["signos"][k] != s:
                difs.append(f"{resp}/{k}: signo {s:+d} → {a['signos'][k]:+d}")
    return (len(difs) == 0), difs


# ============================================================
# MAIN
# ============================================================

def main():
    rapido = "--rapido" in sys.argv
    verbose = "--verbose" in sys.argv
    dry_run = "--dry-run" in sys.argv
    DIR_SALIDA.mkdir(exist_ok=True)

    # ── Comprobación previa: que exista todo antes de gastar horas ──
    faltan = [s for s in PIPELINE if not (RAIZ / s).exists()]
    if faltan:
        print("⚠ Scripts del pipeline que NO existen (se omitirán):")
        for s in faltan:
            print(f"    - {s}")
        print()
    if not F_CAPA1.exists():
        print(f"⚠ No existe {F_CAPA1.name}. Corre el pipeline una vez a mano "
              f"antes de lanzar la sensibilidad.\n")

    if dry_run:
        print("DRY-RUN: se verifica la configuración sin ejecutar nada.\n")
        print("Pipeline que se ejecutaría, en orden:")
        for s in PIPELINE:
            marca = "✓" if (RAIZ / s).exists() else "✗ NO EXISTE"
            print(f"    {marca}  {s}")
        print(f"\nUmbrales actuales:")
        print(f"    MAX_HUECO_INTERP = "
              f"{leer_constante(F_CACHES, 'MAX_HUECO_INTERP')}")
        print(f"    DIST_MAX_KNN_MM  = "
              f"{leer_constante(F_ZONAS, 'DIST_MAX_KNN_MM')}")
        if F_CAPA1.exists():
            print(f"\nConclusiones actuales (caso base):")
            for resp, d in extraer_conclusiones(F_CAPA1).items():
                print(f"    [{resp}] ranking: {' > '.join(d['ranking'])}")
                print(f"             etapa dominante: {d['etapa_dom']}")
        return

    # Guardar los valores originales para restaurarlos al final
    orig = {
        "MAX_HUECO_INTERP": leer_constante(F_CACHES, "MAX_HUECO_INTERP"),
        "DIST_MAX_KNN_MM":  leer_constante(F_ZONAS,  "DIST_MAX_KNN_MM"),
    }
    print("Valores actuales (caso base):")
    for k, v in orig.items():
        print(f"  {k} = {v}")
    print()

    # Construir la lista de combinaciones a probar
    if rapido:
        # Un umbral a la vez, el resto en su valor base (barrido univariado)
        combos = []
        base_vals = {k: v[0] for k, v in MALLA.items()}
        for k, valores in MALLA.items():
            for v in valores:
                c = dict(base_vals)
                c[k] = v
                if c not in combos:
                    combos.append(c)
    else:
        # Malla completa (producto cartesiano)
        claves = list(MALLA)
        combos = [dict(zip(claves, vals))
                  for vals in itertools.product(*(MALLA[k] for k in claves))]

    print(f"Combinaciones a evaluar: {len(combos)}"
          f" ({'univariado' if rapido else 'malla completa'})\n")

    filas = []
    conclusiones_base = None

    for i, combo in enumerate(combos, 1):
        etiqueta = "_".join(f"{k.split('_')[0]}{v}" for k, v in combo.items())
        es_base = all(combo[k] == MALLA[k][0] for k in MALLA)
        print(f"[{i}/{len(combos)}] {combo}"
              f"{'   ← CASO BASE' if es_base else ''}", flush=True)

        # Aplicar umbrales
        escribir_constante(F_CACHES, "MAX_HUECO_INTERP",
                           combo["MAX_HUECO_INTERP"])
        escribir_constante(F_ZONAS, "DIST_MAX_KNN_MM",
                           combo["DIST_MAX_KNN_MM"])

        # Reejecutar
        if not correr_pipeline(verbose=verbose):
            print("    ✗ Pipeline falló; se omite esta combinación\n")
            if es_base:
                print("El CASO BASE falló, así que no habrá contra qué "
                      "comparar. Se aborta.\n"
                      "Corre primero:  python analisis_sensibilidad.py "
                      "--verbose  para ver el error completo.")
                escribir_constante(F_CACHES, "MAX_HUECO_INTERP",
                                   orig["MAX_HUECO_INTERP"])
                escribir_constante(F_ZONAS, "DIST_MAX_KNN_MM",
                                   orig["DIST_MAX_KNN_MM"])
                print("✔ Umbrales originales restaurados")
                return
            continue

        # Extraer conclusiones y archivar el CSV crudo
        concl = extraer_conclusiones(F_CAPA1)
        shutil.copy(F_CAPA1, DIR_SALIDA / f"capa1_{etiqueta}.csv")

        if es_base:
            conclusiones_base = concl
            with open(DIR_SALIDA / "conclusiones_base.json", "w",
                      encoding="utf-8") as f:
                json.dump(concl, f, indent=2, ensure_ascii=False)

        fila = dict(combo)
        fila["etiqueta"] = etiqueta
        # rho principal (el mayor |rho| de orden_S) como valor de referencia
        df = pd.read_csv(F_CAPA1)
        g = df[df["respuesta"] == "orden_S"]
        if len(g):
            top = g.loc[g["rho"].abs().idxmax()]
            fila["top_predictor"] = top["predictor"]
            fila["top_etapa"] = top["etapa"]
            fila["top_rho"] = round(float(top["rho"]), 4)
        filas.append(fila)
        print()

    # Restaurar valores originales
    escribir_constante(F_CACHES, "MAX_HUECO_INTERP", orig["MAX_HUECO_INTERP"])
    escribir_constante(F_ZONAS, "DIST_MAX_KNN_MM", orig["DIST_MAX_KNN_MM"])
    print("✔ Umbrales originales restaurados\n")

    if not filas:
        print("✗ No se completó ninguna combinación.")
        return

    # ── Veredicto ────────────────────────────────────────────
    df_res = pd.DataFrame(filas)
    csv_out = DIR_SALIDA / "resumen_sensibilidad.csv"
    df_res.to_csv(csv_out, index=False)

    print("=" * 64)
    print("RESUMEN")
    print("=" * 64)
    print(df_res.to_string(index=False))
    print()

    if conclusiones_base is None:
        print("⚠ El caso base no se completó; no hay contra qué comparar.")
        return

    estables, inestables = [], []
    for i, combo in enumerate(combos, 1):
        etiqueta = "_".join(f"{k.split('_')[0]}{v}" for k, v in combo.items())
        f_alt = DIR_SALIDA / f"capa1_{etiqueta}.csv"
        if not f_alt.exists():
            continue
        concl = extraer_conclusiones(f_alt)
        igual, difs = comparar(conclusiones_base, concl)
        (estables if igual else inestables).append((etiqueta, difs))

    print(f"Combinaciones que preservan las conclusiones: "
          f"{len(estables)}/{len(estables) + len(inestables)}")
    if inestables:
        print("\n⚠ COMBINACIONES QUE ALTERAN LAS CONCLUSIONES:")
        for et, difs in inestables:
            print(f"\n  {et}")
            for d in difs:
                print(f"    - {d}")
        print("\n→ Los umbrales NO son inocuos. Reportar esta dependencia en "
              "la memoria y justificar la elección con un criterio externo.")
    else:
        print("\n✔ Ninguna variante altera el ranking de predictores, la etapa "
              "dominante ni los signos.")
        print("→ Puedes escribir en la memoria que los resultados son estables "
              "frente a variaciones razonables de estos umbrales, citando "
              f"{csv_out.name}.")


if __name__ == "__main__":
    main()