"""
inspeccionar_json.py
====================
No sabes como esta estructurado tu JSON de etapas? Este script lo lee y te
dice exactamente que tiene dentro y si el pipeline puede usarlo tal cual.

Uso:
    python inspeccionar_json.py ruta/a/tu_archivo.json

No modifica nada. Solo lee y te explica.
"""

import sys
import json

JSON_INSPECTOR = "etapas_zonas.json"  # nombre de ejemplo que uso en el pipeline real

def inspeccionar(path):
    with open(path) as f:
        data = json.load(f)

    print("=" * 60)
    print(f"Archivo: {path}")
    print("=" * 60)
    print(f"Tipo raiz: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"Claves de nivel 1 ({len(data)}): {list(data.keys())[:12]}")
        print()
        # mostrar la forma del primer par util
        for k, v in data.items():
            if k.startswith("_"):
                continue
            print(f"Ejemplo  '{k}'  ->  {type(v).__name__}: {repr(v)[:80]}")
            if isinstance(v, dict):
                print(f"          subclaves: {list(v.keys())}")
            break

        # diagnostico de formato
        print("\n--- DIAGNOSTICO ---")
        zonas_tipicas = {"Vf1c1", "Vf1c2", "Vf2c1", "Vf2c2", "Vf2c3",
                         "Z1", "Z2", "Z3"}
        claves = set(data.keys())

        # FORMATO D (real): claves compuestas '..._{zona}' con dict que trae t_quasi
        es_D = any(isinstance(v, dict) and "t_quasi" in v
                   for v in data.values())
        if es_D:
            # contar tomas y zonas
            tomas, zonas_vistas = set(), set()
            for k, v in data.items():
                if isinstance(v, dict) and "t_quasi" in v:
                    tk, _, zona = k.rpartition("_")
                    tomas.add(tk); zonas_vistas.add(zona)
            print("FORMATO D detectado (el real): clave = {toma}_{zona}, "
                  "valor con t_quasi/t_peak.")
            print(f"  Tomas en el JSON: {len(tomas)}")
            print(f"  Zonas por toma:   {sorted(zonas_vistas)}")
            print("  Corte usado:      t_quasi (inicio del cuasi-estacionario)")
            print("-> El pipeline lo usa DIRECTO. Solo pasa --etapas con la "
                  "ruta del JSON; empareja cada .npz con su toma automaticamente.")
            print("\nSi sale 'usa DIRECTO', ya esta: no tienes que hacer nada mas.")
            return

        valores = [v for k, v in data.items() if not k.startswith("_")]

        if "global" in claves:
            print("FORMATO B detectado: un corte global.")
            print("-> El pipeline lo usa directo. Pasa --etapas a run_real.py")
        elif claves & zonas_tipicas:
            ejemplo = next(v for k, v in data.items()
                           if k in zonas_tipicas)
            if isinstance(ejemplo, dict):
                print("FORMATO C detectado: rangos por zona "
                      "({'transicion':[...], 'cuasi':[...]}).")
            else:
                print("FORMATO A detectado: un t_corte por zona.")
            print("-> El pipeline lo usa directo. Pasa --etapas a run_real.py")
        else:
            print("Formato NO reconocido automaticamente.")
            print("Tus claves no coinciden con nombres de zona "
                  f"({sorted(zonas_tipicas)}).")
            print("Opciones:")
            print("  1) Si las claves son nombres de zona con otra notacion, "
                  "dime cuales y ajusto el lector.")
            print("  2) Mandame este output y adapto cargar_etapas() a tu "
                  "estructura exacta.")
    elif isinstance(data, list):
        print(f"Es una lista de {len(data)} elementos.")
        print(f"Primer elemento: {repr(data[0])[:120]}")
        print("\n--- DIAGNOSTICO ---")
        print("Formato lista: necesito ver como identifica cada zona/tiempo.")
        print("Mandame este output y adapto cargar_etapas().")
    else:
        print(f"Contenido: {repr(data)[:200]}")

    print("\nSi sale 'usa directo', ya esta: no tienes que hacer nada mas.")


if __name__ == "__main__":
    if len(JSON_INSPECTOR) < 2:
        print(f"Uso: {JSON_INSPECTOR}")
        sys.exit(1)
    inspeccionar(JSON_INSPECTOR)