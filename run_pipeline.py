#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
================
Orquestador del pipeline de análisis PIV/PTV (Carbopol + fibras).

Ejecuta, EN ORDEN, todos los scripts documentados en la sección
"Orden de ejecución" del README del repositorio, y deja correr el resto
aunque alguno falle -- pensado para dejarlo corriendo solo durante la noche.

Qué hace:
  - Corre cada script del pipeline con el mismo intérprete de Python
    que ejecuta este orquestador (sys.executable), en el directorio del repo.
  - Guarda la salida completa (stdout + stderr) de cada script en
    logs/<NN>_<script>.log
  - Si un script falla (código de retorno != 0, excepción, o timeout),
    el error se agrega a reporte.txt (no se sobrescribe entre corridas:
    cada corrida queda como una sección con fecha/hora) y se continúa
    con el siguiente script.
  - Al final imprime y guarda un resumen (qué corrió bien, qué falló,
    cuánto tardó cada paso).

Uso:
    python run_pipeline.py                 # corre todo el pipeline
    python run_pipeline.py --dry-run        # solo muestra el plan, no ejecuta nada
    python run_pipeline.py --only analisis.py construir_tabla_zonas_todas.py
    python run_pipeline.py --skip verificacion_estratificada.py
    python run_pipeline.py --stop-on-error   # corta apenas falla un paso obligatorio
    python run_pipeline.py --desde analisis.py   # arranca desde ese script (salta los previos)

Se debe correr desde la raíz del repositorio (donde están analisis.py, etc.),
o indicando --repo-dir /ruta/al/repo.
"""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# --------------------------------------------------------------------------
# 1. DEFINICIÓN DEL PIPELINE (mismo orden que "## Orden de ejecución" del README)
#    optional=True  -> si falla, se registra el error pero NO se marca como
#                       fallo "crítico" en el resumen (son los que el propio
#                       README describe como chequeo manual / validación opcional)
# --------------------------------------------------------------------------

STEPS = [
    # --- 0. Fibras del último fotograma ---
    dict(script="exportar_fibras_ultimo_frame.py",
         etapa="0. Fibras último fotograma",
         salida="fibras_ultimo_frame/",
         optional=False),

    # --- 1. Cachés base (independientes entre sí) ---
    dict(script="construir_caches.py",
         etapa="1. Cachés — polilíneas",
         salida="cache_completo/",
         optional=False),
    dict(script="construir_caches_zonas.py",
         etapa="1. Cachés — zonas",
         salida="cache_zonas/",
         optional=False),

    # --- 2. Segmentación de etapas (criterio V3) ---
    dict(script="calcular_etapas_polilinea.py",
         etapa="2. Etapas — polilínea",
         salida="etapas_polilinea.json",
         optional=False),
    dict(script="calcular_etapas_zonas.py",
         etapa="2. Etapas — zonas",
         salida="etapas_zonas.json",
         optional=False),

    # --- 3a. Análisis I — polilíneas ---
    dict(script="box_act.py",
         etapa="3a. Análisis I — boxplots v, gamma_punto",
         salida="Boxplots/",
         optional=False),
    dict(script="esp_plug_mag.py",
         etapa="3a. Análisis I — espectrogramas plug",
         salida="Espectrogramas_plug/",
         optional=False),
    dict(script="box_piv_ptv.py",
         etapa="3a. Análisis I — verificación cruzada PIV-PTV",
         salida="Boxplots/Boxplots_PIV_PTV/",
         optional=False),
    dict(script="esp_overlay_piv_ptv.py",
         etapa="3a. Análisis I — overlay velocidad fibras/fluido",
         salida="Esp_PIV-PTV/Overlay/",
         optional=False),

    # --- 3b. Comparativa por zona (alimenta el criterio de exclusión) ---
    dict(script="analisis.py",
         etapa="3b. Comparativa por zona (genera *_tests.csv)",
         salida="Analisis_COMPARATIVA_zonas/",
         optional=False),

    # --- 3c. Análisis II — zonas ---
    dict(script="construir_tabla_zonas_todas.py",
         etapa="3c. Análisis II — tabla acumulada de zonas",
         salida="acum_tabla_zona.csv (+ sin_excluir)",
         optional=False,
         depende_de="Analisis_COMPARATIVA_zonas/ (paso 3b)"),
    dict(script="run_real.py",
         etapa="3c. Análisis II — capas 1/2/4",
         salida="acum_capa{1,2,4}_global.csv",
         optional=False),
    dict(script="generar_mapas.py",
         etapa="3c. Análisis II — mapas y figuras",
         salida="figs_memoria/",
         optional=False),
    dict(script="verificacion_estratificada.py",
         etapa="3c. Chequeo manual estratificado (opcional, no forma parte del flujo automatizado)",
         salida="(impresión en consola)",
         optional=True),

    # --- 4. Validación de la atribución de zona (opcional) ---
    dict(script="reconstruccion_lagrangiana.py",
         etapa="4. Validación atribución de zona E1/E2/E3 (opcional)",
         salida="atribucion_E1_E2_E3.csv",
         optional=True),
]

REQUISITOS = ["numpy", "pandas", "scipy", "matplotlib", "sklearn"]


# --------------------------------------------------------------------------
# 2. UTILIDADES
# --------------------------------------------------------------------------

def ahora():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def verificar_requisitos():
    faltantes = []
    for paquete in REQUISITOS:
        try:
            __import__(paquete)
        except ImportError:
            faltantes.append(paquete)
    return faltantes


def escribir_reporte(reporte_path, texto):
    """Agrega texto al reporte.txt (no lo sobrescribe entre corridas)."""
    with open(reporte_path, "a", encoding="utf-8") as f:
        f.write(texto)


def truncar(texto, max_chars=4000):
    """Recorta salidas muy largas para que reporte.txt siga siendo legible."""
    if texto is None:
        return ""
    if len(texto) <= max_chars:
        return texto
    mitad = max_chars // 2
    return (
        texto[:mitad]
        + f"\n\n[... recortado, ver log completo en logs/ ...]\n\n"
        + texto[-mitad:]
    )


# --------------------------------------------------------------------------
# 3. EJECUCIÓN DE UN PASO
# --------------------------------------------------------------------------

def correr_paso(paso, indice, repo_dir, logs_dir, python_exe, timeout):
    script = paso["script"]
    script_path = repo_dir / script
    log_path = logs_dir / f"{indice:02d}_{script.replace('.py', '')}.log"

    resultado = dict(paso=paso, indice=indice, log_path=log_path)

    if not script_path.exists():
        resultado["estado"] = "NO_ENCONTRADO"
        resultado["duracion"] = 0.0
        resultado["detalle"] = f"No se encontró el archivo {script_path}"
        return resultado

    print(f"[{ahora()}] → ({indice:02d}) {paso['etapa']}  ::  {script}")
    print(f"{'·' * 78}")

    # Forzar UTF-8 en el subproceso: en Windows, cuando stdout/stderr se
    # redirige a un pipe, Python usa por defecto la codificación de la
    # consola (normalmente cp1252), que no puede representar los caracteres
    # Unicode (✓, ⚠, 🔨, etc.) que varios scripts del pipeline imprimen.
    # PYTHONUTF8=1 fuerza el modo UTF-8 de Python (PEP 540); PYTHONIOENCODING
    # es un refuerzo por si el intérprete no lo respeta en alguna versión.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    lineas_log = [f"$ {python_exe} {script}\n(cwd={repo_dir})\n\n"]
    inicio = time.time()
    timeout_flag = {"hit": False}
    proceso = None

    def _matar_por_timeout():
        timeout_flag["hit"] = True
        if proceso is not None:
            try:
                proceso.kill()
            except Exception:
                pass

    try:
        # stderr se combina con stdout: así el orden de las líneas en la
        # consola/log respeta el orden real en que el script las emitió
        # (con streams separados y buffers propios, eso no está garantizado).
        proceso = subprocess.Popen(
            [python_exe, str(script_path)],
            cwd=str(repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,  # line-buffered
        )
    except Exception as e:
        duracion = time.time() - inicio
        resultado["estado"] = "EXCEPCION_ORQUESTADOR"
        resultado["duracion"] = duracion
        resultado["detalle"] = f"{type(e).__name__}: {e}"
        log_path.write_text(f"Excepción al intentar lanzar {script}:\n{e}\n", encoding="utf-8")
        print(f"[{ahora()}]    ✗ EXCEPCIÓN AL LANZAR EL SCRIPT — {e}")
        return resultado

    temporizador = threading.Timer(timeout, _matar_por_timeout)
    temporizador.daemon = True
    temporizador.start()

    try:
        for linea in proceso.stdout:
            print(linea, end="", flush=True)   # streaming en vivo a la consola
            lineas_log.append(linea)
        proceso.wait()
    finally:
        temporizador.cancel()

    duracion = time.time() - inicio
    returncode = proceso.returncode

    if timeout_flag["hit"]:
        resultado["estado"] = "TIMEOUT"
        resultado["detalle"] = f"Superó el timeout de {timeout}s configurado; el proceso fue terminado."
        lineas_log.append(f"\n----- TIMEOUT tras {timeout}s: proceso terminado -----\n")
        print(f"{'·' * 78}")
        print(f"[{ahora()}]    ✗ TIMEOUT (>{timeout}s) — ver {log_path.name}")
    elif returncode == 0:
        resultado["estado"] = "OK"
        print(f"{'·' * 78}")
        print(f"[{ahora()}]    ✓ OK  ({duracion:.1f}s)")
    else:
        resultado["estado"] = "ERROR"
        resultado["detalle"] = truncar("".join(lineas_log[1:]).strip())
        print(f"{'·' * 78}")
        print(f"[{ahora()}]    ✗ ERROR (código {returncode}, {duracion:.1f}s) — ver {log_path.name}")

    lineas_log.append(f"\n----- return code: {returncode} -----\n")
    log_path.write_text("".join(lineas_log), encoding="utf-8")

    resultado["duracion"] = duracion
    resultado["returncode"] = returncode
    return resultado




# --------------------------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Corre el pipeline completo de análisis PIV/PTV.")
    parser.add_argument("--repo-dir", default=".", help="Carpeta raíz del repositorio (default: carpeta actual).")
    parser.add_argument("--only", nargs="*", default=None, help="Correr solo estos scripts (nombres de archivo).")
    parser.add_argument("--skip", nargs="*", default=[], help="Saltar estos scripts (nombres de archivo).")
    parser.add_argument("--desde", default=None, help="Arrancar desde este script (salta los anteriores en el orden).")
    parser.add_argument("--stop-on-error", action="store_true",
                         help="Cortar la corrida apenas falle un paso NO opcional (por default sigue con todo).")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar el plan de ejecución, no correr nada.")
    parser.add_argument("--timeout", type=int, default=6 * 3600,
                         help="Timeout máximo en segundos por script (default: 6 horas).")
    parser.add_argument("--python", default=sys.executable, help="Intérprete de Python a usar (default: el actual).")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        print(f"ERROR: no existe la carpeta del repositorio: {repo_dir}")
        sys.exit(1)

    logs_dir = repo_dir / "logs_pipeline"
    logs_dir.mkdir(exist_ok=True)
    reporte_path = repo_dir / "reporte.txt"

    # --- Filtrar pasos según --only / --skip / --desde ---
    pasos = STEPS
    if args.desde:
        nombres = [p["script"] for p in pasos]
        if args.desde not in nombres:
            print(f"ERROR: --desde {args.desde} no es un script reconocido del pipeline.")
            sys.exit(1)
        idx = nombres.index(args.desde)
        pasos = pasos[idx:]
    if args.only:
        pasos = [p for p in pasos if p["script"] in args.only]
    if args.skip:
        pasos = [p for p in pasos if p["script"] not in args.skip]

    if not pasos:
        print("No queda ningún paso por correr con esos filtros. Revisa --only/--skip/--desde.")
        sys.exit(1)

    # --- Plan de ejecución ---
    print("=" * 78)
    print("PIPELINE — Fiber Dynamics and Orientation in UHPC (Carbopol)")
    print("=" * 78)
    print(f"Repositorio : {repo_dir}")
    print(f"Python      : {args.python}")
    print(f"Logs        : {logs_dir}")
    print(f"Reporte     : {reporte_path}")
    print(f"Timeout/paso: {args.timeout}s")
    print("-" * 78)
    for i, p in enumerate(pasos, 1):
        marca = " (opcional)" if p["optional"] else ""
        print(f"  {i:02d}. {p['script']:<38} {p['etapa']}{marca}")
    print("=" * 78)

    if args.dry_run:
        print("\n--dry-run: no se ejecutó nada.")
        return

    faltantes = verificar_requisitos()
    if faltantes:
        print(f"\n⚠ AVISO: faltan paquetes de Python: {', '.join(faltantes)}")
        print("  Instala con: pip install " + " ".join(
            {"sklearn": "scikit-learn"}.get(p, p) for p in faltantes
        ))
        print("  Se continúa igual; los scripts que los necesiten fallarán y quedará en reporte.txt.\n")

    input_esperado = repo_dir.parent / "PIV_INTERPOLADO"
    if not input_esperado.exists():
        print(f"⚠ AVISO: no se encontró la carpeta de datos crudos esperada en {input_esperado}")
        print("  (según el README, se esperan en '../PIV_INTERPOLADO/'). Revisa también las rutas")
        print("  hardcodeadas en utils_etapas.py (BASE_PATH, ETAPAS_JSON, CACHE_PATH) y en")
        print("  polilinea_salida_L.py (ARCHIVO) antes de dejar la corrida sola durante la noche:")
        print("  si no apuntan a tu carpeta real, los pasos que dependen de ellas van a fallar.\n")

    inicio_total = time.time()
    fecha_inicio = ahora()

    escribir_reporte(
        reporte_path,
        f"\n\n{'=' * 78}\n"
        f"CORRIDA INICIADA: {fecha_inicio}\n"
        f"Repositorio: {repo_dir}\n"
        f"Pasos planificados: {len(pasos)}\n"
        f"{'=' * 78}\n",
    )

    resultados = []
    corte_por_error = False

    for i, paso in enumerate(pasos, 1):
        r = correr_paso(paso, i, repo_dir, logs_dir, args.python, args.timeout)
        resultados.append(r)

        if r["estado"] != "OK":
            bloque = (
                f"\n[{ahora()}] PASO {i:02d} — {paso['script']}  "
                f"({paso['etapa']})\n"
                f"Estado: {r['estado']}\n"
                f"Salida esperada: {paso['salida']}\n"
                f"Log completo: {r['log_path']}\n"
                f"Detalle:\n{r.get('detalle', '(sin detalle)')}\n"
                f"{'-' * 70}\n"
            )
            escribir_reporte(reporte_path, bloque)

            if not paso["optional"] and args.stop_on_error:
                print(f"\n--stop-on-error activo: se corta la corrida tras fallar un paso obligatorio.")
                corte_por_error = True
                break

    duracion_total = time.time() - inicio_total

    # --- Resumen final ---
    ok = [r for r in resultados if r["estado"] == "OK"]
    fallidos_obligatorios = [r for r in resultados if r["estado"] != "OK" and not r["paso"]["optional"]]
    fallidos_opcionales = [r for r in resultados if r["estado"] != "OK" and r["paso"]["optional"]]

    resumen = []
    resumen.append(f"\n{'=' * 78}")
    resumen.append(f"RESUMEN — corrida terminada {ahora()} (duración total: {duracion_total/60:.1f} min)")
    resumen.append(f"{'=' * 78}")
    resumen.append(f"OK                     : {len(ok)}/{len(resultados)}")
    resumen.append(f"Fallidos (obligatorios): {len(fallidos_obligatorios)}")
    resumen.append(f"Fallidos (opcionales)  : {len(fallidos_opcionales)}")
    if corte_por_error:
        resumen.append("Corrida CORTADA anticipadamente por --stop-on-error.")
    resumen.append("")
    for r in resultados:
        marca = "✓" if r["estado"] == "OK" else "✗"
        resumen.append(
            f"  {marca} {r['indice']:02d}. {r['paso']['script']:<38} "
            f"{r['estado']:<24} {r.get('duracion', 0):6.1f}s"
        )
    resumen.append(f"{'=' * 78}\n")
    resumen_txt = "\n".join(resumen)

    print(resumen_txt)
    escribir_reporte(reporte_path, resumen_txt)

    if fallidos_obligatorios:
        print(f"Hay {len(fallidos_obligatorios)} paso(s) obligatorio(s) con error. Revisa reporte.txt y logs_pipeline/.")
        sys.exit(2)
    else:
        print("Pipeline terminado sin errores obligatorios pendientes.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido manualmente (Ctrl+C).")
        sys.exit(130)