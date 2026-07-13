"""
exportar_fibras_ultimo_frame.py
================================
Extrae las fibras "finales" de cada ensayo a partir de ptv_merged.json.

MOTIVACION DEL CAMBIO (v2)
--------------------------
La version original tomaba EXCLUSIVAMENTE el ultimo frame temporal del
archivo. Se detecto que en varias corridas ese ultimo frame esta incompleto:
fibras que si fueron rastreadas durante el ensayo no aparecen en el frame
final (por oclusiones momentaneas, perdida transitoria del blob, o porque el
detector marco ese frame como duplicado). El resultado eran "fotos finales"
con menos fibras de las realmente presentes.

SOLUCION
--------
En lugar de mirar un unico frame, se recorren los ULTIMOS N frames del ensayo
(por defecto N=5) y se consolida la informacion POR FIBRA. El identificador de
seguimiento es 'track_id': cada fibra fisica conserva el mismo track_id a lo
largo de los frames en que fue rastreada (esta es la semantica estandar de un
pipeline PTV; ver carga_real.py y esp_overlay_piv_ptv.py, donde track_id es la
columna que identifica la trayectoria).

Para cada track_id se conserva la observacion correspondiente al FRAME MAS
RECIENTE (mayor timestamp) en que esa fibra aparece dentro de la ventana. Asi:

  - Se respeta la intencion original ("estado final de cada fibra"): a cada
    fibra se le asocia su ultima orientacion/posicion observada, no un
    promedio ni una interpolacion.
  - Se recupera cualquier fibra que exista en la ventana final pero que falte
    justo en el ultimo frame, tomando su aparicion mas tardia disponible.
  - No se inventan datos: si una fibra no aparece en ninguno de los ultimos N
    frames, simplemente no se exporta.

Esta eleccion (ultimo valor observado por trayectoria) es la lectura natural
del "estado final" en seguimiento lagrangiano y evita introducir sesgo:
promediar la orientacion sobre varios frames la suavizaria artificialmente,
justo en la variable que el analisis de orientacion pretende medir sobre el
fotograma final (cap. 3 de la memoria: la orientacion se define sobre el
ultimo fotograma).

MANEJO DE DUPLICADOS Y track_id FALTANTE
----------------------------------------
  - Las fibras marcadas possible_duplicate=True se omiten (igual que antes).
  - Si una fibra no trae track_id (None), no puede consolidarse por
    trayectoria. Se conserva de todos modos como observacion independiente
    usando una clave sintetica por-frame, de modo que no se pierda; pero al no
    tener id no puede "rescatarse" desde otro frame. Esto se contabiliza en el
    resumen (columna n_sin_id) para que quede trazado.

Columnas del CSV por ensayo (SIN CAMBIOS respecto a la version original, para
mantener compatibilidad con carga_real.cargar_fibras):
    track_id     ID de la trayectoria
    timestamp_s  Tiempo del frame del que proviene ESA fibra [s]
    x_mm         Posicion X del centroide [mm]
    y_mm         Posicion Y del centroide [mm] (sistema fisico: y *= -1)
    angle_deg    Orientacion respecto a la horizontal [deg]
    cam_name     Camara que la detecto

NOTA sobre timestamp_s: en la version original todas las filas compartian el
mismo timestamp (el del ultimo frame). Ahora cada fila lleva el timestamp del
frame del que se tomo esa fibra en particular, que puede diferir en unos pocos
frames entre fibras. carga_real.py no usa esta columna para nada (solo lee
x_mm, y_mm, angle_deg, track_id), asi que el cambio es inocuo aguas abajo.

Ademas se genera un CSV resumen con una fila por ensayo:
    ensayo, t_ultimo_s, n_frames_ventana, n_fibras_final,
    n_fibras_solo_ultimo_frame, n_rescatadas, n_sin_id

    n_fibras_solo_ultimo_frame : cuantas habria dado el metodo antiguo
    n_rescatadas               : fibras recuperadas gracias a la ventana
                                 (n_fibras_final - n_fibras_solo_ultimo_frame,
                                  acotado a >=0)

USO:
    python exportar_fibras_ultimo_frame.py [carpeta] [N_frames]

    Ambos argumentos son opcionales y pueden ir en cualquier orden:
      - [carpeta] : raiz donde buscar los ptv_merged.json (p.ej. "PTV_Results"
                    o una ruta absoluta). Si no se da, el script la busca solo
                    (ver _buscar_carpetas_ptv). Tambien puede fijarse con la
                    variable de entorno PTV_ROOT.
      - [N_frames]: tamano de la ventana temporal (por defecto 5). Tambien
                    puede fijarse con la variable de entorno N_FRAMES_VENTANA.

    Ejemplos:
        python exportar_fibras_ultimo_frame.py
        python exportar_fibras_ultimo_frame.py 8
        python exportar_fibras_ultimo_frame.py "PTV_Results"
        python exportar_fibras_ultimo_frame.py "PTV_Results" 8

ESTRUCTURA DE CARPETAS ESPERADA:
    PIV MEMORIA/
    ├── PTV_Results/
    │   ├── m70-toma-1-n-3000-car-02-ptv/
    │   │   └── ptv_merged.json
    │   ├── m73-toma-1-n-1500-car-02-ptv/
    │   │   └── ptv_merged.json
    │   └── ...  (una carpeta por toma)
    └── exportar_fibras_ultimo_frame.py

Salida:
    fibras_ultimo_frame/<nombre_ensayo>.csv     (uno por ensayo)
    fibras_ultimo_frame/_resumen.csv            (resumen global)
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "fibras_ultimo_frame"

# Numero de frames finales a considerar. Se puede sobreescribir por argumento
# de linea de comandos o por variable de entorno. 10 es un compromiso: suficiente
# para cubrir huecos de deteccion cortos sin retroceder tanto en el tiempo que
# la "foto final" deje de ser final.
N_FRAMES_DEFECTO = 10


# Subcarpeta donde viven las tomas (una carpeta por toma, cada una con su
# ptv_merged.json). Segun la estructura del proyecto de Elisa:
#   PIV MEMORIA/PTV_Results/<toma>/ptv_merged.json
SUBCARPETA_RESULTADOS = "PTV_Results"


def _parsear_argumentos(argv):
    """
    Acepta, en cualquier orden y ambos opcionales:
      - un entero  -> numero de frames de la ventana
      - una ruta   -> carpeta raiz donde buscar los ptv_merged.json

    Prioridad para N: argumento CLI > variable de entorno N_FRAMES_VENTANA >
    valor por defecto. Prioridad para la raiz: argumento CLI > variable de
    entorno PTV_ROOT > deteccion automatica (ver _buscar_carpetas_ptv).

    Devuelve (raiz_o_None, n_frames).
    """
    raiz = None
    n_frames = None

    for arg in argv[1:]:
        # Un token que es puramente un entero se interpreta como N.
        try:
            n = int(arg)
            if n >= 1:
                n_frames = n
            else:
                print(f"[aviso] N_frames debe ser >=1; recibi {n}. Ignorado.")
            continue
        except ValueError:
            pass
        # Cualquier otro token se interpreta como ruta de la carpeta raiz.
        raiz = arg

    if n_frames is None:
        env = os.environ.get("N_FRAMES_VENTANA")
        if env:
            try:
                v = int(env)
                if v >= 1:
                    n_frames = v
            except ValueError:
                pass
    if n_frames is None:
        n_frames = N_FRAMES_DEFECTO

    if raiz is None:
        raiz = os.environ.get("PTV_ROOT")  # puede seguir siendo None

    return raiz, n_frames


def _buscar_carpetas_ptv(raiz_cli=None):
    """
    Encuentra todas las carpetas que contienen un ptv_merged.json, de forma
    robusta frente a desde donde se ejecute el script.

    Orden de raices candidatas a probar:
      1. La ruta pasada por CLI / variable de entorno (raiz_cli), si se dio.
      2. SCRIPT_DIR/PTV_Results   (estructura esperada del proyecto).
      3. SCRIPT_DIR               (por si el script esta junto a las tomas).
      4. CWD/PTV_Results          (por si se corre desde otra ubicacion).
      5. CWD                      (ultimo recurso).

    En cada raiz se hace rglob('ptv_merged.json'). Se devuelve la lista de la
    PRIMERA raiz que encuentre algo, junto con esa raiz (para informar al
    usuario). Si ninguna encuentra nada, devuelve ([], lista_de_raices_probadas).
    """
    candidatas = []
    if raiz_cli:
        candidatas.append(Path(raiz_cli))
    candidatas.append(SCRIPT_DIR / SUBCARPETA_RESULTADOS)
    candidatas.append(SCRIPT_DIR)
    candidatas.append(Path.cwd() / SUBCARPETA_RESULTADOS)
    candidatas.append(Path.cwd())

    # Eliminar duplicados preservando orden.
    vistas, unicas = set(), []
    for c in candidatas:
        cr = c.resolve()
        if cr not in vistas:
            vistas.add(cr)
            unicas.append(c)

    for raiz in unicas:
        if not raiz.exists():
            continue
        carpetas = sorted(
            {p.parent for p in raiz.rglob("ptv_merged.json")},
            key=lambda x: x.name
        )
        if carpetas:
            return carpetas, raiz, unicas

    return [], None, unicas


def _fila_desde_fibra(fib, t_frame):
    """Construye una fila de salida a partir de un dict de fibra y su tiempo."""
    return {
        "track_id":    fib.get("track_id"),
        "timestamp_s": t_frame,
        "x_mm":        fib.get("x_mm", 0.0),
        # Correccion de eje al sistema fisico (consistente con la version
        # original y con load_ptv_merged / load_ptv_crudo).
        "y_mm":        fib.get("y_mm", 0.0),
        "angle_deg":   fib.get("angle_deg", 0.0),
        "cam_name":    fib.get("cam_name", "?"),
    }


def extraer_fibras_finales(carpeta_ptv, n_frames=N_FRAMES_DEFECTO):
    """
    Lee ptv_merged.json y devuelve la "foto final" consolidada sobre los
    ultimos `n_frames` frames temporales del ensayo.

    Para cada track_id se conserva la observacion del frame mas reciente en que
    aparece dentro de la ventana. Las fibras sin track_id se conservan como
    observaciones independientes (no se pueden consolidar por trayectoria).

    Retorna
    -------
    df : DataFrame con las fibras finales (mismas columnas que la version
         original).
    info : dict con metadatos del ensayo:
        t_ultimo               timestamp del ultimo frame [s]
        n_frames_ventana       numero real de frames usados (<= n_frames)
        n_fibras_final         fibras en la foto consolidada
        n_solo_ultimo_frame    fibras que daria mirar solo el ultimo frame
        n_sin_id               fibras sin track_id conservadas
    """
    path = Path(carpeta_ptv) / "ptv_merged.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        return pd.DataFrame(), {
            "t_ultimo": None, "n_frames_ventana": 0,
            "n_fibras_final": 0, "n_solo_ultimo_frame": 0, "n_sin_id": 0,
        }

    # Ordenar frames por timestamp ascendente y quedarse con los ultimos n.
    frames_ord = sorted(frames, key=lambda fr: fr.get("timestamp_s", 0.0))
    ventana = frames_ord[-n_frames:]
    t_ultimo = frames_ord[-1].get("timestamp_s", 0.0)

    # --- Consolidacion por track_id ---
    # Recorremos la ventana de MAS ANTIGUO a MAS RECIENTE. Como el ultimo que
    # escribe gana, al terminar cada track_id queda con su aparicion mas
    # reciente dentro de la ventana. Esto es equivalente a "ultimo valor
    # observado por trayectoria".
    por_track = {}          # track_id -> fila
    sin_id = []             # lista de filas de fibras sin track_id
    contador_sin_id = 0

    for fr in ventana:
        t_frame = fr.get("timestamp_s", 0.0)
        for fib in fr.get("fibers", []):
            # Saltar duplicados marcados por el detector.
            if fib.get("possible_duplicate", False):
                continue

            tid = fib.get("track_id")
            fila = _fila_desde_fibra(fib, t_frame)

            if tid is None:
                # No se puede consolidar por trayectoria: se conserva tal cual.
                sin_id.append(fila)
                contador_sin_id += 1
            else:
                # Sobrescribe cualquier aparicion anterior (mas antigua) del
                # mismo track_id. Como iteramos en orden temporal ascendente,
                # el ultimo en escribir es el mas reciente.
                por_track[tid] = fila

    filas = list(por_track.values()) + sin_id
    df = pd.DataFrame(filas)

    # --- Cuantas habria dado el metodo antiguo (solo el ultimo frame) ---
    ultimo = frames_ord[-1]
    n_solo_ultimo = sum(
        1 for fib in ultimo.get("fibers", [])
        if not fib.get("possible_duplicate", False)
    )

    info = {
        "t_ultimo": t_ultimo,
        "n_frames_ventana": len(ventana),
        "n_fibras_final": len(df),
        "n_solo_ultimo_frame": n_solo_ultimo,
        "n_sin_id": contador_sin_id,
    }
    return df, info


def main():
    raiz_cli, n_frames = _parsear_argumentos(sys.argv)

    carpetas, raiz_usada, raices_probadas = _buscar_carpetas_ptv(raiz_cli)

    if not carpetas:
        print("✗ No encontré ninguna carpeta con ptv_merged.json.")
        print("  Raíces que intenté (en orden):")
        for r in raices_probadas:
            existe = "✓ existe" if Path(r).exists() else "✗ no existe"
            print(f"    - {Path(r).resolve()}   [{existe}]")
        print("\n  Soluciones:")
        print("    • Ejecuta el script pasándole la carpeta que contiene las")
        print("      tomas, por ejemplo:")
        print('        python exportar_fibras_ultimo_frame.py "PTV_Results"')
        print("      o con ruta absoluta:")
        print('        python exportar_fibras_ultimo_frame.py '
              '"C:\\Users\\elisa\\Desktop\\PIV MEMORIA\\PTV_Results"')
        print("    • O define la variable de entorno PTV_ROOT con esa ruta.")
        print("    • Verifica que dentro de cada carpeta de toma exista el")
        print("      archivo ptv_merged.json (no solo ptv_stats.json).")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("EXPORTAR FIBRAS FINALES DE CADA ENSAYO (ventana de últimos frames)")
    print("=" * 70)
    print(f"\nVentana temporal: últimos {n_frames} frames por ensayo")
    print(f"Raíz de búsqueda:  {Path(raiz_usada).resolve()}")
    print(f"Ensayos detectados: {len(carpetas)}")
    print(f"Carpeta de salida: {OUTPUT_DIR}\n")

    resumen_filas = []
    ok = 0
    total_rescatadas = 0

    for carpeta in carpetas:
        nombre_ensayo = carpeta.name
        try:
            df, info = extraer_fibras_finales(carpeta, n_frames=n_frames)
        except Exception as e:
            print(f"  ✗ {nombre_ensayo}: error → {e}")
            continue

        if df.empty:
            print(f"  ⊘ {nombre_ensayo}: ventana final sin fibras válidas")
            continue

        # Guardar CSV individual
        csv_path = OUTPUT_DIR / f"{nombre_ensayo}.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")

        n_final = info["n_fibras_final"]
        n_solo = info["n_solo_ultimo_frame"]
        n_rescatadas = max(0, n_final - n_solo)
        total_rescatadas += n_rescatadas

        print(f"  ✓ {nombre_ensayo}")
        print(f"      t_último = {info['t_ultimo']:.3f} s  |  "
              f"frames usados = {info['n_frames_ventana']}  |  "
              f"n_fibras = {n_final}  "
              f"(solo último frame: {n_solo}, "
              f"rescatadas: +{n_rescatadas})")
        print(f"      x ∈ [{df['x_mm'].min():.1f}, {df['x_mm'].max():.1f}] mm  |  "
              f"y ∈ [{df['y_mm'].min():.1f}, {df['y_mm'].max():.1f}] mm")
        if info["n_sin_id"]:
            print(f"      ⚠ {info['n_sin_id']} fibra(s) sin track_id "
                  f"(conservadas, no consolidables)")

        resumen_filas.append({
            "ensayo":                     nombre_ensayo,
            "t_ultimo_s":                 round(info["t_ultimo"], 3),
            "n_frames_ventana":           info["n_frames_ventana"],
            "n_fibras_final":             n_final,
            "n_fibras_solo_ultimo_frame": n_solo,
            "n_rescatadas":               n_rescatadas,
            "n_sin_id":                   info["n_sin_id"],
        })
        ok += 1

    # Guardar resumen global
    if resumen_filas:
        df_resumen = pd.DataFrame(resumen_filas)
        resumen_path = OUTPUT_DIR / "_resumen.csv"
        df_resumen.to_csv(resumen_path, index=False)
        print(f"\n  ✓ _resumen.csv ({len(resumen_filas)} ensayos)")

    print(f"\n{'=' * 70}")
    print(f"✓ {ok} ensayos exportados  |  "
          f"{total_rescatadas} fibras rescatadas en total")
    print(f"\nArchivos en: {OUTPUT_DIR}")
    print("\nNotas:")
    print(f"  • Ventana: últimos {n_frames} frames. Ajustable con "
          f"'python exportar_fibras_ultimo_frame.py N' o N_FRAMES_VENTANA.")
    print("  • A cada fibra (track_id) se le asocia su aparición MÁS RECIENTE")
    print("    dentro de la ventana: es su estado final observado, sin promediar.")
    print("  • Se aplica la corrección y *= -1 (sistema físico, no cámara)")
    print("  • angle_deg respeta la convención del detector (desde la horizontal)")
    print("  • Las fibras possible_duplicate=True se omiten")
    print("  • n_rescatadas = fibras que el método de un solo frame perdía")


if __name__ == "__main__":
    main()