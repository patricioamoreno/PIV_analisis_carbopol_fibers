"""
criterio_exclusion.py
=====================
Criterio UNICO de exclusion de celdas (toma, zona, etapa) que no son
comparables con el caso base (tomas n-0000 de la misma reologia).

MOTIVACION -- por que NO se usa el p-valor
------------------------------------------
Los CSV de tests que produce analisis.py comparan, para cada (toma, zona,
etapa), la distribucion de v_mag de esa celda contra la del caso base,
mediante Mann-Whitney. El problema: cada celda agrega DECENAS DE MILES de
puntos PIV (todos los vectores de la zona en todos los frames de la etapa).

Con n de ese tamano, Mann-Whitney declara "significativa" (p < 0.05)
practicamente cualquier diferencia, por trivial que sea. Medido sobre los
CSV reales del proyecto (excluyendo las tomas base, ver mas abajo):

    ~97% de las comparaciones vs_base dan p < 0.05
    pero mas de un tercio de ellas tienen d de Cohen < 0.2 -- efecto que la
    convencion clasifica como "insignificante".

Excluir por p < 0.05 eliminaria casi todo el dataset por diferencias sin
relevancia fisica. El p-valor aqui no mide cuanto difieren las
distribuciones; mide cuantos puntos PIV hay.

A esto se suma que los puntos PIV dentro de una zona NO son independientes
(estan espacialmente correlacionados por la ventana de interrogacion), de
modo que el n efectivo es mucho menor que el nominal y el p-valor esta
aun mas inflado de lo que sugiere el conteo bruto. Es la misma advertencia
de no-independencia que ya aplica a las zonas dentro de una toma.

CRITERIO ADOPTADO
-----------------
Se excluye una celda (toma, zona, etapa) si el tamano de efecto frente al
caso base alcanza d de Cohen >= UMBRAL_D (0.5, "efecto medio" en la
convencion de Cohen 1988). El p-valor se ignora por completo.

Impacto medido sobre las tomas CON fibras (n=168 celdas; las tomas base
n-0000 se excluyen del conteo porque no entran en acum_tabla_zona.csv):

    d >= 0.2  ->  108/168 celdas (64.3%)  -- demasiado agresivo
    d >= 0.5  ->   33/168 celdas (19.6%)  <- ADOPTADO
    d >= 0.8  ->    7/168 celdas ( 4.2%)  -- demasiado permisivo

ADVERTENCIA -- asimetria entre etapas
-------------------------------------
Las celdas de cuasi-estacionario son sistematicamente mas variables frente
al base que las de transicion. En consecuencia, CUALQUIER umbral excluye
mas celdas de cuasi que de transicion:

    umbral      excluidas en cuasi   excluidas en transicion
    d >= 0.2          73.8%                 54.8%
    d >= 0.5          22.6%                 16.7%
    d >= 0.8           6.0%                  2.4%

Esto importa porque la Capa 4 compara exactamente transicion vs cuasi: la
exclusion introduce una asimetria en la comparacion que le da sentido al
analisis. Por eso reportar_asimetria() imprime este desglose, y por eso la
Capa 4 debe correrse CON y SIN exclusiones como chequeo de sensibilidad
(el umbral es una decision metodologica, no un hecho medido).

Verificado sobre los datos reales: la conclusion de la Capa 4
(domina_etapa) es IDENTICA con y sin exclusiones en las 6 combinaciones
respuesta x predictor. La conclusion no depende del umbral.

Uso tipico:
    from criterio_exclusion import cargar_exclusiones, aplicar_exclusiones
    excl = cargar_exclusiones("Analisis_COMPARATIVA_zonas")
    tabla = aplicar_exclusiones(tabla, excl)
"""

import os
import glob

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACION
# ============================================================

# Umbral de d de Cohen. Convencion de Cohen (1988):
#   d < 0.2  insignificante | 0.2-0.5 debil | 0.5-0.8 medio | >= 0.8 fuerte
# Se adopta 0.5 ("efecto medio"). Ver la discusion en el docstring del modulo.
UMBRAL_D = 0.5

# Directorio con los CSV de tests que produce analisis.py
DIR_TESTS = "Analisis_COMPARATIVA_zonas"

# Predictores de fluido y etapas (deben coincidir con carga_real.PREDICTORES
# y carga_real.ETAPAS; se replican aqui para que este modulo no dependa del
# resto del pipeline y pueda usarse de forma aislada).
PREDICTORES = ["V", "omega", "gamma_dot"]
ETAPAS = ["transicion", "cuasi"]

# Tomas descartadas por completo, por decision explicita (no por el criterio
# automatico de d). Se listan aqui para que la decision quede registrada y
# versionada, en vez de vivir solo en la memoria de quien corrio el pipeline.
#
# ATENCION -- granularidad del identificador:
# acum_tabla_zona.csv identifica cada toma SOLO por su codigo de mezcla (mNN),
# sin el sufijo -toma-N. Por tanto, en esta lista un codigo mNN descarta TODAS
# las tomas de esa mezcla, no una sola. Si en el futuro hace falta descartar
# una toma individual de una mezcla con varias, hay que propagar el nombre
# completo hasta la tabla acumulada primero (hoy construir_tabla_zonas_todas.py
# guarda solo mNN via _codigo_toma()).
#
# Nota sobre m93-toma-1: NO se lista aqui porque no hace falta. m93 es una
# toma de control n-0000 (sin fibras), y por tanto no tiene CSV en
# fibras_ultimo_frame/ ni entra nunca en acum_tabla_zona.csv -- el filtro
# estructural del pipeline ya la deja fuera. Agregarla aqui no cambiaria nada
# y ademas descartaria m93-toma-2 por la ambiguedad de granularidad descrita
# arriba.
TOMAS_DESCARTADAS = set()


# ============================================================
# CARGA DE LOS TESTS
# ============================================================

def cargar_tests(dir_tests=DIR_TESTS):
    """
    Concatena todos los *_tests.csv que produce analisis.py.

    Devuelve un DataFrame con las columnas del CSV mas 'grupo' (la subcarpeta
    reoXX_concYYYY de la que viene cada fila). Devuelve DataFrame vacio si no
    encuentra ningun CSV.
    """
    patron = os.path.join(dir_tests, "*", "*_tests.csv")
    archivos = sorted(glob.glob(patron))
    if not archivos:
        print(f"[aviso] no se encontro ningun *_tests.csv en {patron}")
        return pd.DataFrame()

    frames = []
    for f in archivos:
        df = pd.read_csv(f)
        df["grupo"] = os.path.basename(os.path.dirname(f))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _quitar_tomas_base(vb, verbose=False):
    """Descarta las filas de las tomas n-0000 (control, sin fibras).

    Esas tomas no entran en acum_tabla_zona.csv (no tienen CSV de fibras),
    asi que sus celdas nunca se excluyen de nada: solo inflan el denominador
    de los reportes. Ver cargar_exclusiones().
    """
    if "es_toma_base" in vb.columns:
        m_base = vb["es_toma_base"].fillna(False).astype(bool)
    elif "grupo" in vb.columns:
        m_base = vb["grupo"].astype(str).str.contains("conc0000")
    else:
        return vb
    if verbose and m_base.any():
        print(f"[criterio_exclusion] {int(m_base.sum())} filas de tomas base "
              f"(n-0000) ignoradas: no entran en la tabla acumulada")
    return vb[~m_base]


def _codigo_toma(etiqueta):
    """'m73-toma-2' -> 'm73'. La tabla acumulada usa solo el codigo mNN."""
    return str(etiqueta).split("-")[0]


def cargar_exclusiones(dir_tests=DIR_TESTS, umbral_d=UMBRAL_D, verbose=True):
    """
    Devuelve el conjunto de celdas a excluir: {(codigo_toma, zona, etapa)}.

    Solo considera las comparaciones 'vs_base' (cada toma contra el caso base
    de su reologia). Las comparaciones 'entre_tomas' describen dispersion
    dentro del grupo, no desviacion respecto al base, y no se usan aqui.
    """
    tests = cargar_tests(dir_tests)
    if tests.empty:
        return set()

    vb = tests[tests["comparacion"] == "vs_base"].copy()
    vb["cohen_d"] = pd.to_numeric(vb["cohen_d"], errors="coerce")
    vb = _quitar_tomas_base(vb, verbose=verbose)

    # NaN en cohen_d => no se puede evaluar => no se excluye (conservador:
    # no se descarta un dato por no haber podido medir su desviacion).
    marcadas = vb[vb["cohen_d"] >= umbral_d]

    excl = {(_codigo_toma(r["toma"]), r["zona"], r["etapa"])
            for _, r in marcadas.iterrows()}

    if verbose:
        n_tot = len(vb)
        print(f"[criterio_exclusion] d >= {umbral_d}: "
              f"{len(marcadas)}/{n_tot} celdas marcadas "
              f"({100*len(marcadas)/n_tot:.1f}%)")
    return excl


# ============================================================
# APLICACION
# ============================================================

def aplicar_exclusiones(tabla, exclusiones, predictores=PREDICTORES,
                        verbose=True):
    """
    Marca como NaN los predictores de fluido de las celdas excluidas, y
    elimina por completo las filas de las tomas en TOMAS_DESCARTADAS.

    IMPORTANTE -- por que NaN y no eliminar la fila:
    una fila de la tabla es un par (toma, zona) y contiene los predictores de
    AMBAS etapas. Si una toma se desvia del base solo en cuasi (caso real de
    m73: 5 de 8 zonas con d >= 0.5 en cuasi, pero solo 1 de 8 en transicion),
    eliminar la fila entera tiraria tambien sus datos de transicion, que son
    perfectamente utilizables. Marcando solo las columnas de la etapa afectada
    se conserva el resto.

    Esto funciona sin tocar el codigo de las Capas porque:
      - capa1_global enmascara NaN por columna (x/y con ~np.isnan), asi que
        cada correlacion usa las filas donde ESE predictor existe.
      - capa2_global hace dropna(subset=cols) donde cols son solo los
        predictores de la etapa que esta ajustando, asi que un NaN en cuasi
        no elimina la fila del modelo de transicion.

    Devuelve (tabla_modificada, resumen_dict).
    """
    t = tabla.copy()

    # 1) tomas descartadas por decision explicita
    n_antes = len(t)
    if "toma" in t.columns and TOMAS_DESCARTADAS:
        codigos_descartados = {_codigo_toma(x) for x in TOMAS_DESCARTADAS}
        t = t[~t["toma"].astype(str).isin(codigos_descartados)]
    n_por_toma = n_antes - len(t)

    # 2) exclusiones por celda
    n_celdas = 0
    for (cod, zona, etapa) in exclusiones:
        cols = [f"{p}_{etapa}" for p in predictores if f"{p}_{etapa}" in t.columns]
        if not cols:
            continue
        m = (t["toma"].astype(str) == cod) & (t["zona"].astype(str) == zona)
        if m.any():
            t.loc[m, cols] = np.nan
            n_celdas += int(m.sum())

    resumen = {"filas_eliminadas_por_toma": n_por_toma,
               "celdas_marcadas_nan": n_celdas,
               "filas_finales": len(t)}

    if verbose:
        if n_por_toma:
            print(f"[criterio_exclusion] {n_por_toma} filas eliminadas "
                  f"(tomas descartadas: {sorted(TOMAS_DESCARTADAS)})")
        print(f"[criterio_exclusion] {n_celdas} celdas (toma,zona,etapa) "
              f"marcadas como NaN")
    return t, resumen


# ============================================================
# DIAGNOSTICO
# ============================================================

def reportar_asimetria(dir_tests=DIR_TESTS, umbrales=(0.2, 0.5, 0.8)):
    """
    Imprime cuantas celdas caeria cada umbral, desglosado por etapa.

    La asimetria entre etapas NO es un detalle: la Capa 4 compara transicion
    contra cuasi, de modo que excluir mas celdas de una etapa que de la otra
    sesga justo esa comparacion. Este reporte existe para que esa asimetria
    quede documentada junto al resultado, no para corregirla automaticamente.
    """
    tests = cargar_tests(dir_tests)
    if tests.empty:
        return None

    vb = tests[tests["comparacion"] == "vs_base"].copy()
    vb["cohen_d"] = pd.to_numeric(vb["cohen_d"], errors="coerce")
    vb = _quitar_tomas_base(vb)

    print("\n" + "="*62)
    print("ASIMETRIA DE EXCLUSIONES ENTRE ETAPAS")
    print("="*62)

    print("\nd de Cohen por etapa (vs base):")
    print(vb.groupby("etapa")["cohen_d"]
          .agg(["count", "mean", "median", "max"]).round(3).to_string())

    filas = []
    for u in umbrales:
        vb["_excl"] = vb["cohen_d"] >= u
        for etapa, g in vb.groupby("etapa"):
            filas.append({"umbral_d": u, "etapa": etapa,
                          "excluidas": int(g["_excl"].sum()),
                          "total": len(g),
                          "pct": round(100*g["_excl"].mean(), 1)})
    res = pd.DataFrame(filas)
    print("\nCeldas excluidas por umbral y etapa:")
    print(res.pivot(index="umbral_d", columns="etapa",
                    values="pct").to_string())
    print("\n(porcentaje de celdas excluidas; cuasi > transicion en todos los")
    print(" umbrales -> reportar la Capa 4 con y sin exclusiones)")
    print("="*62 + "\n")
    return res


def resumen_por_toma(dir_tests=DIR_TESTS, umbral_d=UMBRAL_D):
    """Tabla toma x etapa con cuantas celdas excluye el criterio."""
    tests = cargar_tests(dir_tests)
    if tests.empty:
        return None
    vb = tests[tests["comparacion"] == "vs_base"].copy()
    vb["cohen_d"] = pd.to_numeric(vb["cohen_d"], errors="coerce")
    vb = _quitar_tomas_base(vb)
    vb["_excl"] = vb["cohen_d"] >= umbral_d
    piv = vb.pivot_table(index="toma", columns="etapa", values="_excl",
                         aggfunc="sum", fill_value=0)
    piv["total"] = piv.sum(axis=1)
    return piv.sort_values("total", ascending=False)


if __name__ == "__main__":
    reportar_asimetria()
    print("Celdas excluidas por toma (d >= "
          f"{UMBRAL_D}):")
    print(resumen_por_toma().to_string())
