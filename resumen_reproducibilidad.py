"""
resumen_reproducibilidad.py
============================
Exporta las cifras EXACTAS que la memoria necesita para la sección de
validación (Resultados, 4.1.1 y 4.1.2), cerrando los dos huecos marcados
con \\hl{} en chapter04.tex:

  1. REPRODUCIBILIDAD entre mezclas de un mismo Carbopol sin fibras
     (¿dos preparaciones independientes del mismo material producen el
     mismo campo de flujo?).
  2. CONSISTENCIA PTV/PIV: dentro de una misma mezcla, ¿la toma destinada
     a PTV (toma-1) y la destinada a PIV (toma-2) caracterizan el mismo
     campo de velocidad? Esto es lo que justifica, en el Cap. 3, haber
     hecho dos tomas por condición base.

Ambas preguntas solo son respondibles sobre las condiciones n-0000
(sin fibras), porque son las únicas en que etapas_zonas.json contiene
BOTH toma-1 y toma-2 para la misma mezcla (las condiciones con fibras
solo procesan toma-2 para el campo de zonas). Esto no es una limitación
del script: es la razón de diseño por la que existen corridas base
duplicadas (ver Sección de Materiales, Cap. 3).

Se reutilizan las funciones de analisis.py (recolectar_toma_zona, cohen_d,
interpretar_d) para no duplicar la lógica de recorte y carga de caché.

Salidas (todas en el directorio de trabajo):
  - reproducibilidad_entre_mezclas.csv   (pregunta 1)
  - consistencia_ptv_piv.csv             (pregunta 2)
  - resumen_validacion.txt               (cifras listas para citar en LaTeX)

Uso:
    python resumen_reproducibilidad.py
"""

import re
import json
import numpy as np
import pandas as pd

from analisis import (cargar_etapas_zonas, zonas_presentes, cohen_d,
                      interpretar_d, recolectar_toma_zona)

ETAPAS_ZONAS_JSON = "etapas_zonas.json"
ETAPAS = ["transicion", "cuasi"]

OUT_REPRO = "reproducibilidad_entre_mezclas.csv"
OUT_CONSIST = "consistencia_ptv_piv.csv"
OUT_RESUMEN = "resumen_validacion.txt"


# ============================================================
# Descubrimiento de mezclas base (n-0000) y sus tomas
# ============================================================

def mezclas_base(etapas_zonas):
    """
    Devuelve {(reo): {mezcla: {tomas disponibles}}} para las claves n-0000.
    Una 'mezcla' es una preparación física distinta del Carbopol (m71, m82,
    ...); cada mezcla puede tener toma-1 (PTV) y/o toma-2 (PIV).
    """
    out = {}
    for clave in etapas_zonas:
        if "n-0000" not in clave:
            continue
        m = re.match(r"(m\d+)-toma-(\d+)-n-0000-car-(\d+)", clave)
        if not m:
            continue
        mez, toma, reo = m.groups()
        out.setdefault(reo, {}).setdefault(mez, set()).add(toma)
    return out


def carpeta(mezcla, toma, reo):
    return f"{mezcla}-toma-{toma}-n-0000-car-{reo}-piv"


# ============================================================
# Comparación 1: reproducibilidad entre mezclas
# ============================================================

def _pool(mezcla, tomas, reo, zona, etapa, etapas_zonas):
    """Concatena v_mag de todas las tomas de una mezcla para (zona, etapa).
    Devuelve un array vacío (no None) si ninguna toma aporta datos, para que
    el llamador pueda chequear len()==0 de forma uniforme."""
    partes = []
    for t in sorted(tomas):
        v = recolectar_toma_zona(carpeta(mezcla, t, reo), zona, etapa,
                                 etapas_zonas)
        if v is not None and len(v) > 0:
            partes.append(v)
    return np.concatenate(partes) if partes else np.array([])


def reproducibilidad_entre_mezclas(etapas_zonas, zonas):
    """
    Para cada reología, compara TODAS las tomas de una mezcla base contra
    TODAS las tomas de otra mezcla base (mismo material, distinta
    preparación), por zona y etapa. Responde: ¿es reproducible el campo
    de flujo entre corridas independientes del mismo Carbopol?
    """
    mezclas = mezclas_base(etapas_zonas)
    filas = []
    for reo, mezcla_tomas in mezclas.items():
        mezcla_ids = sorted(mezcla_tomas.keys())
        if len(mezcla_ids) < 2:
            print(f"  [reproducibilidad] car-{reo}: solo 1 mezcla base "
                  f"({mezcla_ids}); no hay contraste posible.")
            continue
        # Todos los pares de mezclas distintas (normalmente 1 par: A vs B)
        for i in range(len(mezcla_ids)):
            for j in range(i + 1, len(mezcla_ids)):
                mez_a, mez_b = mezcla_ids[i], mezcla_ids[j]
                for zona in zonas:
                    for etapa in ETAPAS:
                        va = _pool(mez_a, mezcla_tomas[mez_a], reo, zona,
                                  etapa, etapas_zonas)
                        vb = _pool(mez_b, mezcla_tomas[mez_b], reo, zona,
                                  etapa, etapas_zonas)
                        if len(va) == 0 or len(vb) == 0:
                            continue
                        d = cohen_d(va, vb)
                        filas.append({
                            "reologia": f"car-{reo}",
                            "mezcla_A": mez_a, "mezcla_B": mez_b,
                            "zona": zona, "etapa": etapa,
                            "n_A": len(va), "n_B": len(vb),
                            "mediana_A": round(float(np.median(va)), 4),
                            "mediana_B": round(float(np.median(vb)), 4),
                            "cohen_d": round(float(d), 4)
                                      if not np.isnan(d) else np.nan,
                            "efecto": interpretar_d(d),
                        })
    return pd.DataFrame(filas)


# ============================================================
# Comparación 2: consistencia PTV (toma-1) vs PIV (toma-2)
# ============================================================

def consistencia_ptv_piv(etapas_zonas, zonas):
    """
    Dentro de cada mezcla base que tiene AMBAS tomas, compara toma-1 (PTV)
    contra toma-2 (PIV), por zona y etapa. Responde: ¿caracterizan ambas
    tomas el mismo campo de velocidad, lo que habilita usar una para PTV
    y la otra para PIV sin perder representatividad?
    """
    mezclas = mezclas_base(etapas_zonas)
    filas = []
    for reo, mezcla_tomas in mezclas.items():
        for mez, tomas in sorted(mezcla_tomas.items()):
            if not {"1", "2"}.issubset(tomas):
                print(f"  [consistencia] car-{reo} {mez}: solo tiene "
                      f"toma(s) {sorted(tomas)}; se omite (requiere 1 y 2).")
                continue
            for zona in zonas:
                for etapa in ETAPAS:
                    v1 = recolectar_toma_zona(carpeta(mez, "1", reo), zona,
                                              etapa, etapas_zonas)
                    v2 = recolectar_toma_zona(carpeta(mez, "2", reo), zona,
                                              etapa, etapas_zonas)
                    if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
                        continue
                    d = cohen_d(v1, v2)
                    filas.append({
                        "reologia": f"car-{reo}", "mezcla": mez,
                        "zona": zona, "etapa": etapa,
                        "n_ptv": len(v1), "n_piv": len(v2),
                        "mediana_ptv": round(float(np.median(v1)), 4),
                        "mediana_piv": round(float(np.median(v2)), 4),
                        "cohen_d": round(float(d), 4)
                                  if not np.isnan(d) else np.nan,
                        "efecto": interpretar_d(d),
                    })
    return pd.DataFrame(filas)


# ============================================================
# Resumen citable
# ============================================================

def resumen_citable(df_repro, df_consist):
    lineas = []
    lineas.append("RESUMEN DE VALIDACIÓN — cifras para Resultados 4.1.1/4.1.2")
    lineas.append("=" * 65)

    lineas.append("\n--- Reproducibilidad entre mezclas (Carbopol sin fibras) ---")
    if df_repro.empty:
        lineas.append("  Sin datos (revisar que existan >=2 mezclas base por reologia).")
    else:
        for reo, s in df_repro.groupby("reologia"):
            d = s["cohen_d"].dropna()
            pct_debil = (d < 0.5).mean() * 100 if len(d) else np.nan
            lineas.append(
                f"  {reo}: n_comparaciones={len(s)}  "
                f"d mediana={d.median():.3f}  d máx={d.max():.3f}  "
                f"{pct_debil:.0f}% de las celdas con d < 0,5 (débil o menor)")

    lineas.append("\n--- Consistencia PTV (toma-1) vs PIV (toma-2) ---")
    if df_consist.empty:
        lineas.append("  Sin datos (revisar que existan mezclas con toma-1 y toma-2).")
    else:
        for reo, s in df_consist.groupby("reologia"):
            d = s["cohen_d"].dropna()
            pct_debil = (d < 0.5).mean() * 100 if len(d) else np.nan
            lineas.append(
                f"  {reo}: n_comparaciones={len(s)}  "
                f"d mediana={d.median():.3f}  d máx={d.max():.3f}  "
                f"{pct_debil:.0f}% de las celdas con d < 0,5 (débil o menor)")
        lineas.append("\n  Por mezcla:")
        for (reo, mez), s in df_consist.groupby(["reologia", "mezcla"]):
            d = s["cohen_d"].dropna()
            lineas.append(f"    {reo} {mez}: d mediana={d.median():.3f} "
                          f"(n={len(s)} zona×etapa)")

    lineas.append("\nNOTA: d de Cohen, no p-valor (ver Sección 3 — "
                  "herramientas_estadisticas). Umbral de referencia: "
                  "d=0,5 (efecto medio, Cohen 1988), el mismo empleado "
                  "en el criterio de exclusion de celdas.")
    return "\n".join(lineas)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    with open(ETAPAS_ZONAS_JSON, "r", encoding="utf-8") as f:
        etapas_zonas = json.load(f)
    zonas = zonas_presentes(etapas_zonas)
    print(f"Zonas detectadas: {zonas}\n")

    print("Calculando reproducibilidad entre mezclas...")
    df_repro = reproducibilidad_entre_mezclas(etapas_zonas, zonas)
    df_repro.to_csv(OUT_REPRO, index=False)
    print(f"  Guardado: {OUT_REPRO} ({len(df_repro)} filas)")

    print("\nCalculando consistencia PTV vs PIV...")
    df_consist = consistencia_ptv_piv(etapas_zonas, zonas)
    df_consist.to_csv(OUT_CONSIST, index=False)
    print(f"  Guardado: {OUT_CONSIST} ({len(df_consist)} filas)")

    resumen = resumen_citable(df_repro, df_consist)
    with open(OUT_RESUMEN, "w", encoding="utf-8") as f:
        f.write(resumen)
    print(f"\n{resumen}")
    print(f"\nGuardado: {OUT_RESUMEN}")
