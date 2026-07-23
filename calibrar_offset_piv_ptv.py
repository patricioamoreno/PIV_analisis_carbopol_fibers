"""
calibrar_offset_piv_ptv.py
============================
Diagnóstico del desfase temporal entre las adquisiciones PIV y PTV (fibras).

CONTEXTO. En este pipeline, "PTV" en esp_overlay_piv_ptv.py NO es el PTV de
trazadores que valida el campo de flujo (eso se verifica aparte con
resumen_reproducibilidad.py, sobre las corridas n-0000, y da tamaños de efecto
bajos). Aquí "PTV" es el seguimiento de las FIBRAS DE ACERO mismas. La
comparación de esp_overlay_piv_ptv.py responde: ¿la velocidad de la fibra
coincide con la velocidad LOCAL del fluido medida por PIV?

Con DT_OFFSET = 0.0 (sin calibrar), diferencia_piv_ptv.csv mostró RMS
superiores a la velocidad tipica y sesgo sistematico (no centrado en cero),
concentrados en car-0,2%. Segun el Cap. 3 (montaje), PTV y PIV se adquieren
en corridas separadas del mismo material, sin que el documento reporte un
valor de sincronizacion entre ellas -- es decir, el "t=0" de cada adquisicion
puede no estar perfectamente alineado, ademas de la variabilidad normal
corrida-a-corrida ya caracterizada como pequena por resumen_reproducibilidad.

Este script NO asume que el desfase es el culpable: lo pone a prueba.
Para cada (reologia, concentracion, zona), barre un desfase dt en un rango
razonable y calcula el RMS de la diferencia PIV-PTV en funcion de dt. Si
existe un dt* que reduce el RMS de forma sustancial y consistente entre
condiciones, es evidencia de un desfase de sincronizacion real, corregible.
Si el RMS no mejora apreciablemente en ningun dt, la discrepancia observada
en dt=0 no es un problema de alineacion temporal, y debe interpretarse como
deslizamiento fibra-fluido genuino (o variabilidad corrida-a-corrida, dado
que el campo PIV aqui es un promedio de varias tomas de la misma condicion
y se compara contra una unica corrida de fibras) -- un resultado a reportar,
no un error a corregir.

Salidas:
  - calibracion_offset_piv_ptv.csv   (dt* y RMS(dt*) vs RMS(dt=0), por celda)
  - resumen_calibracion_offset.txt   (lectura agregada, cifras citables)

Uso:
    python calibrar_offset_piv_ptv.py
"""

import os
import json
import numpy as np
import pandas as pd

from esp_overlay_piv_ptv import (
    ETAPAS_JSON, REOS, CONCS, ZONAS, PTV_DIR, PCT_LO, PCT_HI,
    ruta_ptv, cargar_ptv, construir_campo_promedio, ss_linea_L, ss_viga,
)

# Rango de desfase a explorar. Historial: +/-3s se acumuló en el borde;
# se amplió a +/-15s bajo el argumento de que PTV y PIV son corridas
# SEPARADAS del mismo material (no adquisición simultánea con disparo
# compartido), así que un desalineamiento de varios segundos en t=0 entre
# corridas es físicamente plausible. El barrido en +/-15s volvió a
# acumularse en el borde (car-05, las 18 condiciones de viga), así que el
# mínimo real sigue fuera de rango. Se amplía a +/-30s. Si vuelve a tocar
# el borde, el problema no es el rango sino la comparabilidad misma entre
# corridas (revisar si conviene abandonar la comparación celda a celda y
# apoyar el supuesto de transporte solo en consistencia_ptv_piv.csv).
DT_MIN, DT_MAX, DT_PASO = -30.0, 30.0, 0.05

OUT_CSV = "calibracion_offset_piv_ptv.csv"
OUT_RESUMEN = "resumen_calibracion_offset.txt"


def rms_para_offset(matriz_vel, tiempos, ss, t_ptv, s_ptv, v_ptv, dt):
    """
    RMS de la diferencia PIV-PTV cuando se desplaza t_ptv en 'dt'. Reutiliza
    el mismo binning por histograma que graficar_diferencia() en
    esp_overlay_piv_ptv.py, para que el RMS aqui sea comparable al ya
    reportado en diferencia_piv_ptv.csv (que corresponde a dt=0).

    CONVENCION DE SIGNO: t_shift = t_ptv + dt. Un dt óptimo NEGATIVO
    significa que el reloj de la PTV iba ADELANTADO respecto al PIV (hay que
    RESTARLE tiempo a t_ptv para alinearlo); un dt POSITIVO significa que el
    reloj de la PTV iba ATRASADO. Verificado con un caso sintético: un
    desfase conocido inyectado en v_ptv(t)=señal(t-Δ) se recupera como
    dt*=-Δ con esta convención.
    """
    t_shift = t_ptv + dt
    m = (t_shift >= tiempos[0]) & (t_shift <= tiempos[-1])
    if m.sum() < 5:
        return np.nan, 0
    t_shift, s_shift, v_shift = t_shift[m], s_ptv[m], v_ptv[m]

    t_edges = np.concatenate([tiempos, [tiempos[-1] + (tiempos[-1]-tiempos[-2])]])
    s_edges = np.concatenate([ss, [ss[-1] + (ss[-1]-ss[-2])]])
    suma, _, _   = np.histogram2d(t_shift, s_shift, bins=[t_edges, s_edges],
                                  weights=v_shift)
    cuenta, _, _ = np.histogram2d(t_shift, s_shift, bins=[t_edges, s_edges])
    with np.errstate(invalid='ignore'):
        v_ptv_grid = np.where(cuenta > 0, suma / cuenta, np.nan)

    diff = matriz_vel - v_ptv_grid
    valido = ~np.isnan(diff)
    n = int(valido.sum())
    if n == 0:
        return np.nan, 0
    rms = float(np.sqrt(np.nanmean(diff[valido] ** 2)))
    return rms, n


def calibrar_condicion(etapas, reo, conc, zona_key, prefijo):
    ptv_path = ruta_ptv(reo, conc, prefijo)
    if not os.path.exists(ptv_path):
        return None

    matriz_full, tiempos_full, nombre, _ = construir_campo_promedio(
        etapas, reo, conc, prefijo, zona_key)
    if matriz_full is None:
        return None

    ss = ss_linea_L() if prefijo == "" else ss_viga()
    if len(ss) != matriz_full.shape[1]:
        ss = np.linspace(ss[0], ss[-1], matriz_full.shape[1])

    t_ptv, s_ptv, v_ptv, _ = cargar_ptv(ptv_path)
    if len(t_ptv) < 10:
        return None

    # Recorte de outliers, IDÉNTICO al que aplica procesar() en
    # esp_overlay_piv_ptv.py antes de calcular diferencia_piv_ptv.csv (dt=0).
    # Sin este paso, errores puntuales de tracking PTV (saltos de ID,
    # oclusiones) producen velocidades espurias de varios órdenes de magnitud
    # que dominan el RMS y lo vuelven no comparable con el ya reportado.
    # Se recorta v_ptv por percentil ANTES del barrido de dt porque es una
    # propiedad de la magnitud de la velocidad, independiente del desfase
    # temporal que se está probando.
    if len(v_ptv):
        lo, hi = np.nanpercentile(v_ptv, [PCT_LO, PCT_HI])
        m_pct = (v_ptv >= lo) & (v_ptv <= hi)
        t_ptv, s_ptv, v_ptv = t_ptv[m_pct], s_ptv[m_pct], v_ptv[m_pct]
    lo_piv, hi_piv = np.nanpercentile(matriz_full, [PCT_LO, PCT_HI])
    matriz_full = np.clip(matriz_full, lo_piv, hi_piv)

    if len(t_ptv) < 10:
        return None

    dts = np.arange(DT_MIN, DT_MAX + DT_PASO, DT_PASO)
    rms_vals, n_vals = [], []
    for dt in dts:
        rms, n = rms_para_offset(matriz_full, tiempos_full, ss,
                                 t_ptv, s_ptv, v_ptv, dt)
        rms_vals.append(rms)
        n_vals.append(n)
    rms_vals = np.array(rms_vals)

    if np.all(np.isnan(rms_vals)):
        return None

    i_best = np.nanargmin(rms_vals)
    dt_best = float(dts[i_best])
    rms_best = float(rms_vals[i_best])
    i_cero = np.argmin(np.abs(dts))  # dt mas cercano a 0
    rms_cero = float(rms_vals[i_cero])

    en_borde = np.isclose(dt_best, DT_MIN) or np.isclose(dt_best, DT_MAX)
    mejora_pct = (100 * (rms_cero - rms_best) / rms_cero
                 if rms_cero and not np.isnan(rms_cero) else np.nan)

    # Tendencia cerca del borde tocado: si el RMS SIGUE bajando en el ultimo
    # 10% del rango hacia ese borde, el minimo probablemente esta mas alla
    # del rango explorado, y no basta con reportar "toco el borde" -- hay
    # que ampliar el rango antes de concluir nada.
    aun_bajando = False
    if en_borde:
        n_cola = max(3, len(dts) // 10)
        if dt_best < 0:
            cola = rms_vals[:n_cola]
        else:
            cola = rms_vals[-n_cola:]
        cola = cola[~np.isnan(cola)]
        if len(cola) >= 3:
            # x crece seguiendo el sentido de dt ascendente dentro de la cola.
            # Si el minimo real esta mas alla del borde izquierdo (dt_best<0),
            # el RMS deberia CRECER al alejarse del borde (pendiente>0).
            # Si esta mas alla del borde derecho (dt_best>0), el RMS deberia
            # DECRECER al acercarse al borde (pendiente<0).
            x = np.arange(len(cola))
            pendiente = np.polyfit(x, cola, 1)[0]
            aun_bajando = bool(pendiente > 0) if dt_best < 0 else bool(pendiente < 0)

    return {
        "reologia": f"car-{reo}", "concentracion": conc, "zona": zona_key,
        "nombre_grupo": nombre,
        "rms_dt0": round(rms_cero, 4),
        "dt_optimo_s": round(dt_best, 3),
        "rms_dt_optimo": round(rms_best, 4),
        "mejora_pct": round(mejora_pct, 1) if not np.isnan(mejora_pct) else np.nan,
        "n_dt_optimo": int(n_vals[i_best]),
        "optimo_en_borde_rango": bool(en_borde),
        "rms_aun_bajando_en_borde": aun_bajando,
    }


if __name__ == "__main__":
    with open(ETAPAS_JSON, "r", encoding="utf-8") as f:
        etapas = json.load(f)

    filas = []
    for reo in REOS:
        for conc in CONCS:
            for zona_key, prefijo, _ in ZONAS:
                print(f"Calibrando car-{reo}_n-{conc} / {zona_key} ...")
                r = calibrar_condicion(etapas, reo, conc, zona_key, prefijo)
                if r is not None:
                    filas.append(r)
                    borde = "  ⚠ ÓPTIMO EN EL BORDE DEL RANGO — ampliar DT_MIN/DT_MAX" \
                        if r["optimo_en_borde_rango"] else ""
                    print(f"    dt*={r['dt_optimo_s']:+.2f}s  "
                          f"RMS: {r['rms_dt0']:.2f} -> {r['rms_dt_optimo']:.2f} "
                          f"({r['mejora_pct']:.0f}% mejora){borde}")

    if not filas:
        print("\nSin datos suficientes para calibrar (revisar rutas de PTV/cache).")
        raise SystemExit(1)

    df = pd.DataFrame(filas)
    df.to_csv(OUT_CSV, index=False)

    lineas = []
    lineas.append("CALIBRACIÓN DE DESFASE TEMPORAL PIV-PTV")
    lineas.append("=" * 60)
    lineas.append(f"\nRango explorado: [{DT_MIN}, {DT_MAX}] s, paso {DT_PASO} s")
    lineas.append(f"Condiciones calibradas: {len(df)}")

    lineas.append(f"\ndt óptimo: mediana={df.dt_optimo_s.median():+.3f}s  "
                  f"IQR=[{df.dt_optimo_s.quantile(.25):+.3f}, "
                  f"{df.dt_optimo_s.quantile(.75):+.3f}]  "
                  f"rango=[{df.dt_optimo_s.min():+.3f}, {df.dt_optimo_s.max():+.3f}]")
    lineas.append(f"Mejora de RMS con dt óptimo: mediana={df.mejora_pct.median():.0f}%  "
                  f"(vs. dt=0, mismo binning que diferencia_piv_ptv.csv)")
    n_borde = int(df.optimo_en_borde_rango.sum())
    n_sigue_bajando = int(df.get("rms_aun_bajando_en_borde", pd.Series(dtype=bool)).sum())
    if n_borde:
        lineas.append(f"\n⚠ {n_borde} condición(es) con óptimo en el borde del rango "
                      f"explorado [{DT_MIN},{DT_MAX}]s.")
        if n_sigue_bajando:
            lineas.append(
                f"  De ellas, {n_sigue_bajando} muestran el RMS TODAVÍA EN DESCENSO al "
                f"llegar al borde: el verdadero dt* está casi con certeza FUERA del "
                f"rango explorado. NO USAR la 'Lectura' de abajo hasta ampliar "
                f"DT_MIN/DT_MAX y volver a correr — con el rango actual la conclusión "
                f"de 'no hay desfase' sería prematura para esas condiciones.")

    lineas.append("\n--- Lectura ---")
    if n_sigue_bajando > 0:
        lineas.append(
            "NO CONCLUYENTE: hay condiciones cuyo óptimo aún no se alcanza dentro del "
            "rango explorado (ver advertencia arriba). Ampliar el rango antes de "
            "interpretar el resto de esta sección.")
    else:
        dt_disperso = df.dt_optimo_s.std() > 0.5
        mejora_alta = df.mejora_pct.median() > 40
        if mejora_alta and not dt_disperso:
            lineas.append(
                "El dt óptimo es CONSISTENTE entre condiciones y reduce el RMS de forma\n"
                "sustancial: compatible con un desfase de sincronización sistemático,\n"
                "corregible fijando DT_OFFSET a la mediana encontrada.")
        elif mejora_alta and dt_disperso:
            lineas.append(
                "El desfase óptimo reduce el RMS pero varía mucho entre condiciones:\n"
                "no hay un DT_OFFSET único que sirva para todas. Podría deberse a que\n"
                "cada corrida tiene su propio error de t=0 (no un retraso fijo de\n"
                "hardware). No se recomienda fijar un DT_OFFSET global.")
        else:
            lineas.append(
                "El desfase óptimo NO reduce el RMS de forma sustancial: la discrepancia\n"
                "observada en dt=0 no se explica por un problema de sincronización.\n"
                "Debe interpretarse como deslizamiento fibra-fluido genuino y/o\n"
                "variabilidad corrida-a-corrida (el PIV aquí es un promedio de varias\n"
                "tomas de la condición, comparado contra una única corrida de fibras).\n"
                "Esto es un resultado a reportar y discutir, no un error de código.")

    resumen = "\n".join(lineas)
    with open(OUT_RESUMEN, "w", encoding="utf-8") as f:
        f.write(resumen)
    print(f"\n{resumen}")
    print(f"\nGuardado: {OUT_CSV}\nGuardado: {OUT_RESUMEN}")