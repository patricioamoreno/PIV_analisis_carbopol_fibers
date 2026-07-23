"""
esp_overlay_piv_ptv.py
======================
Overlay PIV (campo promediado = BASE/fondo) + PTV (puntos, encima).

Filosofia de la figura:
  - El espectrograma PIV es la BASE continua: cubre todos los instantes.
  - El PTV se dibuja SOLO donde hay observacion real. Los huecos del PTV
    son informacion, no se rellenan.
  - PIV se muestra atenuado (alpha < 1) para que los puntos PTV resalten.

Entradas PTV (formato npz, una por zona, mismo esquema de nombres que el PIV):
  zona L        -> datos_esp_ptv/car-{reo}_n-{conc}_ptv_datos_suavizados.npz
  zona viga175  -> datos_esp_ptv/viga175_car-{reo}_n-{conc}_ptv_datos_suavizados.npz
  zona viga250  -> datos_esp_ptv/viga250_car-{reo}_n-{conc}_ptv_datos_suavizados.npz
Columnas del npz: t, s, v_paralela, track_id, x_mm, y_mm, vx_mm_s, vy_mm_s.
El caso SIN FIBRA no se estudia (no hay PTV para el).

Genera overlay + diferencia para las TRES polilineas y TODAS las muestras
(combinaciones reo x conc) configuradas.
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from construir_caches import (
    cargar_cache_completo, natural_sort_key, nombre_base_carpeta,
    ss_linea_L, ss_viga,
)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACION
# ============================================================

ETAPAS_JSON = "etapas_polilinea.json"
CACHE_DIR   = "cache_completo"
OUTPUT_DIR  = "Esp_PIV-PTV/Overlay"
PTV_DIR     = "datos_esp_ptv"

# --- Muestras a procesar: todas las combinaciones reo x conc ---
# (sin fibra no se incluye: no hay datos PTV)
REOS  = ["02", "05"]
CONCS = ["0750", "1500", "3000"]

# --- Polilineas a procesar (zona_key, prefijo, etiqueta) ---
ZONAS = [
    ("L",       "",        "Salida de la L  (canal cerrado)"),
    ("viga175", "viga175", "Viga 175 mm"),
    ("viga250", "viga250", "Viga 250 mm"),
]

# --- Incertidumbre de la medición PIV, δU [mm/s] ─────────────
# Valores de la tabla de incertidumbres del Capítulo 3 (calculados con
# ε_corr = 0.05 px), por zona y reología. La zona L se mide con la Cámara 4
# (ventana más fina, mayor δU); las vigas con las Cámaras 1-3.
#
# Su papel aquí: el criterio de la memoria establece que las fibras siguen al
# fluido si la diferencia |V_PIV - V_PTV| se mantiene dentro de la
# incertidumbre de la propia medición. Una diferencia por debajo de δU no es
# evidencia de deslizamiento entre fases: es indistinguible del ruido del
# instrumento. Sin este umbral, el RMS es un número sin escala de referencia.
DELTA_U = {
    'L':       {'02': 3.08, '05': 3.08},
    'viga175': {'02': 1.41, '05': 1.41},
    'viga250': {'02': 1.38, '05': 1.38},
}

DT_OFFSET = 0.0

# --- Aspecto del overlay ---
PIV_ALPHA   = 0.8
PTV_SIZE    = 18
PTV_EDGE_LW = 0.15
DIBUJAR_TRAYECTORIAS = False

DPI = 160
PCT_LO, PCT_HI = 5, 95

# ============================================================
# CARGA PTV  (formato npz)
# ============================================================

def ruta_ptv(reo, conc, prefijo):
    """Ruta al npz de PTV de la zona, con el mismo esquema de nombres del PIV."""
    base = f"car-{reo}_n-{conc}_ptv_datos_suavizados.npz"
    nombre = f"{prefijo}_{base}" if prefijo else base
    return os.path.join(PTV_DIR, nombre)


def cargar_ptv(path):
    """
    Lee un npz de PTV con columnas:
      t, s, v_paralela, track_id, x_mm, y_mm, vx_mm_s, vy_mm_s
    Devuelve (t, s, v, tid) donde v = |V| = hypot(vx, vy).
    """
    data = np.load(path, allow_pickle=True)
    t  = np.asarray(data['t'], dtype=float)
    s  = np.asarray(data['s'], dtype=float)
    vx = np.asarray(data['vx_mm_s'], dtype=float)
    vy = np.asarray(data['vy_mm_s'], dtype=float)
    v  = np.hypot(vx, vy)
    tid = np.asarray(data['track_id']) if 'track_id' in data.files else None
    return t, s, v, tid


# ============================================================
# CAMPO PIV PROMEDIADO
# ============================================================

def construir_campo_promedio(etapas, reo, conc, prefijo, zona_key):
    nombre_grupo = f"car-{reo}_n-{conc}"
    carpetas = set()
    for clave in etapas.keys():
        carpeta = nombre_base_carpeta(clave)
        if re.search(rf'car-{reo}', carpeta) and re.search(rf'n-{conc}', carpeta):
            carpetas.add(carpeta)
    carpetas = sorted(carpetas, key=natural_sort_key)

    tomas_mat, tomas_tiem, tiempos_list = [], [], None
    for carpeta in carpetas:
        matriz_full, tiempos_full = cargar_cache_completo(carpeta, prefijo, CACHE_DIR, usar_magnitud=True)
        if matriz_full is None:
            print(f"  ⚠ Sin cache: {carpeta}"); continue
        clave = f"{carpeta}_{zona_key}"
        if clave not in etapas:
            print(f"  ⚠ Sin etapas: {clave}"); continue

        etapas_mat, etapas_tim, fronteras = [], [], []
        for etapa in ['inicio', 'transicion', 'cuasi']:
            i0, i1 = etapas[clave]['etapas'][etapa]
            i1 = min(i1, matriz_full.shape[0])
            if i0 >= i1:
                continue
            etapas_mat.append(matriz_full[i0:i1])
            etapas_tim.append(tiempos_full[i0:i1])
            fronteras.append(tiempos_full[i0:i1])
        if not etapas_mat:
            continue
        tomas_mat.append(np.concatenate(etapas_mat, axis=0))
        tomas_tiem.append(np.concatenate(etapas_tim, axis=0))
        if tiempos_list is None:
            tiempos_list = fronteras

    if not tomas_mat:
        raise SystemExit(f"Sin datos PIV para {nombre_grupo} zona {zona_key}")

    if len({m.shape for m in tomas_mat}) > 1:
        n_min = min(m.shape[0] for m in tomas_mat)
        tomas_mat  = [m[:n_min] for m in tomas_mat]
        tomas_tiem = [t[:n_min] for t in tomas_tiem]

    stack        = np.stack(tomas_mat, axis=0)
    matriz_full  = np.nanmean(stack, axis=0)
    tiempos_full = tomas_tiem[0]
    print(f"  {nombre_grupo}: {stack.shape[0]} toma(s) → campo medio {matriz_full.shape}")
    return matriz_full, tiempos_full, nombre_grupo, tiempos_list


# ============================================================
# COLORMAP COMPARTIDO
# ============================================================

def construir_cmap_y_norma(matriz_vel):
    v_min    = np.nanpercentile(matriz_vel, 5)
    v_max    = np.nanpercentile(matriz_vel, 95)
    v_median = np.nanmedian(matriz_vel)
    frac   = (v_median - v_min) / max(v_max - v_min, 1e-10)
    n_gris = max(1, int(frac * 256))
    n_rojo = max(1, 256 - n_gris)
    colores = (
        [plt.cm.Greys(x) for x in np.linspace(0.8, 0.2, n_gris)] +
        [plt.cm.Reds(x)  for x in np.linspace(0.2, 1.0, n_rojo)]
    )
    return LinearSegmentedColormap.from_list('gris_rojo', colores), v_min, v_max

def _agregar_divisores(ax, tiempos_list, color='white'):
    for tl in tiempos_list[:-1]:
        ax.axvline(tl[-1], color=color, lw=1.5, ls='--', alpha=0.8)

def mascara_ptv_sobre_piv(t_ptv, s_ptv, tiempos, ss, matriz_vel):
    """
    Devuelve un booleano por cada punto PTV: True si la celda PIV más cercana
    (t, s) tiene lectura válida (no NaN). Los puntos PTV que caen donde el PIV
    es NaN (sin material en ese instante/posición) se consideran artefactos de
    tracking y se descartan.

    matriz_vel tiene forma (N_t, N_s) con NaN donde no hubo medición PIV.
    """
    tiempos = np.asarray(tiempos, dtype=float)
    ss      = np.asarray(ss, dtype=float)

    # Índice de la celda PIV más cercana a cada punto PTV (vecino más cercano).
    it = np.searchsorted(tiempos, t_ptv)
    it = np.clip(it, 1, len(tiempos) - 1)
    # elegir el borde más cercano entre it-1 e it
    izq = (t_ptv - tiempos[it - 1]) <= (tiempos[it] - t_ptv)
    it  = np.where(izq, it - 1, it)

    js = np.searchsorted(ss, s_ptv)
    js = np.clip(js, 1, len(ss) - 1)
    izq_s = (s_ptv - ss[js - 1]) <= (ss[js] - s_ptv)
    js    = np.where(izq_s, js - 1, js)

    valido = np.isfinite(matriz_vel[it, js])
    return valido

# ============================================================
# OVERLAY:  PIV base (difuminado) + PTV solo donde hay dato
# ============================================================

def graficar_overlay(matriz_vel, tiempos, tiempos_list, ss, t_ptv, s_ptv, v_ptv, tid_ptv,
                     cmap, vmin, vmax, nombre, zona_label, output_path):
    fig, ax = plt.subplots(figsize=(15, 6))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    pcm = ax.pcolormesh(tiempos, ss, matriz_vel.T,
                        cmap=cmap, norm=norm, shading='auto', alpha=PIV_ALPHA)

    if DIBUJAR_TRAYECTORIAS and tid_ptv is not None:
        for uid in np.unique(tid_ptv):
            m = tid_ptv == uid
            if m.sum() < 2:
                continue
            orden = np.argsort(t_ptv[m])
            ax.plot(t_ptv[m][orden], s_ptv[m][orden], '-', lw=0.6, alpha=0.45,
                    color=cmap(norm(np.nanmean(v_ptv[m]))), zorder=3)
    ax.scatter(t_ptv, s_ptv, c=v_ptv, cmap=cmap, norm=norm,
               s=PTV_SIZE, edgecolors='black', linewidths=PTV_EDGE_LW,
               alpha=0.75, zorder=4)

    cax = ax.inset_axes([0, 1.04, 1, 0.04])
    fig.colorbar(pcm, cax=cax, orientation='horizontal',
                 label='|V| [mm/s]   (fondo PIV = base,  puntos = PTV)')
    cax.xaxis.set_ticks_position('top'); cax.xaxis.set_label_position('top')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Posicion s [mm]')
    ax.set_ylim(ss[0], ss[-1]); ax.set_xlim(tiempos[0], tiempos[-1])
    # Título omitido: descripción completa en el caption de la memoria (figura tipo).
    # ax.set_title(f'{nombre}  —  {zona_label}\n'
    #              f'PIV (base) + PTV (puntos donde hay medicion)',
    #              fontsize=11, fontweight='bold')
    _agregar_divisores(ax, tiempos_list, color='white')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  OK -> {os.path.basename(output_path)}")


def graficar_diferencia(matriz_vel, tiempos, ss, t_ptv, s_ptv, v_ptv,
                        nombre, zona_label, output_path, delta_u=None):
    """
    Grafica el campo de diferencias PIV-PTV y devuelve sus métricas
    cuantitativas (RMS, sesgo, cobertura), para permitir su exportación a
    CSV en main() y su cita directa en la memoria (Sección res_overlay).

    Si se entrega `delta_u` (incertidumbre de la medición PIV en mm/s), se
    evalúa además el criterio de la memoria: qué fracción de las celdas
    comparadas presenta |V_PIV - V_PTV| < δU. Una diferencia por debajo del
    umbral de detección del instrumento no constituye evidencia de
    deslizamiento entre fases; el RMS por sí solo no permite esa lectura,
    pues carece de escala de referencia.
    """
    t_edges = np.concatenate([tiempos, [tiempos[-1] + (tiempos[-1]-tiempos[-2])]])
    s_edges = np.concatenate([ss, [ss[-1] + (ss[-1]-ss[-2])]])
    suma, _, _   = np.histogram2d(t_ptv, s_ptv, bins=[t_edges, s_edges], weights=v_ptv)
    cuenta, _, _ = np.histogram2d(t_ptv, s_ptv, bins=[t_edges, s_edges])
    with np.errstate(invalid='ignore'):
        v_ptv_grid = np.where(cuenta > 0, suma / cuenta, np.nan)
    diff = matriz_vel - v_ptv_grid
    lim  = np.nanpercentile(np.abs(diff), 95)
    fig, ax = plt.subplots(figsize=(15, 6))
    pcm = ax.pcolormesh(tiempos, ss, diff.T, cmap='RdBu_r',
                        shading='auto', vmin=-lim, vmax=lim)
    cax = ax.inset_axes([0, 1.04, 1, 0.04])
    cb = fig.colorbar(pcm, cax=cax, orientation='horizontal',
                      label='|V|_PIV - |V|_PTV [mm/s]   (blanco = sin dato PTV o acuerdo)')
    # Marcar la banda ±δU sobre la barra de color: todo lo que cae dentro es
    # indistinguible del ruido del instrumento y no evidencia deslizamiento.
    if delta_u is not None and delta_u > 0 and delta_u < lim:
        for signo in (-1, 1):
            cax.axvline(signo * delta_u, color='k', lw=1.1, ls='--', alpha=0.9)
        cax.text(0, 1.6, f'±$\\delta U$ = {delta_u:.2f} mm/s',
                 transform=cax.transAxes, ha='left', va='bottom', fontsize=8)
    cax.xaxis.set_ticks_position('top'); cax.xaxis.set_label_position('top')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Posicion s [mm]')
    ax.set_ylim(ss[0], ss[-1]); ax.set_xlim(tiempos[0], tiempos[-1])
    # Título omitido: descripción en el caption de la memoria (figura tipo).
    # ax.set_title(f'{nombre}  —  {zona_label}\nDiferencia PIV - PTV',
    #              fontsize=11, fontweight='bold')
    valido = ~np.isnan(diff)
    metricas = {
        "n_celdas_comparadas": int(valido.sum()),
        "n_celdas_totales": int(diff.size),
        "cobertura_pct": round(100 * valido.sum() / diff.size, 2) if diff.size else np.nan,
        "rms_mm_s": np.nan, "sesgo_mm_s": np.nan,
        "mediana_abs_diff_mm_s": np.nan, "p95_abs_diff_mm_s": np.nan,
        # Criterio de la memoria: comparación contra la incertidumbre del PIV
        "delta_u_mm_s": delta_u if delta_u is not None else np.nan,
        "frac_dentro_delta_u_pct": np.nan,
        "rms_sobre_delta_u": np.nan,
        "sesgo_sobre_delta_u": np.nan,
        "veredicto": "sin evaluar",
    }
    if valido.any():
        rms  = np.sqrt(np.nanmean(diff[valido]**2))
        bias = np.nanmean(diff[valido])
        metricas.update({
            "rms_mm_s": round(float(rms), 4),
            "sesgo_mm_s": round(float(bias), 4),
            "mediana_abs_diff_mm_s": round(float(np.nanmedian(np.abs(diff[valido]))), 4),
            "p95_abs_diff_mm_s": round(float(lim), 4),
        })

        # ── Criterio δU: ¿la discrepancia es distinguible del ruido? ──
        if delta_u is not None and delta_u > 0:
            dentro = np.abs(diff[valido]) < delta_u
            frac = 100.0 * dentro.sum() / dentro.size
            metricas.update({
                "frac_dentro_delta_u_pct": round(float(frac), 2),
                "rms_sobre_delta_u": round(float(rms / delta_u), 3),
                "sesgo_sobre_delta_u": round(float(bias / delta_u), 3),
            })
            # Lectura automática, para no dejar la interpretación al ojo.
            # El sesgo importa más que el RMS: un sesgo sistemático por encima
            # de δU indica deslizamiento entre fases; un RMS alto con sesgo
            # nulo indica dispersión, no deriva sistemática.
            if abs(bias) < delta_u and rms < delta_u:
                metricas["veredicto"] = "acuerdo (RMS y sesgo < delta_U)"
            elif abs(bias) < delta_u:
                metricas["veredicto"] = "sesgo nulo, dispersion > delta_U"
            else:
                metricas["veredicto"] = "SESGO SISTEMATICO > delta_U"

        # Anotación sobre la figura, con la referencia de escala incluida.
        txt = f'RMS={rms:.1f} mm/s   sesgo={bias:+.1f} mm/s'
        if delta_u is not None and delta_u > 0:
            txt += (f'\n$\\delta U$={delta_u:.2f} mm/s   '
                    f'|diff|<$\\delta U$ en {metricas["frac_dentro_delta_u_pct"]:.0f}% '
                    f'de las celdas')
        ax.text(0.99, 0.02, txt,
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    out2 = output_path.replace('Overlay', 'Diferencia').replace('overlay', 'diferencia')
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    plt.savefig(out2, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    print(f"  OK -> {os.path.basename(out2)}")
    return metricas


# ============================================================
# PROCESAR UNA MUESTRA EN UNA ZONA
# ============================================================

def procesar(etapas, reo, conc, zona_key, prefijo, zona_label, filas_diff=None):
    print(f"\n=== car-{reo}_n-{conc}  |  zona {zona_key} ===")

    ptv_path = ruta_ptv(reo, conc, prefijo)
    if not os.path.exists(ptv_path):
        print(f"  ⚠ Sin PTV: {ptv_path} (omitida)")
        return

    matriz_full, tiempos_full, nombre, tiempos_list = construir_campo_promedio(
        etapas, reo, conc, prefijo, zona_key)

    ss = ss_linea_L() if prefijo == "" else ss_viga()
    if len(ss) != matriz_full.shape[1]:
        ss = np.linspace(ss[0], ss[-1], matriz_full.shape[1])

    t_ptv, s_ptv, v_ptv, tid_ptv = cargar_ptv(ptv_path)
    t_ptv = t_ptv + DT_OFFSET

    print("  [Verificacion de ejes]")
    print(f"    PIV t: [{tiempos_full[0]:.3f}, {tiempos_full[-1]:.3f}] s, {len(tiempos_full)} instantes")
    print(f"    PTV t: [{t_ptv.min():.3f}, {t_ptv.max():.3f}] s, {len(np.unique(t_ptv))} instantes unicos")
    print(f"    PIV s: [{ss[0]:.2f}, {ss[-1]:.2f}] mm   PTV s: [{s_ptv.min():.2f}, {s_ptv.max():.2f}] mm")

    # Filtro 1: ventana temporal del PIV.
    m_t = (t_ptv >= tiempos_full[0]) & (t_ptv <= tiempos_full[-1])
    n_fuera_t = int((~m_t).sum())
    if n_fuera_t:
        print(f"    {n_fuera_t} obs PTV fuera de la ventana temporal PIV (descartadas)")
    t_ptv, s_ptv, v_ptv = t_ptv[m_t], s_ptv[m_t], v_ptv[m_t]
    tid_ptv = tid_ptv[m_t] if tid_ptv is not None else None

    # Filtro 2: la celda PIV correspondiente debe tener lectura (no NaN).
    # Un punto PTV donde el PIV es NaN (sin material allí en ese instante) es
    # un artefacto de tracking y no se dibuja.
    if len(t_ptv):
        m_piv = mascara_ptv_sobre_piv(t_ptv, s_ptv, tiempos_full, ss, matriz_full)
        n_fuera_piv = int((~m_piv).sum())
        if n_fuera_piv:
            print(f"    {n_fuera_piv} obs PTV sobre celdas PIV=NaN (descartadas)")
        t_ptv, s_ptv, v_ptv = t_ptv[m_piv], s_ptv[m_piv], v_ptv[m_piv]
        tid_ptv = tid_ptv[m_piv] if tid_ptv is not None else None
    print(f"    PTV dibujado: {len(t_ptv)} obs")
    if len(v_ptv):
        lo, hi = np.nanpercentile(v_ptv, [PCT_LO, PCT_HI])
        m_pct = (v_ptv >= lo) & (v_ptv <= hi)
        t_ptv, s_ptv, v_ptv = t_ptv[m_pct], s_ptv[m_pct], v_ptv[m_pct]
        tid_ptv = tid_ptv[m_pct] if tid_ptv is not None else None
        print(f"    PTV tras recorte P{PCT_LO}-P{PCT_HI}: {len(t_ptv)} obs")

    lo_piv, hi_piv = np.nanpercentile(matriz_full, [PCT_LO, PCT_HI])
    matriz_full = np.clip(matriz_full, lo_piv, hi_piv)
    cmap, vmin, vmax = construir_cmap_y_norma(matriz_full)

    sufijo = zona_key  # L / viga175 / viga250
    ruta = os.path.join(OUTPUT_DIR, zona_key)
    os.makedirs(ruta, exist_ok=True)
    out = os.path.join(ruta, f"overlay_{nombre}_{sufijo}.png")
    graficar_overlay(matriz_full, tiempos_full, tiempos_list, ss, t_ptv, s_ptv, v_ptv,
                     tid_ptv, cmap, vmin, vmax, nombre, zona_label, out)
    # δU de esta zona y reología (Tabla de incertidumbres, Capítulo 3).
    delta_u = DELTA_U.get(zona_key, {}).get(reo)
    if delta_u is None:
        print(f"  ⚠ Sin δU definido para zona {zona_key} / car-{reo}: "
              f"el criterio de acuerdo no se evaluará.")

    metricas = graficar_diferencia(matriz_full, tiempos_full, ss, t_ptv, s_ptv, v_ptv,
                                   nombre, zona_label, out, delta_u=delta_u)
    if metricas.get("veredicto", "sin evaluar") != "sin evaluar":
        print(f"    δU={delta_u:.2f} mm/s → "
              f"{metricas['frac_dentro_delta_u_pct']:.0f}% de celdas dentro   "
              f"[{metricas['veredicto']}]")
    if filas_diff is not None:
        filas_diff.append({
            "reologia": f"car-{reo}", "concentracion": conc,
            "zona": zona_key, "nombre_grupo": nombre, **metricas,
        })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ETAPAS_JSON, 'r', encoding='utf-8') as f:
        etapas = json.load(f)

    # Acumulador de las métricas cuantitativas del campo de diferencias
    # PIV-PTV (RMS, sesgo, cobertura). Se vuelca a CSV al final para citar
    # en la memoria (Sección Resultados — verificación cruzada PIV-PTV).
    filas_diff = []

    for reo in REOS:
        for conc in CONCS:
            for zona_key, prefijo, zona_label in ZONAS:
                try:
                    procesar(etapas, reo, conc, zona_key, prefijo, zona_label,
                            filas_diff=filas_diff)
                except SystemExit as e:
                    print(f"  ⚠ {e}")
                except Exception as e:
                    print(f"  ⚠ Error en car-{reo}_n-{conc} zona {zona_key}: {e}")

    if filas_diff:
        df_diff = pd.DataFrame(filas_diff)
        csv_diff = os.path.join(OUTPUT_DIR, "diferencia_piv_ptv.csv")
        df_diff.to_csv(csv_diff, index=False)
        print(f"\n✅ Métricas de diferencia PIV-PTV guardadas: {csv_diff}")
        print(df_diff.to_string(index=False))

        # ── Veredicto global: la frase que va a la memoria ──────────
        ev = df_diff[df_diff["veredicto"] != "sin evaluar"]
        if len(ev):
            print("\n" + "=" * 64)
            print("CRITERIO δU  —  ¿las fibras siguen al fluido?")
            print("=" * 64)
            n_ok = int((ev["veredicto"].str.startswith("acuerdo")).sum())
            print(f"  casos evaluados:              {len(ev)}")
            print(f"  con RMS y sesgo < δU:         {n_ok}/{len(ev)}")
            print(f"  celdas dentro de δU (media):  "
                  f"{ev['frac_dentro_delta_u_pct'].mean():.0f}%  "
                  f"(rango {ev['frac_dentro_delta_u_pct'].min():.0f}"
                  f"-{ev['frac_dentro_delta_u_pct'].max():.0f}%)")
            print(f"  |sesgo|/δU  mediana:          "
                  f"{ev['sesgo_sobre_delta_u'].abs().median():.2f}")
            print(f"  RMS/δU      mediana:          "
                  f"{ev['rms_sobre_delta_u'].median():.2f}")
            malos = ev[ev["veredicto"].str.contains("SESGO")]
            if len(malos):
                print(f"\n  ⚠ {len(malos)} caso(s) con sesgo sistemático > δU "
                      f"(posible deslizamiento entre fases):")
                for r in malos.itertuples():
                    print(f"      {r.nombre_grupo} / {r.zona}: "
                          f"sesgo={r.sesgo_mm_s:+.2f} mm/s "
                          f"({r.sesgo_sobre_delta_u:+.2f}·δU)")
                print("  → Reportar explícitamente en la memoria.")
            else:
                print("\n  ✔ Ningún caso presenta sesgo sistemático superior a δU.")
                print("  → El supuesto de que las fibras son transportadas por "
                      "el fluido se sostiene.")

    print(f"\nListo. Resultados en: {OUTPUT_DIR}")