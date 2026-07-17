# PIV_analisis_carbopol_fibers

Pipeline de análisis de datos PIV/PTV para la memoria ***Fiber Dynamics and Orientation in Ultra-High-Performance Concrete***.

El objetivo experimental es determinar **qué variable del campo de flujo, y en qué etapa del llenado, predice la orientación final de las fibras** en un fluido viscoplástico. Se emplea Carbopol (Herschel-Bulkley) como análogo del UHPC fresco, con índice de refracción igualado, escurriendo por una geometría en L y una viga transparente.

> **Alcance de este repositorio.** Contiene el **análisis posterior** a la adquisición y al procesamiento PIV/PTV inicial. La adquisición y el procesamiento base son trabajo de Lukas Wolff ([`Exp_Func_PIV_PTV`](https://github.com/LukasWolff2002/Exp_Func_PIV_PTV)), sobre el **mismo dataset crudo**. El aporte propio es el pipeline de etapas, polilíneas, zonas, orientación y estadística.

---

## Diseño del análisis: dos reducciones

El pipeline aplica dos reducciones distintas, **en este orden**, que responden preguntas diferentes y no se sustituyen entre sí.

### 1. Polilíneas (1D) — cómo se comporta el fluido

Tres polilíneas de corte fijas (zona L, viga175, viga250). Sobre ellas se extraen $v_\parallel(s,t)$ y $\dot\gamma(s,t)$, se segmentan las etapas (inicio / transición / cuasi-estacionario) con el criterio **V3** de pendiente normalizada, y se contrastan tres casos: reproducibilidad, efecto de la concentración de fibras (0/750/1500/3000) y efecto de la reología (Car-0,2 % vs Car-0,5 %).

Aquí **las fibras son tratamiento, no respuesta**: la orientación no se mide. También se realiza la verificación cruzada PIV–PTV sobre estas polilíneas.

### 2. Zonas (2D) — cómo el fluido interactúa con las fibras

Una polilínea es de medida nula y no contiene fibras, por lo que la orientación solo puede asociarse a regiones de área positiva. El dominio se parte en 8 zonas (Z1 brazo vertical, Z2 codo, Z3 salida, y 6 celdas de viga `Vf{fila}c{col}`).

El campo PIV se agrega por (zona, etapa) → vector de predictores ($|V|$, $\omega$, $\dot\gamma$ en transición y cuasi-estacionario). Las fibras del último fotograma se asignan por centroide y se resumen con el orden-parámetro $S$, la dirección media circular $\bar\theta$ y $\sigma_{\rm iso}$ (fiable si $n_f \ge 5$).

La unidad de observación es el par **(toma, zona)**.

---

## ⚠ Advertencia estadística permanente

**Las zonas de una misma toma no son observaciones independientes.** El número efectivo de réplicas es el **número de corridas**, no el número de filas de la tabla. Todo contraste conducido sobre la tabla completa es **anticonservador** y debe interpretarse como tal.

Con $N \le 8$ zonas por toma, la Capa 1 global es **exploratoria**: detecta señal, no la confirma.

---

## Estructura del pipeline

Los scripts deben ejecutarse en este orden. Cada etapa consume los cachés de la anterior.

```
   PIV_INTERPOLADO/  (datos crudos, externos al repo)
            │
    ┌───────┴────────┐
    ▼                ▼
┌─────────────────────────────┐   ┌──────────────────────────────┐
│ RAMA 1 — POLILÍNEAS (1D)    │   │ RAMA 2 — ZONAS (2D)          │
├─────────────────────────────┤   ├──────────────────────────────┤
│ construir_caches.py         │   │ construir_caches_zonas.py    │
│      → cache_completo/      │   │      → cache_zonas/          │
│            │                │   │            │                 │
│ calcular_etapas_polilinea   │   │ calcular_etapas_zonas.py     │
│      → etapas_polilinea.json│   │      → etapas_zonas.json     │
│            │                │   │            │                 │
│ box_act.py                  │   │ construir_tabla_zonas_todas  │
│ esp_plug_mag.py             │   │      → acum_tabla_zona.csv   │
│ box_piv_ptv.py              │   │            │                 │
│ esp_overlay_piv_ptv.py      │   │ run_real.py / analisis_global│
└─────────────────────────────┘   └──────────────────────────────┘
                                              │
                                  construir_caches_adveccion.py
                                       → cache_adveccion/
                                              │
                                     analizar_adveccion.py
```

### Orden de ejecución

```bash
# 0. Fibras del último fotograma (desde ptv_merged.json)
python exportar_fibras_ultimo_frame.py     # → fibras_ultimo_frame/

# 1. Cachés base (los dos son independientes entre sí)
python construir_caches.py                 # → cache_completo/   (polilíneas)
python construir_caches_zonas.py           # → cache_zonas/      (zonas)

# 2. Segmentación de etapas (criterio V3)
python calcular_etapas_polilinea.py        # → etapas_polilinea.json
python calcular_etapas_zonas.py            # → etapas_zonas.json

# 3a. Análisis I — polilíneas
python box_act.py                          # → Boxplots/
python esp_plug_mag.py                     # → espectrogramas_plug/
python box_piv_ptv.py                      # → Boxplots/Boxplots_PIV_PTV/
python esp_overlay_piv_ptv.py              # → Esp_PIV-PTV/Overlay/

# 3b. Análisis II — zonas
python construir_tabla_zonas_todas.py      # → acum_tabla_zona.csv
python run_real.py                         # → acum_capa{1,2,4}_global.csv

# 4. Retro-advección (opcional; ver advertencias)
python construir_caches_adveccion.py       # → cache_adveccion/
python analizar_adveccion.py
```

---

## Descripción de los módulos

### Construcción de cachés

| Script | Entrada | Salida | Qué hace |
|---|---|---|---|
| `construir_caches.py` | `PIV_INTERPOLADO/` | `cache_completo/*.npz` | Interpola el campo PIV (IDW, 4 vecinos) sobre las 3 polilíneas. Matriz `[n_frames × n_puntos]` + timestamps. |
| `construir_caches_zonas.py` | `PIV_INTERPOLADO/` | `cache_zonas/*.npz` | Cachea los puntos PIV válidos ya procesados: zona asignada + $\dot\gamma$ + $\omega_z$ por KNN. No interpola sobre grilla. |
| `construir_caches_adveccion.py` | `cache_zonas/`, `fibras_ultimo_frame/` | `cache_adveccion/` | Retro-advecta las fibras finales por el campo PIV (RK2) para reconstruir su historia cinemática. |

**Nota de diseño.** `construir_caches_zonas.py` guarda arrays paralelos de los puntos que **realmente existen**, no una grilla rellenada. Un punto sin vector PIV simplemente no está en el archivo. Es la razón por la que la rama de zonas está menos expuesta a artefactos de interpolación que la de polilíneas.

### Segmentación temporal

| Script | Qué hace |
|---|---|
| `detectar_etapas.py` | Implementa el criterio **V3**: suavizado por media móvil ($w$=50 frames para Car-0,2 %, $w$=15 para Car-0,5 %), detección del peak, y umbral de pendiente normalizada $\Gamma(t)$ para el inicio del cuasi-estacionario. |
| `calcular_etapas_polilinea.py` | Aplica V3 sobre la velocidad media de cada polilínea. |
| `calcular_etapas_zonas.py` | Aplica V3 sobre la velocidad media por zona. |
| `utils_etapas.py`, `timeline.py`, `inspeccionar_json.py` | Utilidades de carga, línea de tiempo y diagnóstico de los JSON de etapas. |

### Análisis I — polilíneas

| Script | Qué produce |
|---|---|
| `box_act.py` | Boxplots de $v$ y $\dot\gamma$ por (reología, concentración, zona, etapa). |
| `esp_plug_mag.py` | Espectrogramas $s$–$t$: clasificación plug/no-plug, $Re$, $\mu_{\rm eff}$. |
| `box_piv_ptv.py` | Verificación cruzada PIV vs PTV (boxplots comparados). |
| `esp_overlay_piv_ptv.py` | Superposición de velocidad de fibras (PTV) sobre campo del fluido (PIV). |
| `polilinea_salida_L.py` | Define la geometría de la polilínea de salida de la L (`Polilinea_L/linea_salida.npy`). |

### Análisis II — zonas

| Script | Qué hace |
|---|---|
| `definir_zonas.py` | Geometría de las zonas (Z1/Z2/Z3 por polígono; celdas de viga por bounding box). Test de punto en polígono. |
| `exportar_fibras_ultimo_frame.py` | Extrae la "foto final" de fibras desde `ptv_merged.json`. Reconstruye fibras ausentes del último frame por oclusión transitoria. |
| `construir_tabla_zonas_todas.py` | Acumula todas las corridas → `acum_tabla_zona.csv`, una fila por (toma, zona). |
| `carga_real.py` | Carga PIV + fibras + etapas → DataFrame largo. Agregación robusta: clip al percentil 99 + mediana. |
| `nucleo.py` | Convenciones de zonas, $\theta$ circular, dispersión de centroides, VIF. |
| `analisis_global.py` | **Las capas de análisis en uso.** Unidad de observación: la zona. |
| `run_real.py` | Orquestador del Análisis II. Punto de entrada. |
| `orientacion_objetivo.py` | Definición de la dirección de referencia para $\bar\theta$. |
| `generar_mapas.py` | Mapas de campo por zona y etapa. |
| `analizar_adveccion.py` | Análisis de las trayectorias retro-advectadas. |

### Las capas del análisis estratificado

Implementadas en `analisis_global.py`, con protocolo de decisión fijado **antes** de observar los resultados:

- **Capa 1** — Spearman entre cada predictor (por etapa) y cada respuesta, a través de las zonas. Correlación parcial controlando por $n_f$ para descartar el confundente de conteo.
- **Capa 2** — Regresión multivariada estandarizada. **Condicionada a VIF**: si $\mathrm{VIF}_j \ge 10$ para algún predictor, la capa se descarta como herramienta inferencial (los coeficientes individuales carecen de interpretación).
- **Capa 3** — Random Forest + importancia por permutación. **Condicionada a $R^2_{\rm CV}$**: si $R^2_{\rm CV} \le 0$, la capa se descarta íntegramente.
- **Capa 4** — Comparativa temporal transición vs cuasi-estacionario. Responde la pregunta central de la memoria.

El descarte de una capa **no es un resultado nulo**: documenta que el diseño no sustenta un modelo de esa complejidad.

> **`capas.py` está obsoleto.** Usa la fibra-dentro-de-zona como unidad de observación, lo cual es mal planteado: el PIV entrega un escalar por (zona, etapa), de modo que dentro de una zona el predictor es *constante* y no puede correlacionarse fibra a fibra. Fue reemplazado por `analisis_global.py`. Se conserva por trazabilidad; **no debe usarse**.

---

## Requisitos

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

`construir_caches_adveccion.py` importa además `win10toast` (notificaciones, **solo Windows**). Si se ejecuta en Linux/macOS, comentar ese import.

## Datos de entrada

Los datos crudos **no están versionados**. Se esperan en `../PIV_INTERPOLADO/` (fuera del repo), con una carpeta por corrida y archivos `.txt` por frame:

```
x  y  u  v  mag  valid        # cabecera con "timestamp_s:"
```

Convención de nombres: `m{mezcla}-toma-{n}-n-{concentración}-car-{reología}`
— p. ej. `m74-toma-1-n-0750-car-02`.

Correcciones aplicadas al leer: `y *= -1`, `v *= -1`.

---

## Notas de calidad de datos

Estas decisiones existen porque su ausencia **falseaba los resultados**. Documentadas aquí para que no se reviertan por accidente.

### Relleno acotado de huecos (`MAX_HUECO_INTERP = 3`)

La interpolación lineal 1D sobre las polilíneas está limitada a huecos **internos** de como máximo 3 puntos consecutivos ($\lesssim$ 2,2 mm, con $\Delta s \approx$ 0,75 mm). Tramos mayores conservan sus NaN.

Rellenar huecos largos inventa un perfil de velocidad, y $\dot\gamma = dv/ds$ amplifica cualquier rampa artificial. Los NaN de **borde** tampoco se rellenan: `np.interp` extrapola plano, generando $\dot\gamma \approx 0$ espurio justo en las paredes. Si se requiere $v = 0$ en una pared, debe imponerse como condición de contorno explícita, no colarse por la interpolación.

### Contaminación por suavizado (`contaminados()`)

Anular los NaN **después** de suavizar llega tarde: el kernel ya propagó los valores rellenados a los puntos válidos vecinos. La máscara de NaN se dilata en $w/2 + 1$ puntos antes de invalidar. Afecta a `box_act.py` y `esp_plug_mag.py` (donde $\dot\gamma$ alimenta la clasificación plug).

### Límites de distancia en interpoladores espaciales

- `construir_caches_zonas.py` — `DIST_MAX_KNN_MM = 5.0`. Sin este límite, un punto en el borde ajusta su gradiente contra vectores al otro lado de un vacío.
- `construir_caches_adveccion.py` — `DIST_MAX_NN_MM = 3.0`. `NearestNDInterpolator` sin límite extrapola indefinidamente. En un integrador el error **se compone**: la velocidad inventada actualiza la posición, y la trayectoria diverge sin recuperación. Las partículas que salen del material se marcan como perdidas y su trayectoria se trunca; el script reporta el porcentaje de supervivencia.

> **Los umbrales de distancia (3 mm, 5 mm) son provisionales**, fijados por argumento dimensional desde $\Delta y_{\rm PIV}$ (0,75–2,05 mm según cámara y reología). No están validados empíricamente. Si la supervivencia en retro-advección es baja, eso no indica un umbral mal puesto sino que la cobertura PIV no sustenta el método.

### Dependencia instrumental del criterio plug

La fracción de área clasificada como plug **depende de la configuración PIV** ($\delta\dot\gamma$ por cámara y reología) y no es una propiedad intrínseca del fluido. Toda comparación entre zonas o reologías exige igualdad de configuración o el reporte explícito del umbral aplicado.

---

## Limitaciones del diseño

- **Independencia.** Ver la advertencia estadística arriba.
- **Naturaleza de la medición de fibras.** La orientación se registra sobre el **estado final depositado**, mientras que los predictores describen etapas anteriores. La asociación es estadística, no causal: no permite trazar la trayectoria de una fibra individual.
- **Separabilidad de los predictores.** $\omega$ y $\dot\gamma$ son las partes antisimétrica y simétrica del **mismo tensor gradiente**. Su colinealidad es intrínseca al diseño y no se elimina con tratamiento estadístico posterior.
- **Cobertura espacial.** La distribución de fibras entre zonas es marcadamente heterogénea. `Vf1c3` no registra fibra alguna en ninguna corrida; `Vf1c2` tiene una mediana de 2 fibras, bajo el criterio de fiabilidad $n_f \ge 5$.

---

## Salidas versionadas

| Archivo | Contenido |
|---|---|
| `acum_tabla_zona.csv` | Tabla acumulada (toma, zona) — la materia prima del Análisis II. |
| `acum_capa{1,2,4}_global.csv` | Resultados de cada capa. |
| `atribucion_E1_E2_E3.csv` | Atribución por etapa. |
| `resumen_concordancia.csv` | Concordancia PIV–PTV. |
| `etapas_polilinea.json`, `etapas_zonas.json` | Segmentación V3 por corrida. |
| `Boxplots/`, `Esp_PIV-PTV/`, `figs_memoria/` | Figuras. |

Los cachés (`cache_completo/`, `cache_zonas/`, `cache_adveccion/`) **no se versionan**: se regeneran desde los datos crudos.

---

## Pendientes conocidos

- `utils_etapas.py:30` tiene una ruta absoluta hardcodeada (`C:/Users/elisa/Desktop/PIV_INTERPOLADO`) que rompe la portabilidad.
- No hay `.gitignore`; conviene añadir uno que excluya `cache_*/`, `__pycache__/` y `*.npz`.
- `RECALCULO = True` está fijado en varios scripts de caché: reconstruye siempre, aunque el caché exista.
- `capas.py` obsoleto (ver arriba).

---

## Referencias

- Wolff, L. — [`Exp_Func_PIV_PTV`](https://github.com/LukasWolff2002/Exp_Func_PIV_PTV): adquisición y procesamiento PIV/PTV base.
- Jeffery, G. B. (1922). The motion of ellipsoidal particles immersed in a viscous fluid.
- Westerweel, J. (1997). Fundamentals of digital particle image velocimetry.
- Teng, L. et al. (2021). Flow-induced fibre orientation in UHPC.
