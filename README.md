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

Una polilínea es de medida nula y no contiene fibras, por lo que la orientación solo puede asociarse a regiones de área positiva. El dominio se parte en 8 zonas (Z1 brazo vertical, Z2 codo, Z3 salida, y celdas de viga `Vf{fila}c{col}`).

El campo PIV se agrega por (zona, etapa) → vector de predictores ($|V|$, $\omega$, $\dot\gamma$ en transición y cuasi-estacionario). Las fibras del último fotograma se asignan por **posición final** y se resumen con el orden-parámetro $S$, la dirección media circular $\bar\theta$ y $\sigma_{\rm iso}$ (fiable si $n_f \ge 5$).

La unidad de observación es el par **(toma, zona)**.

---

## ⚠ Advertencias estadísticas permanentes

**1. Las zonas de una misma toma no son observaciones independientes.** El número efectivo de réplicas es el **número de corridas**, no el número de filas de la tabla. Todo contraste sobre la tabla completa es **anticonservador**.

**2. Los puntos PIV de una misma zona tampoco son independientes.** La ventana de interrogación impone una longitud de correlación espacial: vectores vecinos comparten información. El $n$ efectivo es órdenes de magnitud menor que el recuento bruto.

**3. En consecuencia, el $p$-valor no discrimina.** Con $10^4$–$10^6$ puntos por celda, Mann-Whitney declara significativa casi cualquier diferencia (~97 % de las comparaciones de este proyecto), y más de un tercio de ellas tiene efecto insignificante. **Toda decisión de exclusión usa el $d$ de Cohen, nunca el $p$-valor.** Ver `criterio_exclusion.py`.

---

## Orden de ejecución

```bash
# 0. Fibras del último fotograma (desde ptv_merged.json)
python exportar_fibras_ultimo_frame.py     # → fibras_ultimo_frame/

# 1. Cachés base (independientes entre sí)
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

# 3b. Comparativa por zona (genera los tests que alimentan la exclusión)
python analisis.py                         # → Analisis_COMPARATIVA_zonas/

# 3c. Análisis II — zonas
python construir_tabla_zonas_todas.py      # → acum_tabla_zona.csv (+ sin_excluir)
python run_real.py                         # → acum_capa{1,2,4}_global.csv
python generar_mapas.py                    # → figs_memoria/
python verificacion_estratificada.py       # chequeo manual (opcional)

# 4. Validación de la atribución de zona (opcional)
python reconstruccion_lagrangiana.py       # → atribucion_E1_E2_E3.csv
```

**Dependencia importante:** `construir_tabla_zonas_todas.py` lee los CSV que produce `analisis.py`. Si `Analisis_COMPARATIVA_zonas/` no existe, la exclusión no se aplica y el script avisa.

---

## Módulos

### Cachés

| Script | Salida | Qué hace |
|---|---|---|
| `construir_caches.py` | `cache_completo/*.npz` | Interpola el campo PIV (IDW, 4 vecinos) sobre las 3 polilíneas. Matriz `[n_frames × n_puntos]`. |
| `construir_caches_zonas.py` | `cache_zonas/*.npz` | Guarda los puntos PIV válidos con zona asignada + $\dot\gamma$ + $\omega_z$ por KNN acotado. |

**Nota de diseño.** `construir_caches_zonas.py` guarda arrays paralelos de los puntos que **realmente existen**, no una grilla rellenada. Un punto sin vector PIV simplemente no está. Por eso la rama de zonas está menos expuesta a artefactos de interpolación que la de polilíneas.

### Etapas

| Script | Qué hace |
|---|---|
| `detectar_etapas.py` | Criterio **V3**: suavizado ($w$=50 para Car-0,2 %, $w$=15 para Car-0,5 %), detección del peak, umbral de pendiente normalizada $\Gamma(t)$. |
| `calcular_etapas_polilinea.py` | V3 sobre la velocidad media de cada polilínea. |
| `calcular_etapas_zonas.py` | V3 sobre la velocidad **promedio** de cada zona (mínimo 3 vectores por fotograma). Calcula `t_peak` y `t_quasi`; el Análisis II solo usa el segundo. |
| `utils_etapas.py`, `timeline.py`, `inspeccionar_json.py` | Utilidades y diagnóstico de los JSON. |

### Análisis I — polilíneas

| Script | Qué produce |
|---|---|
| `box_act.py` | Boxplots de $v$ y $\dot\gamma$ por (reología, concentración, zona, etapa). |
| `esp_plug_mag.py` | Espectrogramas $s$–$t$: clasificación plug/no-plug, $Re$, $\mu_{\rm eff}$. |
| `box_piv_ptv.py` | Verificación cruzada PIV vs PTV. |
| `esp_overlay_piv_ptv.py` | Velocidad de fibras (PTV) sobre campo del fluido (PIV). |
| `polilinea_salida_L.py` | Geometría de la polilínea de salida de la L. |

### Análisis II — zonas

| Script | Qué hace |
|---|---|
| `definir_zonas.py` | Geometría de zonas (`asignar_zona`). |
| `exportar_fibras_ultimo_frame.py` | "Foto final" de fibras desde `ptv_merged.json`. |
| `analisis.py` | **Comparativa por zona.** Boxplots + tests (toma vs base, entre tomas) por (zona, etapa) → `Analisis_COMPARATIVA_zonas/`. Sustituyó a la versión por polilínea del mismo script. |
| `criterio_exclusion.py` | **Criterio de exclusión de celdas.** Ver abajo. |
| `construir_tabla_zonas_todas.py` | Acumula todas las corridas y aplica la exclusión → `acum_tabla_zona.csv`. |
| `carga_real.py` | Carga PIV + fibras + etapas → DataFrame largo. Clip al percentil 99 + mediana. |
| `nucleo.py` | Convenciones de zonas, $\theta$ circular, dispersión de centroides, VIF. |
| `analisis_global.py` | Las capas de análisis. Unidad de observación: la zona. |
| `run_real.py` | Orquestador del Análisis II. |
| `orientacion_objetivo.py` | Dirección de referencia para $\bar\theta$. |
| `indice_dispersion_acotado.py` | Índice de Clark & Evans acotado a [0,1]. Importado dinámicamente por `nucleo.py`. |
| `generar_mapas.py` | Mapas y figuras → `figs_memoria/` (subcarpetas `mapas_viga/`, `mapas_generales/`, `correlacion/`, `por_factor/`). |
| `verificacion_estratificada.py` | Chequeo manual estratificado por reología. No forma parte del flujo automatizado. |

### Las capas

En `analisis_global.py`, con protocolo fijado **antes** de observar resultados:

- **Capa 1** — Spearman predictor–respuesta a través de zonas, con correlación parcial controlando por $n_f$.
- **Capa 2** — Regresión estandarizada. **Condicionada a VIF**: si $\mathrm{VIF}_j \ge 10$, se descarta como herramienta inferencial.
- **Capa 3** — Random Forest + importancia por permutación. **Condicionada a $R^2_{\rm CV}$**: si $\le 0$, se descarta.
- **Capa 4** — Transición vs cuasi-estacionario. Responde la pregunta central.

El descarte de una capa **no es un resultado nulo**: documenta que el diseño no sustenta un modelo de esa complejidad.

### Validación de la atribución de zona (E1/E2/E3)

`reconstruccion_lagrangiana.py` + `trayectoria_comun.py` contrastan la asignación por posición final con tres estimaciones independientes de la zona de tránsito:

- **E1 — traza real.** Moda de la zona observada en el track PTV durante la transición.
- **E2 — traza con reconexión.** Igual, re-enlazando fragmentos. Tolerancias relajadas al cruzar la transición L→viga (zona donde la detección se interrumpe).
- **E3 — advección PIV.** Integración hacia atrás (RK2) desde la posición final, con límite de distancia por paso.

**Es una validación, no una entrada al Análisis II.** Ningún script lee `atribucion_E1_E2_E3.csv`.

**Limitación conocida de E3:** existe un vacío de cobertura PIV de ~21 mm entre la salida de la L y la viga (medido en `cache_zonas`, frame 412 de m70). Ninguna trayectoria reconstruida puede cruzarlo, y todas se truncan ahí. No es un bug ni un umbral mal calibrado: es un límite del montaje óptico. E3 sigue siendo útil dentro de cada región, no a través de la transición.

---

## El criterio de exclusión

Implementado en `criterio_exclusion.py`. Tres decisiones, todas con consecuencias:

**1. La unidad es la celda, no la corrida.** Una toma puede desviarse de su base en cuasi y ser usable en transición (caso real: m73 tiene 5/8 zonas con $d \ge 0{,}5$ en cuasi, pero solo 1/8 en transición). Se marcan como `NaN` los predictores de la etapa afectada; el resto de la fila se conserva.

Esto funciona sin tocar las capas porque `capa1_global` enmascara `NaN` por columna, y `capa2_global` hace `dropna` solo sobre los predictores de la etapa que ajusta.

**2. El estadístico es el $d$ de Cohen, no el $p$-valor.** Umbral: $d \ge 0{,}5$ (efecto medio). Medido sobre las tomas con fibras ($n=168$ celdas):

| Umbral | Celdas excluidas | Veredicto |
|---|---|---|
| $d \ge 0{,}2$ | 108/168 (64,3 %) | Demasiado agresivo |
| **$d \ge 0{,}5$** | **33/168 (19,6 %)** | **Adoptado** |
| $d \ge 0{,}8$ | 7/168 (4,2 %) | Demasiado permisivo |

**3. La base se construye con exclusión mutua.** Al evaluar una toma sin fibras contra "la base", esa toma se retira del conjunto de referencia (leave-one-out). Sin esto se comparaba consigo misma, sesgando $d$ hacia cero.

### Asimetría entre etapas

Cuasi-estacionario se desvía de su base más que transición. Todo umbral excluye más celdas de cuasi:

| Umbral | Cuasi | Transición |
|---|---|---|
| $d \ge 0{,}2$ | 73,8 % | 54,8 % |
| $d \ge 0{,}5$ | 22,6 % | 16,7 % |
| $d \ge 0{,}8$ | 6,0 % | 2,4 % |

Como la Capa 4 compara exactamente esas dos etapas, la exclusión no es neutral. Por eso `construir_tabla_zonas_todas.py` guarda **también** `acum_tabla_zona_sin_excluir.csv`, y la Capa 4 debe reportarse con y sin exclusiones.

**Verificado:** `domina_etapa` resulta idéntico con y sin exclusiones en las 6 combinaciones respuesta × predictor. La conclusión no depende del umbral.

---

## Requisitos

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

`win10toast` (notificaciones) se importa de forma protegida en los scripts que lo usan; corren igual en Linux/macOS sin él.

## Datos de entrada

No versionados. Se esperan en `../PIV_INTERPOLADO/`, una carpeta por corrida, `.txt` por frame:

```
x  y  u  v  mag  valid        # cabecera con "timestamp_s:"
```

Nombres: `m{mezcla}-toma-{n}-n-{concentración}-car-{reología}`.
Correcciones al leer: `y *= -1`, `v *= -1`.

**Cobertura real:** 20 carpetas crudas → 8 son control (`n-0000`, sin fibras) → 12 con fibras → 11 en la tabla final (m73 quedaba excluida manualmente; ahora el criterio decide). Las tomas de control no entran en `acum_tabla_zona.csv`: sin fibras no hay $S$ que analizar.

**Cobertura desigual entre reologías:** Car-0,2 % tiene 8 zonas; Car-0,5 % solo 6 (faltan Vf1c2 y Vf2c3). Toda comparación entre reologías se hace sobre conjuntos distintos de zonas. `Vf1c3` no registra fibras en ninguna corrida.

---

## Notas de calidad de datos

Estas decisiones existen porque su ausencia **falseaba los resultados**.

### Relleno acotado (`MAX_HUECO_INTERP = 3`)

Interpolación 1D limitada a huecos **internos** de ≤3 puntos ($\lesssim$ 2,3 mm). Tramos mayores conservan sus `NaN`. Los NaN de **borde** no se rellenan: `np.interp` extrapola plano, generando $\dot\gamma \approx 0$ espurio justo en las paredes.

### Contaminación por suavizado (`contaminados()`)

Anular los `NaN` **después** de suavizar llega tarde: el kernel ya propagó los valores rellenados a los vecinos válidos. La máscara se dilata en $w/2 + 1$ puntos. Afecta a `box_act.py` y `esp_plug_mag.py`.

### Límites de distancia

| Script | Constante | Uso |
|---|---|---|
| `construir_caches_zonas.py` | `DIST_MAX_KNN_MM = 5.0` | Ajuste local de $\dot\gamma$/$\omega$. Sin límite, un punto de borde ajusta contra vectores al otro lado de un vacío. |
| `construir_caches_adveccion.py` | `DIST_MAX_NN_MM = 3.0` | Retro-advección (sistema legado). |
| `reconstruccion_lagrangiana.py` | `DIST_MAX_NN_MM = 3.0` | Retro-advección E3. |

En un integrador el error **se compone**: la velocidad inventada actualiza la posición, que puede caer fuera, y la trayectoria diverge. Las partículas fuera del radio se marcan perdidas y se truncan.

> **Los umbrales (3 mm, 5 mm, $d\ge0{,}5$) son provisionales**, fijados por argumento dimensional o convención. No están validados por análisis de sensibilidad propio.

### Dependencia instrumental del criterio plug

La fracción de área plug **depende de la configuración PIV** ($\delta\dot\gamma$ por cámara y reología), no es propiedad intrínseca del fluido. Comparar entre zonas o reologías exige igualdad de configuración o reporte explícito del umbral.

---

## Limitaciones del diseño

- **Independencia.** Ver las advertencias arriba.
- **Medición de fibras.** La orientación se registra sobre el **estado final depositado**; los predictores describen etapas anteriores. La asociación es estadística, no causal.
- **Separabilidad.** $\omega$ y $\dot\gamma$ son las partes antisimétrica y simétrica del **mismo tensor**. Su colinealidad es intrínseca al diseño.
- **Cobertura espacial.** Distribución de fibras marcadamente heterogénea entre zonas.

---

## Salidas versionadas

| Archivo | Contenido |
|---|---|
| `acum_tabla_zona.csv` | Tabla (toma, zona) **con** exclusiones aplicadas. |
| `acum_tabla_zona_sin_excluir.csv` | Misma tabla sin exclusiones — para el análisis de sensibilidad. |
| `acum_capa{1,2,4}_global.csv` | Resultados de cada capa. |
| `Analisis_COMPARATIVA_zonas/` | Boxplots + `*_tests.csv` por (reología, concentración). |
| `atribucion_E1_E2_E3.csv`, `resumen_concordancia.csv` | Validación de atribución. |
| `etapas_polilinea.json`, `etapas_zonas.json` | Segmentación V3. |
| `figs_memoria/` | Figuras del Análisis II. |

Los cachés (`cache_completo/`, `cache_zonas/`) **no se versionan**.

---

## Estado y pendientes

**Limpieza ya hecha:** se eliminaron `adveccion_fibras.py`, `run_adveccion.py`, `run_adveccion_todas.py` (sistema de advección duplicado, sin uso registrado) y `capas.py` (unidad de observación mal planteada).

**Pendientes:**

- **Sistema A de advección** (`construir_caches_adveccion.py`, `analizar_adveccion.py` + sus `adv_*.csv`) sigue en el repo. Calcula algo que ningún otro script reproduce: correlación entre exposición temporal ponderada por zona y orientación final. No está integrado al diseño estadístico. Decidir si se adopta como capa adicional o se archiva.
- **Rutas hardcodeadas de Windows:** `utils_etapas.py` (líneas 30–34) y `polilinea_salida_L.py` (línea 23). No bloquean el uso actual, pero rompen la portabilidad.
- **No hay `.gitignore`.** Conviene excluir `cache_*/`, `__pycache__/`, `*.npz`.
- `RECALCULO = True` fijado en varios scripts: reconstruye siempre.
- **Base de Car-0,5 % más ruidosa:** solo 3 tomas de control (m92-toma-1, m92-toma-2, m93-toma-2) frente a 4 de Car-0,2 %, porque m93-toma-1 no se considera. Con leave-one-out, cada toma base de Car-0,5 % se compara contra solo 2.

---

## Referencias

- Wolff, L. — [`Exp_Func_PIV_PTV`](https://github.com/LukasWolff2002/Exp_Func_PIV_PTV): adquisición y procesamiento PIV/PTV base.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2ª ed.
- Clark, P. J. & Evans, F. C. (1954). Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations. *Ecology*, 35(4).
- Jeffery, G. B. (1922). The motion of ellipsoidal particles immersed in a viscous fluid.
- Westerweel, J. (1997). Fundamentals of digital particle image velocimetry.