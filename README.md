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

El campo PIV se agrega por (zona, etapa) → vector de predictores ($|V|$, $\omega$, $\dot\gamma$ en transición y cuasi-estacionario). Las fibras del último fotograma se asignan por **posición final (bounding box)** y se resumen con el orden-parámetro $S$, la dirección media circular $\bar\theta$ y $\sigma_{\rm iso}$ (fiable si $n_f \ge 5$).

La unidad de observación es el par **(toma, zona)**.

> **La asignación de fibra a zona por posición final es un supuesto, no un hecho medido**, y el pipeline lo valida explícitamente — ver más abajo, *Validación cruzada de la atribución de zona (E1/E2/E3)*.

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
│ esp_plug_mag.py              │   │      → acum_tabla_zona.csv   │
│ box_piv_ptv.py               │   │            │                 │
│ esp_overlay_piv_ptv.py       │   │ run_real.py / analisis_global│
└─────────────────────────────┘   └──────────────┬────────────────┘
                                                   │
                                    ┌──────────────┴────────────────┐
                                    ▼                                ▼
                     reconstruccion_lagrangiana.py         verificacion_estratificada.py
                     (validación E1/E2/E3, opcional)        indice_dispersion_acotado.py
                     → atribucion_E1_E2_E3.csv                  (auxiliar de nucleo.py)
                     → resumen_concordancia.csv
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
python verificacion_estratificada.py       # chequeo manual estratificado (opcional)

# 4. Validación cruzada de la atribución de zona (opcional, ver más abajo)
python reconstruccion_lagrangiana.py       # → atribucion_E1_E2_E3.csv, resumen_concordancia.csv
```

---

## Descripción de los módulos

### Construcción de cachés

| Script | Entrada | Salida | Qué hace |
|---|---|---|---|
| `construir_caches.py` | `PIV_INTERPOLADO/` | `cache_completo/*.npz` | Interpola el campo PIV (IDW, 4 vecinos) sobre las 3 polilíneas. Matriz `[n_frames × n_puntos]` + timestamps. |
| `construir_caches_zonas.py` | `PIV_INTERPOLADO/` | `cache_zonas/*.npz` | Cachea los puntos PIV válidos ya procesados: zona asignada + $\dot\gamma$ + $\omega_z$ por KNN. No interpola sobre grilla. |

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
| `definir_zonas.py` | Geometría de las zonas (Z1/Z2/Z3 por polígono; celdas de viga por bounding box). Test de punto en polígono (`asignar_zona`). |
| `exportar_fibras_ultimo_frame.py` | Extrae la "foto final" de fibras desde `ptv_merged.json`. Reconstruye fibras ausentes del último frame por oclusión transitoria. |
| `construir_tabla_zonas_todas.py` | Acumula todas las corridas → `acum_tabla_zona.csv`, una fila por (toma, zona). |
| `carga_real.py` | Carga PIV + fibras + etapas → DataFrame largo. Agregación robusta: clip al percentil 99 + mediana. |
| `nucleo.py` | Convenciones de zonas, $\theta$ circular, dispersión de centroides ($\sigma_{\rm iso}$ + índice de Clark-Evans acotado vía `indice_dispersion_acotado.py`), VIF. |
| `analisis_global.py` | **Las capas de análisis en uso.** Unidad de observación: la zona. |
| `run_real.py` | Orquestador del Análisis II. Punto de entrada. |
| `orientacion_objetivo.py` | Definición de la dirección de referencia para $\bar\theta$. |
| `generar_mapas.py` | Mapas de campo por zona y etapa; recalcula sus propias correlaciones desde `acum_tabla_zona.csv` (independiente de `run_real.py`). |
| `verificacion_estratificada.py` | Chequeo manual, estratificado por reología, de las correlaciones sobre `acum_tabla_zona.csv`. Reemplaza a un `verificacion_resultados.py` anterior que agregaba ambas reologías y por tanto reportaba artefactos de la paradoja de Simpson. No forma parte del flujo automatizado; se ejecuta a mano para inspeccionar resultados. |
| `indice_dispersion_acotado.py` | Índice de dispersión de Clark & Evans (1954), acotado a [0,1]. Importado dinámicamente dentro de `nucleo.py`; no es un script huérfano aunque no aparezca en los imports de nivel de módulo. |

### Las capas del análisis estratificado

Implementadas en `analisis_global.py`, con protocolo de decisión fijado **antes** de observar los resultados:

- **Capa 1** — Spearman entre cada predictor (por etapa) y cada respuesta, a través de las zonas. Correlación parcial controlando por $n_f$ para descartar el confundente de conteo.
- **Capa 2** — Regresión multivariada estandarizada. **Condicionada a VIF**: si $\mathrm{VIF}_j \ge 10$ para algún predictor, la capa se descarta como herramienta inferencial (los coeficientes individuales carecen de interpretación).
- **Capa 3** — Random Forest + importancia por permutación. **Condicionada a $R^2_{\rm CV}$**: si $R^2_{\rm CV} \le 0$, la capa se descarta íntegramente.
- **Capa 4** — Comparativa temporal transición vs cuasi-estacionario. Responde la pregunta central de la memoria.

El descarte de una capa **no es un resultado nulo**: documenta que el diseño no sustenta un modelo de esa complejidad.

### Validación cruzada de la atribución de zona (E1/E2/E3)

`reconstruccion_lagrangiana.py` + `trayectoria_comun.py` responden una pregunta metodológica distinta a la del Análisis II: **¿es razonable asignar cada fibra a la zona donde quedó, en vez de a la zona donde realmente se orientó?**

El pipeline principal (Análisis II) asigna cada fibra a su zona por **posición final**. Esto es débil en principio: una fibra pudo orientarse cruzando otra zona y solo terminar donde terminó. `reconstruccion_lagrangiana.py` contrasta esa atribución por **tres caminos independientes**:

- **E1 — Track real (oro).** Moda de la zona observada directamente en el track PTV medido durante la transición.
- **E2 — Stitching simplificado.** Igual que E1, pero re-enlazando segmentos de track fragmentados por pérdida de detección (gap-closing).
- **E3 — Pathline PIV.** Integración hacia atrás (RK2) de la posición final sobre el campo $(u,v)$, clasificando la zona modal visitada durante la transición. Hereda el supuesto de partícula pasiva.

Salida: `atribucion_E1_E2_E3.csv` (una fila por track con las 3 atribuciones) y `resumen_concordancia.csv` (tasa de acuerdo E1–E2, E1–E3, E2–E3 por toma).

**Este análisis es una validación de robustez, no una entrada al Análisis II.** Ningún otro script del pipeline lee `atribucion_E1_E2_E3.csv`; el Análisis II sigue usando la asignación por posición final. Un acuerdo alto entre E1/E2/E3 respalda que esa simplificación es razonable; un acuerdo bajo señalaría que la zona final subestima el recorrido real de la fibra y que las conclusiones del Análisis II deberían leerse con esa reserva.

---

## Scripts obsoletos, duplicados o superados

Al revisar el repositorio completo se identificaron **tres sistemas paralelos** que reconstruyen la trayectoria de las fibras por advección hacia atrás sobre el campo PIV, todos motivados por la misma pregunta metodológica (reunión 10-07). Solo uno debería quedar activo.

| Sistema | Scripts | Estado |
|---|---|---|
| **A — original** | `construir_caches_adveccion.py`, `analizar_adveccion.py` | Generó salidas ya versionadas (`adv_resumen_zonas.csv`, `adv_influencia_global.csv`, `adv_influencia_por_reologia.csv`). El propio docstring de `reconstruccion_lagrangiana.py` lo describe como el "Método 2 original", superado porque no tiene forma de validarse contra otro método. |
| **B — reescritura intermedia** | `adveccion_fibras.py`, `run_adveccion.py`, `run_adveccion_todas.py` | Reescritura del Sistema A con interfaz por `argparse` y ejecución batch. **No generó ninguna salida versionada** (no hay `resultados_adveccion/`, `log_adveccion.txt`, ni ningún `adv_*.csv` de este sistema en el repo) — no hay evidencia de que se haya usado para un resultado real. Además, `_interp_frame()` usa `NearestNDInterpolator` **sin límite de distancia**: el mismo bug de extrapolación ilimitada que se corrigió en el Sistema A no se propagó aquí. |
| **C — vigente** | `reconstruccion_lagrangiana.py`, `trayectoria_comun.py` | Sistema con triple validación cruzada (E1/E2/E3). Generó las salidas más recientes (`atribucion_E1_E2_E3.csv`, `resumen_concordancia.csv`). Es el que el propio código describe como solución al problema de fondo ("no tienen forma de validarse entre sí") de los sistemas A y B. |

### Recomendación

**Eliminar el Sistema B completo** (`adveccion_fibras.py`, `run_adveccion.py`, `run_adveccion_todas.py`): es una reescritura sin uso registrado, con un bug ya corregido en otro lado, y superada por el Sistema C. No hay razón para conservarlo.

**El Sistema A queda a tu criterio.** A diferencia de B, sí generó resultados reales y versionados — si esos números llegaron a citarse en algún borrador de la memoria, hay que conservarlos con trazabilidad aunque el método ya no sea el vigente. Sugerencia: mover `construir_caches_adveccion.py`, `analizar_adveccion.py` y los `adv_*.csv` que produjeron a una carpeta `legado/` con una nota de una línea explicando por qué se dejó de usar, en vez de borrarlos. Si esos números nunca se usaron fuera de pruebas, entonces sí se pueden eliminar sin más.

**`capas.py` también está obsoleto** (ya señalado en una revisión anterior de este README). Usa la fibra-dentro-de-zona como unidad de observación, lo cual está mal planteado: el PIV entrega un escalar por (zona, etapa), de modo que dentro de una zona el predictor es *constante* y no puede correlacionarse fibra a fibra. Nadie lo importa. `run_real.py` usa `analisis_global.py`, que es la versión correcta del mismo análisis. **Recomendado eliminar.**

---

## Requisitos

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

`win10toast` (notificaciones de escritorio) se usa en varios scripts para avisar cuando termina un cálculo largo. Todos los scripts activos lo importan de forma protegida:

```python
try:
    from win10toast import ToastNotifier
    USAR_NOTIFICACION = True
except ImportError:
    USAR_NOTIFICACION = False
```

y por tanto corren igual en Linux/macOS sin esa librería, solo sin notificación de escritorio.

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

Rellenar huecos largos inventa un perfil de velocidad, y $\dot\gamma = dv/ds$ amplifica cualquier rampa artificial. Los NaN de **borde** tampoco se rellenan: `np.interp` extrapola plano, generando $\dot\gamma \approx 0$ espurio justo en las paredes.

### Contaminación por suavizado (`contaminados()`)

Anular los NaN **después** de suavizar llega tarde: el kernel ya propagó los valores rellenados a los puntos válidos vecinos. La máscara de NaN se dilata en $w/2 + 1$ puntos antes de invalidar. Afecta a `box_act.py` y `esp_plug_mag.py` (donde $\dot\gamma$ alimenta la clasificación plug).

### Límites de distancia en interpoladores espaciales

Tres puntos independientes del código interpolan (o extrapolan) sobre una malla dispersa de vectores PIV. Los tres necesitaron el mismo tipo de corrección, porque el problema de fondo es el mismo: **un interpolador sin límite de distancia no distingue entre "hueco de correlación dentro del material" y "punto fuera del material"**.

| Script | Constante | Uso |
|---|---|---|
| `construir_caches_zonas.py` | `DIST_MAX_KNN_MM = 5.0` | Ajuste local de $\dot\gamma$ y $\omega$ por KNN. Sin límite, un punto en el borde ajusta su gradiente contra vectores al otro lado de un vacío. |
| `construir_caches_adveccion.py` | `DIST_MAX_NN_MM = 3.0` | Retro-advección (Sistema A, legado). |
| `reconstruccion_lagrangiana.py` (E3) | `DIST_MAX_NN_MM = 3.0` | Retro-advección (Sistema C, vigente). |

En un integrador (advección hacia atrás) el error de un relleno sin límite **se compone**: la velocidad inventada actualiza la posición, la posición errónea puede volver a caer fuera del material, y la trayectoria diverge sin recuperación posible. Las partículas que salen del radio de confianza se marcan como **perdidas** y su trayectoria se trunca en ese punto; los scripts reportan qué fracción de fibras conserva trayectoria completa.

> **Los umbrales de distancia (3 mm, 5 mm) son provisionales**, fijados por argumento dimensional desde $\Delta y_{\rm PIV}$ (0,75–2,05 mm según cámara y reología). No están validados empíricamente. Si la supervivencia en retro-advección es baja, eso no indica un umbral mal puesto sino que la cobertura PIV no sustenta el método para esa toma.

### Dependencia instrumental del criterio plug

La fracción de área clasificada como plug **depende de la configuración PIV** ($\delta\dot\gamma$ por cámara y reología) y no es una propiedad intrínseca del fluido. Toda comparación entre zonas o reologías exige igualdad de configuración o el reporte explícito del umbral aplicado.

---

## Limitaciones del diseño

- **Independencia.** Ver la advertencia estadística arriba.
- **Naturaleza de la medición de fibras.** La orientación se registra sobre el **estado final depositado**, mientras que los predictores describen etapas anteriores. La asociación es estadística, no causal: no permite trazar la trayectoria de una fibra individual sin recurrir a E1/E2/E3, que a su vez heredan el supuesto de partícula pasiva.
- **Separabilidad de los predictores.** $\omega$ y $\dot\gamma$ son las partes antisimétrica y simétrica del **mismo tensor gradiente**. Su colinealidad es intrínseca al diseño y no se elimina con tratamiento estadístico posterior.
- **Cobertura espacial.** La distribución de fibras entre zonas es marcadamente heterogénea. `Vf1c3` no registra fibra alguna en ninguna corrida; `Vf1c2` tiene una mediana de 2 fibras, bajo el criterio de fiabilidad $n_f \ge 5$.

---

## Salidas versionadas

| Archivo | Contenido |
|---|---|
| `acum_tabla_zona.csv` | Tabla acumulada (toma, zona) — la materia prima del Análisis II. |
| `acum_capa{1,2,4}_global.csv` | Resultados de cada capa. |
| `atribucion_E1_E2_E3.csv` | Atribución de zona de transición por track, por los 3 métodos (E1/E2/E3). Validación, no entrada al Análisis II. |
| `resumen_concordancia.csv` | Tasa de acuerdo E1–E2, E1–E3, E2–E3 por toma. |
| `etapas_polilinea.json`, `etapas_zonas.json` | Segmentación V3 por corrida. |
| `Boxplots/`, `Esp_PIV-PTV/`, `figs_memoria/` | Figuras. |

Los cachés (`cache_completo/`, `cache_zonas/`) **no se versionan**: se regeneran desde los datos crudos.

---

## Pendientes conocidos

- **Rutas hardcodeadas de Windows** que rompen la portabilidad:
  - `utils_etapas.py` (líneas 30–34): `BASE_PATH`, `ETAPAS_JSON`, `CACHE_PATH` apuntan a `C:/Users/elisa/Desktop/...`. Hoy no bloquean el pipeline porque los scripts que importan de `utils_etapas.py` (`box_act.py`, `box_piv_ptv.py`) le pasan la ruta explícitamente y no dependen del valor por defecto — pero cualquier uso nuevo de `cargar_etapas()` sin argumento fallará en otra máquina.
  - `polilinea_salida_L.py` (línea 23): `ARCHIVO` apunta a un `.txt` específico en `C:/Users/elisa/Desktop/PIV_INTERPOLADO/...`. No bloquea el día a día porque `Polilinea_L/linea_salida.npy` ya está versionado, pero el script fallará si alguien necesita regenerar esa geometría en otra máquina.
- No hay `.gitignore` propio del proyecto más allá del sugerido aquí; conviene revisar que siga excluyendo `cache_*/`, `__pycache__/` y `*.npz`.
- `RECALCULO = True` está fijado en varios scripts de caché: reconstruye siempre, aunque el caché exista.
- Ver la sección *Scripts obsoletos, duplicados o superados* arriba para la limpieza pendiente de los tres sistemas de advección.

---

## Referencias

- Wolff, L. — [`Exp_Func_PIV_PTV`](https://github.com/LukasWolff2002/Exp_Func_PIV_PTV): adquisición y procesamiento PIV/PTV base, y metodología de origen de la reconstrucción lagrangiana E1/E2/E3.
- Clark, P. J. & Evans, F. C. (1954). Distance to Nearest Neighbor as a Measure of Spatial Relationships in Populations. *Ecology*, 35(4).
- Jeffery, G. B. (1922). The motion of ellipsoidal particles immersed in a viscous fluid.
- Westerweel, J. (1997). Fundamentals of digital particle image velocimetry.
- Teng, L. et al. (2021). Flow-induced fibre orientation in UHPC.