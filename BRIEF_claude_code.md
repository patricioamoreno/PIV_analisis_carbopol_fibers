# Brief para Claude Code — pipeline PIV/PTV Carbopol

Contexto: repositorio de análisis de una memoria de ingeniería. Los CSV de
resultados quedaron inconsistentes con el código porque un barrido de
sensibilidad se interrumpió a medias. Hay que reparar eso y luego dejar el
pipeline a prueba de que vuelva a pasar.

**Regla general: no cambies ningún número a mano. Todo valor que termine en la
memoria debe salir de una corrida del pipeline.**

---

## TAREA 0 — Diagnóstico (hacer primero, no destructivo)

`sensibilidad.py` reescribe constantes en el código fuente, corre el pipeline y
las restaura con `atexit`. La restauración de constantes funciona. Lo que no
restaura son los CSV derivados, que quedan con la última combinación barrida.

El parámetro en disputa es `DIST_MAX_KNN_MM` en `construir_caches_zonas.py`
(línea ~78). El código dice `5.0`. Los CSV en la raíz contienen valores
calculados con `3.0`.

Verifícalo así:

```bash
python - <<'EOF'
import pandas as pd
d3 = pd.read_csv('sensibilidad/capa1_MAX3_DIST3.0.csv')
d5 = pd.read_csv('sensibilidad/capa1_MAX3_DIST5.0.csv')
def v(df):
    r = df[(df.respuesta=='orden_S')&(df.predictor=='omega')&(df.etapa=='transicion')]
    return round(r.rho.iloc[0], 6)
print('DIST=3.0 ->', v(d3))          # 0.072874
print('DIST=5.0 ->', v(d5))          # 0.069028
print('capa1    ->', v(pd.read_csv('acum_capa1_global.csv')))
EOF
```

Si `capa1` coincide con DIST=3.0, el diagnóstico está confirmado. Reporta el
resultado antes de seguir.

---

## TAREA 1 — Regenerar los CSV con la configuración base

```bash
python construir_tabla_zonas_todas.py
```

**Criterio de aceptación.** Tras regenerar, este valor debe ser `-0.3832`:

```bash
python -c "import pandas as pd; d=pd.read_csv('contraste_simpson.csv'); print(d[d.predictor=='omega_transicion'].rho_car02.iloc[0])"
```

- Si da `-0.3832` → correcto, sigue a la Tarea 2.
- Si da `-0.3750` → el caché `cache_zonas/` también quedó con 3,0 mm.
  Hay que reconstruirlo (`python construir_caches_zonas.py`, es lento).
  **Avisa antes de lanzarlo**, no lo corras sin confirmar.

---

## TAREA 2 — Sellar la procedencia

`procedencia.py` ya está en la carpeta. Integrarlo:

En `construir_tabla_zonas_todas.py`:
- Importar `from procedencia import sellar`.
- Tras `acum.to_csv(SALIDA_CSV, index=False)` (línea ~252), llamar
  `sellar(SALIDA_CSV)`.
- Tras los tres `to_csv` de las capas (líneas ~275-277), sellar los tres
  archivos.

Verificar con `python procedencia.py`: debe imprimir "procedencia verificada"
sin errores.

---

## TAREA 3 — Impedir que el barrido vuelva a contaminar

En `sensibilidad.py`, dentro de `_restaurar()`, después de las seis llamadas a
`escribir_constante`, invalidar los CSV derivados:

```python
    for f in ("acum_tabla_zona.csv", "acum_tabla_zona_sin_excluir.csv",
              "acum_capa1_global.csv", "acum_capa2_global.csv",
              "acum_capa4_global.csv"):
        p = RAIZ / f
        if p.exists():
            p.rename(p.with_suffix(".csv.INVALIDO"))
    print("CSV derivados invalidados. Correr construir_tabla_zonas_todas.py "
          "para regenerarlos con la configuracion base.")
```

Renombrar, no borrar.

---

## TAREA 4 — Registrar el parámetro en el caché

En `construir_caches_zonas.py`, en el `np.savez` (línea ~299) que hoy guarda
`k_vecinos = K_VECINOS`, agregar:

```python
        dist_max_knn = DIST_MAX_KNN_MM,
```

Sin esto, un caché de 3,0 mm y uno de 5,0 mm son indistinguibles en disco.

---

## TAREA 5 — Versión estratificada de sigma_iso

`verificacion_estratificada.py` solo recorre `orden_S` como respuesta. Debe
recorrer también `sigma_iso`.

- Importar `RESPUESTAS` desde `analisis_global` (`["orden_S", "sigma_iso"]`).
- Envolver los bloques 2 y 3 (líneas ~109-210) en un bucle sobre `RESPUESTAS`.
- Escribir el resultado a `acum_capa1_estratificado.csv` con columnas
  `respuesta, reologia, predictor, etapa, rho, p_value, n`, además de
  imprimirlo por consola.
- Sellar la salida con `procedencia.sellar`.

Esto alimenta un párrafo del Capítulo 4 que hoy no tiene fuente estratificada.

---

## TAREA 6 — Exponer el factor direccional

`analisis_global.py` línea 49 define `RESPUESTAS = ["orden_S", "sigma_iso"]`.
`calidad_orientacion` se calcula en `tabla_por_zona` pero no está en esa lista,
así que nunca llega a `capa1_global` ni a `capa4_global`.

No la agregues directamente a `RESPUESTAS`. Vale la identidad exacta

    calidad = 1/2 + (S/2) * cos(2*(theta_med - objetivo))

y el objetivo es constante dentro de cada columna de la viga, de modo que
agregar entre columnas confunde el gradiente cinemático con la asignación del
objetivo. Lo correcto es exponer el segundo factor por separado:

- En `tabla_por_zona`, junto al cálculo de `calidad`, agregar la columna
  `cos2delta = np.cos(2*np.deg2rad(theta_med - objetivo))` (NaN si la zona no
  tiene objetivo).
- Agregar también una columna `columna` extraída del nombre de zona
  (`Vf2c3` → `3`; NaN para Z1/Z2/Z3).
- Verificar la identidad como test: el máximo de
  `abs(0.5 + 0.5*orden_S*cos2delta - calidad_orientacion)` debe ser < 1e-12.
  Si no lo es, detente y reporta.

No agregues análisis por capas sobre `cos2delta`: solo deja las columnas
disponibles.

---

## TAREA 7 — Figuras sin título

`quitar_titulos.py` ya está en la carpeta.

```bash
python quitar_titulos.py     # deja respaldo generar_mapas.py.bak
python generar_mapas.py
```

Distingue dos casos y **no los confundas**: los `fig.suptitle` y los
`ax.set_title` de figuras de un solo panel se eliminan (duplican el pie de
figura en LaTeX); los `ax.set_title` de paneles dentro de figuras multi-panel
se conservan, porque identifican cuál panel es cuál. El script ya hace esa
distinción; solo verifica que tras correrlo sigan existiendo 5 `ax.set_title`
activos.

---

## LO QUE NO DEBES HACER

- No corras `sensibilidad.py` completo. Es lo que causó el problema y tarda
  horas. Solo después de la Tarea 3, y avisando antes.
- No edites valores numéricos en los `.tex` de la memoria. Si un número de la
  memoria no coincide con el pipeline, repórtalo; no lo cambies.
- No reconstruyas `cache_zonas/` sin confirmación explícita.

## AL TERMINAR

Reporta:
1. Resultado del diagnóstico de la Tarea 0.
2. Si el valor de control quedó en `-0.3832`.
3. Qué archivos cambiaron y cuáles se regeneraron.
4. Cualquier número que haya cambiado respecto a los CSV anteriores, con su
   magnitud. Diferencias por debajo de 0,01 en un rho son esperables entre
   3,0 y 5,0 mm; diferencias mayores hay que mirarlas.
