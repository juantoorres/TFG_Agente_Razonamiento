# Paso 6: medicion del tiempo total de ejecucion

## Objetivo del paso

El objetivo de este paso es anadir una medicion sencilla del tiempo total de ejecucion del flujo principal.

Esta medicion servira para documentar en la memoria del TFG el coste temporal de distintas configuraciones experimentales, por ejemplo:

```text
- sin reparacion metrica y sin reparacion de rima;
- con reparacion metrica activada;
- con reparacion de rima exterior A activada;
- con ambas reparaciones activadas.
```

La medicion debe ser auxiliar, no debe afectar al comportamiento del Beam Search ni a las puntuaciones.

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

No debe tocar:

- `src/sonnet_metrics.py`;
- la evaluacion metrica;
- la extraccion de rima;
- la reparacion metrica;
- la reparacion de rima exterior A;
- la futura reparacion de rima interior B;
- el Beam Search;
- `aggregate_scores(...)`;
- los pesos de scoring;
- `k`;
- `max_steps`;
- `alpha`;
- los prompts.

Este paso solo debe medir y mostrar tiempo de ejecucion.

## Requisito importante de salida

El tiempo de ejecucion debe mostrarse unicamente en el archivo:

```text
final_stanza_YYYY-MM-DD_HH-MM-SS.txt
```

No debe anadirse a:

```text
final_stanza_metrics_YYYY-MM-DD_HH-MM-SS.json
final_stanza_trace_YYYY-MM-DD_HH-MM-SS.json
```

Por tanto:

- no debe incluirse en `json_payload`;
- no debe incluirse en `trace_payload`;
- no debe incluirse en `run_parameters`.

## Funcion auxiliar nueva

Crear una funcion auxiliar en `src/langgraph_beam_stanza.py`, preferiblemente cerca de la seccion de guardado de resultados o antes de `main(...)`:

```python
def format_execution_time(seconds: float | None) -> str:
    ...
```

### Comportamiento esperado

La funcion debe:

1. Aceptar segundos como `float`.
2. Si el valor es `None`, devolver:

```text
no disponible
```

3. Si el valor es negativo o no numerico, devolver:

```text
no disponible
```

4. Si el valor es valido, devolver una cadena con segundos y minutos aproximados:

```text
12.34 segundos (0.21 minutos)
```

5. Usar dos decimales.

Ejemplos:

```python
format_execution_time(12.343)
```

debe devolver:

```text
12.34 segundos (0.21 minutos)
```

```python
format_execution_time(None)
```

debe devolver:

```text
no disponible
```

## Import necesario

Anadir:

```python
from time import perf_counter
```

al principio de `src/langgraph_beam_stanza.py`.

No usar `datetime.now()` para medir duracion, porque ya se usa para timestamps de salida y no es la herramienta adecuada para tiempos transcurridos.

## Cambios en `save_final_result(...)`

La firma actual es:

```python
def save_final_result(
    best_beam: dict[str, Any],
    trace: list[dict[str, Any]],
    run_parameters: dict[str, Any],
    output_dir: str = "outputs",
) -> None:
```

Debe cambiarse a:

```python
def save_final_result(
    best_beam: dict[str, Any],
    trace: list[dict[str, Any]],
    run_parameters: dict[str, Any],
    execution_time_seconds: float | None = None,
    output_dir: str = "outputs",
) -> None:
```

Dentro de `save_final_result(...)`, construir:

```python
execution_time_summary = format_execution_time(execution_time_seconds)
```

Y anadir al bloque `text_lines` del `.txt` una linea como:

```text
Tiempo total de ejecucion: 12.34 segundos (0.21 minutos)
```

### Ubicacion recomendada en el `.txt`

Incluir esta linea en el primer bloque de parametros, despues de:

```text
alpha: 1.0
```

Ejemplo:

```text
Modelo: mistral:7b-instruct
Temperatura: 0.3
Num predict: 600
k: 2
max_steps: 3
alpha: 1.0
Tiempo total de ejecucion: 12.34 segundos (0.21 minutos)
Reparacion metrica activa: True
...
```

No anadir esta informacion al JSON de metricas ni al JSON de traza.

## Cambios en `main(...)`

En `main(...)`, medir el tiempo alrededor de:

```python
result = graph.invoke(initial_state)
```

La medicion debe hacerse asi:

```python
execution_start = perf_counter()
result = graph.invoke(initial_state)
execution_time_seconds = perf_counter() - execution_start
```

La variable `execution_time_seconds` debe pasarse a:

```python
save_final_result(
    best_beam=result["beams"][0],
    trace=result.get("trace", []),
    run_parameters=run_parameters,
    execution_time_seconds=execution_time_seconds,
)
```

No medir solo `save_final_result(...)`.

No medir solo la generacion con Ollama.

La medida debe cubrir el flujo completo del grafo:

```text
expand -> score -> prune -> ... -> stop
```

## Salida por terminal

Este paso no exige mostrar el tiempo por terminal.

Si se decide mostrarlo por terminal, debe ser solo como informacion adicional breve, pero no es obligatorio.

La obligacion principal es escribirlo en `final_stanza_*.txt`.

## Cambios que NO deben hacerse

No hacer en este paso:

- no crear un nuevo archivo JSON de tiempos;
- no anadir tiempos por nodo;
- no anadir tiempos por llamada a Ollama;
- no anadir tiempos por reparacion metrica;
- no anadir tiempos por reparacion de rima;
- no cambiar la estructura de `trace_payload`;
- no cambiar la estructura de `json_payload`;
- no meter el tiempo dentro de `run_parameters`;
- no cambiar prompts;
- no cambiar scoring;
- no cambiar reparadores;
- no cambiar `max_steps`, `k` ni `alpha`.

Este paso debe ser minimo y centrado solo en tiempo total de ejecucion.

## Pruebas minimas recomendadas

### 1. Probar formato de tiempo

Ejecutar:

```powershell
$env:PYTHONPATH='.\src'
python -c "from langgraph_beam_stanza import format_execution_time; print(format_execution_time(12.343)); print(format_execution_time(None))"
```

Salida esperada:

```text
12.34 segundos (0.21 minutos)
no disponible
```

### 2. Probar que `save_final_result(...)` escribe el tiempo solo en TXT

Usar una llamada simulada a `save_final_result(...)` con:

```python
execution_time_seconds=12.343
```

Comprobar que el archivo `final_stanza_*.txt` contiene:

```text
Tiempo total de ejecucion: 12.34 segundos (0.21 minutos)
```

### 3. Probar que el JSON de metricas no contiene el tiempo

Abrir el archivo:

```text
final_stanza_metrics_*.json
```

Y comprobar que no contiene:

```text
execution_time
Tiempo total de ejecucion
```

### 4. Probar que el JSON de traza no contiene el tiempo

Abrir el archivo:

```text
final_stanza_trace_*.json
```

Y comprobar que no contiene:

```text
execution_time
Tiempo total de ejecucion
```

### 5. Prueba de compilacion

Ejecutar:

```powershell
python -m compileall src
```

Debe compilar sin errores.

## Criterio de exito del Paso 6

El Paso 6 estara completo cuando:

1. Exista `format_execution_time(...)`.
2. Se use `perf_counter` para medir la duracion.
3. `main(...)` mida el tiempo alrededor de `graph.invoke(initial_state)`.
4. `save_final_result(...)` acepte `execution_time_seconds`.
5. El `.txt` final muestre `Tiempo total de ejecucion: ...`.
6. El JSON de metricas no incluya el tiempo.
7. El JSON de traza no incluya el tiempo.
8. No se haya modificado el Beam Search.
9. No se hayan modificado los reparadores.
10. No se haya implementado todavia la reparacion de rima interior B.

## Resultado esperado

Tras este paso, una salida `final_stanza_*.txt` deberia incluir una linea como:

```text
Tiempo total de ejecucion: 145.82 segundos (2.43 minutos)
```

Esto permitira comparar de forma sencilla el coste de ejecucion de distintas configuraciones experimentales sin contaminar los archivos JSON de metricas ni de trazas.
