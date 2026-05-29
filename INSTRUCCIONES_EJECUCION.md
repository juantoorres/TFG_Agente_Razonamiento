# Instrucciones de ejecución

Este documento explica como ejecutar el sistema actual de generacion de una estrofa de 4 versos endecasilabos con rima consonante ABBA.

El archivo principal que se debe ejecutar es:

```text
src/langgraph_beam_stanza.py
```

## 1. Requisitos

### Python

Se recomienda usar Python 3.10 o superior.

Comprobar version:

```powershell
python --version
```

### Ollama

El proyecto usa un modelo local servido con Ollama.

Ollama debe estar instalado y ejecutandose antes de lanzar el script.

Pagina oficial:

```text
https://ollama.com/
```

En Windows, normalmente Ollama queda ejecutandose en segundo plano tras abrir la aplicacion.

Si se quiere arrancar manualmente desde terminal:

```powershell
ollama serve
```

Por defecto, el codigo espera que Ollama este disponible en:

```text
http://localhost:11434
```

## 2. Instalación de dependencias

Desde la raiz del proyecto, instalar las dependencias:

```powershell
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye actualmente:

```text
requests
langgraph
langchain
pyphen
```

Notas:

- `requests` se usa para llamar a la API HTTP de Ollama.
- `langgraph` se usa para construir el grafo de Beam Search.
- `pyphen` se usa en `sonnet_metrics.py` para la separacion silabica aproximada.
- `json`, `pathlib`, `typing`, `datetime`, `re`, etc. no aparecen en `requirements.txt` porque forman parte de la libreria estandar de Python.

## 3. Modelo de Ollama

El modelo configurado por defecto es:

```python
GENERATION_MODEL = "mistral:7b-instruct"
```

Antes de ejecutar el programa por primera vez, descargar el modelo:

```powershell
ollama pull mistral:7b-instruct
```

Se puede probar que Ollama responde con:

```powershell
ollama run mistral:7b-instruct
```

Para usar otro modelo, primero descargarlo con Ollama y despues cambiar la constante `GENERATION_MODEL` en:

```text
src/langgraph_beam_stanza.py
```

Ejemplo:

```python
GENERATION_MODEL = "llama3.1:8b"
```

En ese caso, tambien habria que ejecutar:

```powershell
ollama pull llama3.1:8b
```

## 4. Puerto y URL de Ollama

La URL de Ollama se configura en `src/langgraph_beam_stanza.py`:

```python
BASE_URL = "http://localhost:11434"
```

Si Ollama se ejecuta en otro puerto o en otra maquina, modificar solo esa constante.

Ejemplo:

```python
BASE_URL = "http://localhost:11435"
```

## 5. Como ejecutar

Desde la raiz del proyecto:

```powershell
python src/langgraph_beam_stanza.py
```

Durante la ejecucion se imprimen por consola las fases principales:

- `EXPAND`
- `SCORE`
- `PRUNE`
- resultado final

La ejecucion puede tardar varios minutos, porque cada reparacion puede hacer llamadas adicionales a Ollama.

## 6. Prompt inicial

El prompt inicial esta dentro de `main()`, en `initial_state["question"]`:

```python
"Genera una estrofa de 4 versos endecasilabos con rima consonante ABBA sobre el paso del tiempo y la memoria."
```

Este texto se puede cambiar para probar otros temas.

Ejemplos:

```python
"Genera una estrofa de 4 versos endecasilabos con rima consonante ABBA sobre la nostalgia de la infancia."
```

```python
"Genera una estrofa de 4 versos endecasilabos con rima consonante ABBA sobre el mar al amanecer."
```

```python
"Genera una estrofa de 4 versos endecasilabos con rima consonante ABBA sobre la soledad en una ciudad moderna."
```

Este prompt es importante porque permite personalizar la tematica de la estrofa sin cambiar el algoritmo.

## 7. Archivos de salida

Cada ejecucion genera archivos en:

```text
outputs/
```

Se crean tres archivos con timestamp:

```text
final_stanza_YYYY-MM-DD_HH-MM-SS.txt
final_stanza_metrics_YYYY-MM-DD_HH-MM-SS.json
final_stanza_trace_YYYY-MM-DD_HH-MM-SS.json
```

### `final_stanza_*.txt`

Es el archivo mas comodo para leer el resultado.

Incluye:

- modelo usado;
- temperatura;
- `k`;
- `max_steps`;
- tiempo total de ejecucion;
- parametros de reparacion;
- score global;
- historial de scores;
- resumen de metrica;
- resumen de rima A;
- resumen de rima B;
- feedback formal;
- estrofa final.

Este es el archivo recomendado para una primera revision manual.

### `final_stanza_metrics_*.json`

Contiene el resultado final en formato estructurado.

Incluye:

- parametros de ejecucion;
- metrica completa;
- diagnosticos de rima;
- reportes de reparacion;
- estrofa final.

Es util para analizar si una reparacion se ha activado o no.

Campos especialmente utiles:

```text
outer_rhyme_repair_report
inner_rhyme_repair_report
meter_repair_report
```

Dentro de ellos:

```text
post_a_meter_repair_attempts
post_a_meter_repair_successes
post_b_meter_repair_attempts
post_b_meter_repair_successes
conditioned_generation_reports
variants
rejection_reason
```

### `final_stanza_trace_*.json`

Contiene la traza completa del Beam Search.

Incluye:

- candidatos generados en cada paso;
- scores;
- beams seleccionados;
- beams preservados por elitismo;
- diagnosticos y reparaciones intermedias.

Es el archivo mas util para estudiar el comportamiento del algoritmo.

## 8. Parametros principales que se pueden tocar

Todos estan en la parte superior de:

```text
src/langgraph_beam_stanza.py
```

### Modelo y servidor

```python
BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "mistral:7b-instruct"
```

### Generacion principal

```python
TEMPERATURE = 0.3
RETRY_TEMPERATURE = 0.1
NUM_PREDICT = 600
ALPHA = 1.0
```

Significado:

- `TEMPERATURE`: creatividad del modelo en la generacion principal.
- `RETRY_TEMPERATURE`: temperatura usada si falla el parseo JSON.
- `NUM_PREDICT`: longitud maxima de respuesta del modelo.
- `ALPHA`: peso de la puntuacion historica en el score agregado.

### Beam Search

Dentro de `initial_state`:

```python
"max_steps": 3,
"k": 2,
```

Significado:

- `k`: numero de beams conservados en cada poda.
- `max_steps`: numero maximo de iteraciones del grafo.

Subir estos valores puede mejorar la busqueda, pero aumenta el tiempo de ejecucion.

### Reparacion métrica inicial

```python
ENABLE_LOCAL_METER_REPAIR = True
METER_REPAIR_VARIANTS_PER_VERSE = 5
METER_REPAIR_TEMPERATURE = 0.4
METER_REPAIR_NUM_PREDICT = 200
```

Esta fase intenta convertir en endecasilabos los versos que no tienen 11 silabas.

### Reparacion rima exterior A

```python
ENABLE_OUTER_RHYME_REPAIR = True
OUTER_RHYME_REPAIR_VARIANTS = 5
OUTER_RHYME_REPAIR_TEMPERATURE = 0.3
OUTER_RHYME_REPAIR_NUM_PREDICT = 200
```

La rima A corresponde a:

```text
verso 1 con verso 4
```

Si falla, se reescribe solo el verso 4.

### Reparacion métrica posterior a rima A

```python
ENABLE_POST_A_RHYME_METER_REPAIR = True
POST_A_RHYME_METER_REPAIR_VARIANTS = 5
POST_A_RHYME_METER_REPAIR_TEMPERATURE = 0.4
POST_A_RHYME_METER_REPAIR_NUM_PREDICT = 200
```

Solo se activa si una variante del verso 4 ya corrige la rima A, pero falla por metrica.

La palabra final que consigue la rima queda bloqueada.

### Reparación rima interior B

```python
ENABLE_INNER_RHYME_REPAIR = True
INNER_RHYME_REPAIR_VARIANTS = 5
INNER_RHYME_REPAIR_TEMPERATURE = 0.3
INNER_RHYME_REPAIR_NUM_PREDICT = 200
INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE = True
INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE = 2
INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS = 4
```

La rima B corresponde a:

```text
verso 2 con verso 3
```

Si falla, se reescribe solo el verso 3.

Si `INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE` esta activo, el sistema puede pedir variantes terminadas en palabras candidatas concretas.

### Reparación métrica posterior a rima B

```python
ENABLE_POST_B_RHYME_METER_REPAIR = True
POST_B_RHYME_METER_REPAIR_VARIANTS = 5
POST_B_RHYME_METER_REPAIR_TEMPERATURE = 0.4
POST_B_RHYME_METER_REPAIR_NUM_PREDICT = 200
```

Solo se activa si una variante del verso 3 ya corrige la rima B, pero falla por metrica.

La palabra final que consigue la rima queda bloqueada.

### Elitismo y protección de restricciones

```python
ENABLE_BEAM_ELITISM = True
ELITE_BEAMS_TO_KEEP = 1
ENABLE_PROMPT_CONSTRAINT_PROTECTION = True
```

Efecto:

- conserva buenos beams aunque sus descendientes sean peores;
- pide al modelo que preserve rimas ya correctas;
- pide conservar versos que ya son endecasilabos.

## 9. Palabras candidatas de rima

El sistema incluye un diccionario interno:

```python
RHYME_HINT_EXAMPLES
```

Sirve para sugerir palabras finales reales que comparten una rima.

Ejemplo:

```python
"ar": ["llorar", "pasar", "mirar", "soñar", "callar", "recordar"]
```

Si el sistema detecta que la rima objetivo es `ar`, puede pedir al modelo variantes que terminen exactamente en alguna de esas palabras.

Este diccionario no es el nucleo del Beam Search. Es una heuristica auxiliar para reducir el espacio de busqueda en la reparacion de rima.

## 10. Comprobación rápida

Para comprobar que el codigo compila:

```powershell
python -m compileall src
```

Para ejecutar el sistema completo:

```powershell
python src/langgraph_beam_stanza.py
```

Si todo funciona, al final deberia aparecer en consola:

```text
=== RESULTADO FINAL ===
```

y deberian generarse archivos nuevos en:

```text
outputs/
```

## 11. Errores frecuentes

### Ollama no está arrancado

Sintoma habitual:

```text
Connection refused
```

Solucion:

```powershell
ollama serve
```

o abrir la aplicacion de Ollama.

### El modelo no está descargado

Sintoma habitual:

```text
model not found
```

Solucion:

```powershell
ollama pull mistral:7b-instruct
```

o descargar el modelo configurado en `GENERATION_MODEL`.

### La ejecución tarda mucho

Es normal que tarde, porque el sistema puede hacer varias llamadas al modelo:

- generacion de candidatos;
- reparacion metrica;
- reparacion de rima A;
- post-reparacion metrica A;
- reparacion de rima B;
- post-reparacion metrica B.

Para acelerar pruebas se puede reducir:

```python
"max_steps": 3
"k": 2
```

o bajar el numero de variantes de reparacion.

### El resultado no cumple siempre ABBA completo

Es esperable. El sistema usa Beam Search y reparaciones locales, pero el LLM puede seguir fallando. Por eso se guardan trazas y metricas para analizar que restricciones se cumplieron y cuales no.

## 12. Configuracion mínima recomendada para pruebas, Sergio

Para hacer pruebas sin tocar casi nada:

1. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

2. Arrancar Ollama.

3. Descargar el modelo:

```powershell
ollama pull mistral:7b-instruct
```

4. Ejecutar:

```powershell
python src/langgraph_beam_stanza.py
```

5. Revisar:

```text
outputs/final_stanza_*.txt
```

Si se quiere usar otro modelo o puerto, modificar solo:

```python
BASE_URL
GENERATION_MODEL
```
