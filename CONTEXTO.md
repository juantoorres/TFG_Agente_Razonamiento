# Contexto del proyecto

Este proyecto implementa un sistema de generacion automatica de una estrofa clasica en espanol usando un LLM local con Ollama, LangGraph y Beam Search.

El objetivo actual no es generar un soneto completo, sino una sola estrofa de 4 versos:

```text
- 4 versos.
- Versos endecasilabos.
- Rima consonante ABBA.
```

La unidad de trabajo del sistema es siempre la estrofa completa, no el soneto.

## Archivos principales

### `src/sonnet_metrics.py`

Contiene la evaluacion formal objetiva.

Aunque conserva funciones antiguas para sonetos completos, el flujo actual usa principalmente las funciones de estrofa:

- `count_verse_syllables(...)`
  - Cuenta silabas metricas aproximadas.
  - Usa separacion silabica con `pyphen`, ajuste por palabra final y sinalefa basica.

- `extract_verse_rhyme(...)`
  - Extrae rima consonante desde la vocal tonica de la ultima palabra.

- `evaluate_stanza_abba(...)`
  - Evalua una estrofa de 4 versos.
  - Comprueba numero de versos, endecasilabos y rima ABBA.
  - Devuelve `score`, `score_20`, errores y feedback formal.

- `diagnose_stanza_outer_rhyme(...)`
  - Diagnostica la rima exterior A: verso 1 con verso 4.

- `diagnose_stanza_inner_rhyme(...)`
  - Diagnostica la rima interior B: verso 2 con verso 3.

Limitaciones conocidas:

- La metrica espanola se aproxima; no cubre todos los casos poeticos.
- La sinalefa es basica.
- La deteccion de vocal tonica en palabras sin tilde es heuristica.

### `src/langgraph_beam_stanza.py`

Es el archivo principal del sistema actual.

Implementa:

- generacion de estrofas candidatas con Ollama;
- Beam Search con LangGraph;
- evaluacion formal con `evaluate_stanza_abba(...)`;
- reparacion local de metrica;
- diagnostico y reparacion de rima exterior A;
- diagnostico y reparacion de rima interior B;
- proteccion de mejores beams;
- exportacion de resultados y trazas.

## Flujo general

1. `main()` construye un grafo LangGraph con tres nodos:
   - `expand`
   - `score`
   - `prune`

2. `expand_node(...)` genera estrofas candidatas:
   - desde cero en el primer paso;
   - como correcciones de una estrofa previa en pasos posteriores.

3. `score_node(...)` aplica el pipeline formal:

```text
1. Reparacion metrica local.
2. Diagnostico y reparacion de rima exterior A.
3. Diagnostico y reparacion de rima interior B.
4. Evaluacion final ABBA.
```

4. `prune_node(...)` selecciona los mejores beams.

5. El proceso se repite hasta `max_steps`.

6. Al final se guardan:
   - `final_stanza_*.txt`
   - `final_stanza_metrics_*.json`
   - `final_stanza_trace_*.json`

## Reparacion metrica

Controlada por:

```python
ENABLE_LOCAL_METER_REPAIR = True
```

Funcion principal:

```python
repair_stanza_meter_with_ollama(...)
```

La reparacion metrica:

- revisa versos que no tienen 11 silabas;
- pide variantes del verso concreto al LLM;
- acepta solo variantes que acerquen el verso a 11 silabas;
- no intenta reparar rima;
- suele mejorar mucho el porcentaje de versos endecasilabos.

## Rima exterior A

La rima exterior A corresponde a:

```text
verso 1 = ancla
verso 4 = verso reparable
```

Funciones principales:

- `diagnose_stanza_outer_rhyme(...)`
- `repair_stanza_outer_rhyme_with_ollama(...)`
- `_build_outer_rhyme_repair_messages(...)`

Estrategia:

- Si la rima A ya es correcta, no se repara.
- Si falla, se reescribe solo el verso 4.
- El verso 1 no se modifica.
- Se usan palabras finales candidatas mediante `RHYME_HINT_EXAMPLES`.
- Se evita repetir exactamente la palabra final del verso 1 cuando hay alternativas.
- Se acepta una variante solo si:
  - corrige la rima A;
  - no empeora la distancia metrica del verso 4;
  - respeta la palabra final candidata cuando existe lista forzable.

## Rima interior B

La rima interior B corresponde a:

```text
verso 2 = ancla
verso 3 = verso reparable
```

Funciones principales:

- `diagnose_stanza_inner_rhyme(...)`
- `repair_stanza_inner_rhyme_with_ollama(...)`
- `_build_inner_rhyme_repair_messages(...)`
- `_build_inner_rhyme_repair_messages_for_final_word(...)`
- `generate_conditioned_inner_rhyme_repair_variants_with_ollama(...)`

Estrategia actual:

- Si la rima B ya es correcta, no se repara.
- Si falla, se reescribe solo el verso 3.
- El verso 2 no se modifica.
- Si hay palabras finales candidatas, la generacion puede condicionarse por candidata concreta.
- Ejemplo: generar variantes terminadas exactamente en `llorar`, luego en `pasar`, etc.
- Si no hay candidatas forzables, se usa el metodo anterior basado en una lista general.
- Se acepta una variante solo si:
  - corrige la rima B;
  - no empeora la distancia metrica del verso 3;
  - termina en palabra candidata cuando hay lista forzable.

La reparacion B es la parte mas reciente y aun esta en evaluacion experimental.

## Palabras candidatas de rima

El diccionario:

```python
RHYME_HINT_EXAMPLES
```

contiene ejemplos de palabras finales para rimas frecuentes.

Lo usan tanto A como B para reducir la libertad del modelo.

Ejemplos:

```python
"eva": ["lleva", "nueva", "eleva", "conlleva", "nieva"]
"ar": ["llorar", "pasar", "mirar", "soñar", "callar", "recordar"]
"ombra": ["sombra", "alfombra", "asombra", "nombra"]
"ado": ["helado", "olvidado", "callado", "pasado", "soñado", "amado"]
```

Si no hay candidatas utiles, el LLM suele fallar mas al reparar rima.

## Protecciones del Beam Search

Se anadio elitismo para no perder buenas estrofas ya encontradas.

Constantes:

```python
ENABLE_BEAM_ELITISM = True
ELITE_BEAMS_TO_KEEP = 1
ENABLE_PROMPT_CONSTRAINT_PROTECTION = True
```

Funciones relacionadas:

- `is_preservable_beam(...)`
- `mark_beam_as_elite_candidate(...)`
- `deduplicate_beams_by_stanza(...)`
- `build_constraint_protection_prompt(...)`

Efecto:

- Un beam bueno puede sobrevivir aunque sus descendientes sean peores.
- Si una rima ya esta bien, el prompt de correccion pide conservar esos versos.
- Si todos o algunos versos ya son endecasilabos, el prompt pide conservarlos.

## Medicion de tiempo

El sistema mide el tiempo total de ejecucion alrededor de:

```python
graph.invoke(initial_state)
```

Funcion:

```python
format_execution_time(...)
```

El tiempo se muestra solo en `final_stanza_*.txt`.

No se guarda en:

- `final_stanza_metrics_*.json`
- `final_stanza_trace_*.json`

## Parametros principales

Modelo:

```python
GENERATION_MODEL = "mistral:7b-instruct"
```

Parametros habituales:

```python
TEMPERATURE = 0.3
NUM_PREDICT = 600
ALPHA = 1.0
k = 2
max_steps = 3
```

Reparacion metrica:

```python
METER_REPAIR_VARIANTS_PER_VERSE = 5
METER_REPAIR_TEMPERATURE = 0.4
METER_REPAIR_NUM_PREDICT = 200
```

Reparacion rima A:

```python
OUTER_RHYME_REPAIR_VARIANTS = 5
OUTER_RHYME_REPAIR_TEMPERATURE = 0.3
OUTER_RHYME_REPAIR_NUM_PREDICT = 200
```

Reparacion rima B:

```python
INNER_RHYME_REPAIR_VARIANTS = 5
INNER_RHYME_REPAIR_TEMPERATURE = 0.3
INNER_RHYME_REPAIR_NUM_PREDICT = 200
INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE = True
INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE = 2
INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS = 4
```

## Salidas generadas

Cada ejecucion crea archivos con timestamp en `outputs/`:

- `final_stanza_*.txt`
  - resumen legible;
  - parametros;
  - tiempo total;
  - score;
  - diagnosticos;
  - estrofa final.

- `final_stanza_metrics_*.json`
  - resultado final estructurado;
  - metricas completas;
  - reportes de reparacion.

- `final_stanza_trace_*.json`
  - traza de expand, score, prune y stop;
  - beams generados y seleccionados;
  - reportes intermedios.

## Estado actual

Funciona razonablemente bien:

- generacion de estrofas ABBA;
- evaluacion formal objetiva;
- reparacion metrica;
- diagnostico de rima A y B;
- reparacion de rima A;
- preservacion de beams buenos;
- medicion de tiempo;
- trazas para analisis experimental.

En evaluacion:

- reparacion de rima interior B.

Problemas observados:

- La rima B mejora en algunos casos, pero no siempre.
- A veces el LLM no respeta la palabra final candidata.
- A veces corrige la rima pero empeora la metrica.
- Todavia no se ha implementado reparacion metrica posterior a la reparacion B.

## Posibles siguientes pasos

1. Ejecutar varias pruebas con la reparacion B condicionada por palabra candidata.
2. Analizar en los JSON:
   - `inner_rhyme_repair_report.changed`;
   - `conditioned_by_candidate`;
   - `conditioned_generation_reports`;
   - motivos de rechazo por variante.
3. Si B corrige rima pero falla por metrica, valorar una reparacion metrica posterior a B.
4. Comparar experimentalmente:
   - sin reparaciones;
   - solo metrica;
   - metrica + rima A;
   - metrica + rima A + rima B.
5. Documentar tiempos y resultados para la memoria del TFG.
