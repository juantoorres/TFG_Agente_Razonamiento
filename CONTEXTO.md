# Contexto del proyecto

Este proyecto implementa un agente de generacion poetica centrado en una estrofa de 4 versos endecasilabos con rima consonante ABBA.

La unidad de trabajo del sistema es siempre la estrofa completa:

```text
- 4 versos.
- 11 silabas metricas por verso.
- Rima consonante ABBA.
```

El sistema combina un LLM local servido con Ollama, un grafo de ejecucion con LangGraph y un algoritmo Beam Search para explorar varias soluciones candidatas antes de seleccionar la mejor.

## Archivos principales

### `src/langgraph_beam_stanza.py`

Es el archivo principal del sistema.

Contiene:

- configuracion global del experimento;
- llamada a Ollama;
- construccion de prompts;
- generacion de estrofas candidatas;
- reparacion metrica local;
- reparacion de rima exterior A;
- reparacion de rima interior B;
- reparacion metrica posterior a rima A y B;
- elitismo y proteccion de restricciones;
- nodos del grafo LangGraph;
- guardado de resultados y trazas.

### `src/sonnet_metrics.py`

Contiene la evaluacion formal.

Aunque conserva algunas funciones historicas, el flujo actual usa las funciones de estrofa ABBA:

- conteo de versos;
- conteo silabico aproximado;
- deteccion de rima consonante;
- diagnostico de rima exterior A;
- diagnostico de rima interior B;
- puntuacion formal de la estrofa.

## Flujo general del sistema

1. `main()` construye el grafo LangGraph.
2. El grafo empieza con un beam inicial sin estrofa.
3. `expand_node(...)` genera estrofas candidatas con Ollama.
4. `score_node(...)` evalua y, si las fases estan activadas, aplica reparaciones locales.
5. `prune_node(...)` selecciona los mejores beams.
6. `should_continue(...)` decide si continuar o terminar.
7. `save_final_result(...)` guarda la estrofa final, las metricas y la traza.

El ciclo principal es:

```text
EXPAND -> SCORE -> PRUNE -> EXPAND -> ...
```

hasta alcanzar `max_steps`.

## `langgraph_beam_stanza.py`

### 1. Constantes globales

Definen el comportamiento del experimento:

- `BASE_URL`: URL local de Ollama.
- `GENERATION_MODEL`: modelo de Ollama usado.
- `TEMPERATURE`, `RETRY_TEMPERATURE`, `NUM_PREDICT`: parametros de generacion.
- `ALPHA`: peso historico para agregar puntuaciones.
- `ENABLE_LOCAL_METER_REPAIR`: activa o desactiva la reparacion metrica inicial.
- `ENABLE_OUTER_RHYME_REPAIR`: activa o desactiva la reparacion de rima exterior A.
- `ENABLE_POST_A_RHYME_METER_REPAIR`: activa o desactiva la reparacion metrica posterior a rima A.
- `ENABLE_INNER_RHYME_REPAIR`: activa o desactiva la reparacion de rima interior B.
- `ENABLE_POST_B_RHYME_METER_REPAIR`: activa o desactiva la reparacion metrica posterior a rima B.
- `ENABLE_BEAM_ELITISM`: permite conservar buenos beams aunque sus descendientes sean peores.
- `ENABLE_PROMPT_CONSTRAINT_PROTECTION`: anade al prompt instrucciones para conservar restricciones ya satisfechas.

La reparacion de rima se apoya en prompts y diagnosticos formales, no en listas fijas de palabras finales.

### 2. Estado del grafo

`BeamSearchState` define la estructura del estado que viaja por LangGraph:

- `question`: prompt inicial o tarea poetica.
- `beams`: beams vivos en cada iteracion.
- `candidates`: estrofas candidatas generadas en el paso actual.
- `trace`: historial estructurado del proceso.
- `step`: paso actual.
- `max_steps`: numero maximo de iteraciones.
- `k`: numero de beams conservados tras la poda.

### 3. Utilidad general Ollama

`chat_ollama(...)`

- Construye la peticion HTTP a `/api/chat`.
- Envia el modelo, los mensajes, la temperatura y `num_predict`.
- Devuelve el contenido textual generado por Ollama.
- Es la funcion comun usada por generacion principal y reparaciones.

### 4. Limpieza y parseo JSON

`clean_generated_verse(...)`

- Limpia numeracion, guiones y comillas alrededor de un verso.

`clean_generated_stanza(...)`

- Aplica limpieza a los versos generados.
- Elimina lineas vacias.
- Conserva como maximo 4 versos.

`_parse_stanza_payload(...)`

- Interpreta un objeto JSON devuelto por el modelo.
- Extrae una estrofa desde campos como `stanza` o `verses`.

`parse_stanza_candidates_response(...)`

- Parsea la respuesta JSON del LLM.
- Extrae varias candidatas.
- Normaliza sus versos.
- Si la respuesta no cumple formato, lanza error para permitir reintento.

### 5. Construccion del prompt

`format_numbered_verses(...)`

- Formatea una estrofa como lista numerada.
- Se usa en prompts y salidas por consola.

`summarize_feedback_for_prompt(...)`

- Resume el feedback formal para incluirlo en prompts de correccion.

`build_constraint_protection_prompt(...)`

- Detecta restricciones ya satisfechas.
- Si una rima o metrica ya esta bien, genera instrucciones para conservar esos versos.
- Ayuda a no degradar buenos beams en iteraciones posteriores.

`_build_stanza_generation_messages(...)`

- Construye los mensajes de sistema y usuario para Ollama.
- En el primer paso pide una estrofa desde cero.
- En pasos posteriores pide corregir una estrofa previa usando feedback formal.
- Exige salida JSON estructurada.

### 6. Generacion con Ollama

`generate_stanza_candidates_with_ollama(...)`

- Llama a `_build_stanza_generation_messages(...)`.
- Envia el prompt a Ollama mediante `chat_ollama(...)`.
- Parsea la respuesta con `parse_stanza_candidates_response(...)`.
- Si falla el JSON, reintenta con temperatura mas baja.
- Devuelve una lista de estrofas candidatas.

### 7. Reparacion local de metrica

`syllable_distance_to_target(...)`

- Calcula la distancia absoluta entre las silabas de un verso y el objetivo de 11.

`parse_meter_repair_variants_response(...)`

- Parsea variantes de reparacion metrica generadas por el LLM.

`_build_meter_repair_messages(...)`

- Construye el prompt para reescribir un unico verso que no tiene 11 silabas.
- Pide no modificar el resto de la estrofa.

`generate_meter_repair_variants_with_ollama(...)`

- Solicita variantes metricas para un verso concreto.

`_build_meter_repair_messages_preserving_final_word(...)`

- Construye un prompt de reparacion metrica que obliga a conservar una palabra final concreta.
- Se usa tras reparar rima, para no perder la palabra que consigue rimar.

`generate_meter_repair_variants_preserving_final_word_with_ollama(...)`

- Genera variantes metricas que deben terminar exactamente en una palabra final dada.

`build_meter_repair_report(...)`

- Crea la estructura del informe de reparacion metrica inicial.

`build_post_rhyme_meter_repair_report(...)`

- Crea la estructura del informe de reparacion metrica posterior a rima.

`repair_verse_meter_preserving_final_word_with_ollama(...)`

- Repara metricamente un unico verso.
- Obliga a conservar la palabra final.
- Acepta solo variantes que mantengan esa palabra y no empeoren la distancia metrica.

`repair_stanza_meter_with_ollama(...)`

- Recorre los versos de la estrofa.
- Detecta versos que no tienen 11 silabas.
- Pide variantes al LLM para esos versos.
- Sustituye un verso solo si mejora su distancia a 11 silabas.

### 8. Reparacion de rima exterior A

La rima exterior A corresponde a:

```text
verso 1 = ancla
verso 4 = verso reparable
```

`parse_rhyme_repair_variants_response(...)`

- Parsea respuestas JSON con variantes de un verso.

`parse_outer_rhyme_repair_variants_response(...)`

- Alias especializado para variantes de rima exterior A.

`build_rhyme_target_hint(...)`

- Explica al LLM que la rima objetivo es una terminacion de rima, no una palabra obligatoria.
- Pide usar una palabra real completa que comparta esa rima.

`_build_outer_rhyme_repair_messages(...)`

- Construye el prompt para reescribir solo el verso 4.
- Mantiene el verso 1 como ancla.
- Pide que el verso 4 rime consonantemente con el verso 1.
- Evita pedir palabras hardcodeadas.

`generate_outer_rhyme_repair_variants_with_ollama(...)`

- Solicita al LLM variantes del verso 4.

`build_outer_rhyme_repair_report(...)`

- Crea la estructura del informe de reparacion de rima A.

`repair_stanza_outer_rhyme_with_ollama(...)`

- Si la rima A ya es correcta, no hace nada.
- Si falla, genera variantes del verso 4.
- Acepta una variante si corrige rima A y no empeora la metrica.
- Si una variante corrige la rima pero falla metricamente, puede activar reparacion metrica posterior a A conservando la palabra final.

### 9. Reparacion de rima interior B

La rima interior B corresponde a:

```text
verso 2 = ancla
verso 3 = verso reparable
```

`_build_inner_rhyme_repair_messages(...)`

- Construye el prompt para reescribir solo el verso 3.
- Mantiene el verso 2 como ancla.
- Pide que el verso 3 rime consonantemente con el verso 2.

`generate_inner_rhyme_repair_variants_with_ollama(...)`

- Solicita al LLM variantes del verso 3.

`build_inner_rhyme_repair_report(...)`

- Crea la estructura del informe de reparacion de rima B.

`repair_stanza_inner_rhyme_with_ollama(...)`

- Si la rima B ya es correcta, no hace nada.
- Si falla, genera variantes del verso 3.
- Acepta una variante si corrige rima B y no empeora la metrica.
- Si una variante corrige la rima pero falla metricamente, puede activar reparacion metrica posterior a B conservando la palabra final.

### 10. Resumenes y trazas

`_get_nested_metric(...)`

- Obtiene valores internos de diccionarios de metricas.

`summarize_metrics(...)`

- Resume numero de versos, endecasilabos, rima y score.

`summarize_meter_repair_report(...)`

- Resume si la reparacion metrica se intento, mejoro algo y cambio la estrofa.

`summarize_outer_rhyme_repair_report(...)`

- Resume la reparacion de rima A.

`summarize_inner_rhyme_repair_report(...)`

- Resume la reparacion de rima B.

`summarize_outer_rhyme_diagnosis(...)`

- Resume si la rima exterior AXYA es correcta.

`summarize_inner_rhyme_diagnosis(...)`

- Resume si la rima interior B es correcta.

`build_trace_entry(...)`

- Construye entradas JSON compactas para la traza.
- Incluye estrofa, scores, metricas, diagnosticos y reportes.

`get_main_feedback_line(...)`

- Extrae la primera linea relevante del feedback formal.

`is_preservable_beam(...)`

- Decide si un beam merece preservarse por elitismo.

`mark_beam_as_elite_candidate(...)`

- Marca una copia del beam como preservada por elitismo.

`deduplicate_beams_by_stanza(...)`

- Elimina beams duplicados con la misma estrofa.

### 11. Nodos de LangGraph

`expand_node(...)`

- Genera nuevas candidatas desde cada beam actual.
- En el primer paso genera desde cero.
- En pasos posteriores corrige estrofas previas.

`score_node(...)`

- Aplica el pipeline formal a cada candidata:
  - reparacion metrica inicial, si esta activada;
  - diagnostico y reparacion de rima A, si esta activada;
  - diagnostico y reparacion de rima B, si esta activada;
  - evaluacion final con `evaluate_stanza_abba(...)`.
- Calcula score del paso y score agregado.

`prune_node(...)`

- Construye el conjunto de seleccion.
- Puede anadir beams preservados por elitismo.
- Ordena candidatos por score.
- Conserva los `k` mejores.

### 12. Agregacion de puntuaciones

`aggregate_scores(...)`

- Agrega el historial de scores de un beam.
- Usa `ALPHA` para ponderar el rendimiento historico.

### 13. Decision de continuacion

`should_continue(...)`

- Termina si se alcanza `max_steps`.
- Termina si no quedan beams.
- En caso contrario vuelve a `expand`.

### 14. Guardado de resultados

`format_execution_time(...)`

- Convierte segundos a un texto legible.

`save_final_result(...)`

- Crea la carpeta `outputs/` si no existe.
- Guarda:
  - `final_stanza_*.txt`;
  - `final_stanza_metrics_*.json`;
  - `final_stanza_trace_*.json`.
- El tiempo total de ejecucion se muestra solo en el TXT.

### 15. Main

`main()`

- Construye el grafo LangGraph.
- Define el prompt inicial.
- Inicializa el estado con un beam vacio.
- Lanza `graph.invoke(initial_state)`.
- Mide el tiempo total.
- Imprime el resultado final por consola.
- Guarda los archivos de salida.

## `sonnet_metrics.py`

### Normalizacion y versos

`normalize_text(...)`

- Normaliza espacios.

`normalize_verses_input(...)`

- Acepta texto o lista de versos.
- Devuelve una lista limpia de versos no vacios.

`evaluate_verse_count(...)`

- Funcion historica no central en el flujo actual.
- No es central en el flujo actual.

### Limpieza de texto

`remove_punctuation(...)`

- Elimina signos de puntuacion.

`clean_verse(...)`

- Normaliza un verso para analisis.

`split_words(...)`

- Divide un verso en palabras limpias.

`get_last_word(...)`

- Devuelve la ultima palabra de un verso.

`get_last_words(...)`

- Devuelve las ultimas palabras de una lista de versos.

### Acento y silabas

`has_written_accent(...)`

- Detecta si una palabra contiene tilde escrita.

`get_stressed_vowel_index(...)`

- Estima la vocal tonica de una palabra.

`split_word_syllables(...)`

- Divide una palabra en silabas usando `pyphen`.

`count_word_syllables(...)`

- Cuenta silabas de una palabra.

`_get_stressed_syllable_index(...)`

- Ubica la silaba tonica dentro de una palabra.

`get_word_stress_type(...)`

- Clasifica la palabra final como aguda, llana, esdrujula o desconocida.

`get_final_word_syllable_adjustment(...)`

- Aplica el ajuste metrico por palabra final:
  - aguda: +1;
  - llana: 0;
  - esdrujula: -1.

`count_raw_verse_syllables(...)`

- Cuenta silabas sin sinalefa.

`count_verse_syllables_without_sinalefa(...)`

- Cuenta silabas de verso sin aplicar sinalefas.

`analyze_verse_syllables_basic(...)`

- Devuelve analisis silabico basico sin sinalefa.

### Sinalefa

`is_vowel_sound_start(...)`

- Detecta si una palabra empieza por sonido vocalico.

`is_vowel_sound_end(...)`

- Detecta si una palabra termina por sonido vocalico.

`get_sinalefa_pairs(...)`

- Encuentra pares de palabras consecutivas que pueden formar sinalefa.

`count_sinalefas(...)`

- Cuenta sinalefas detectadas.

`count_verse_syllables(...)`

- Funcion principal para contar silabas metricas de un verso.
- Combina conteo base, sinalefa y ajuste por palabra final.

`analyze_verse_syllables(...)`

- Devuelve analisis detallado del conteo silabico.

### Rima

`strip_accents(...)`

- Elimina tildes para comparar rimas.

`_get_syllable_start_positions(...)`

- Calcula posiciones iniciales de silabas dentro de una palabra.

`_find_stressed_vowel_offset_in_syllable(...)`

- Busca vocal tonica dentro de una silaba.

`get_last_stressed_vowel_position(...)`

- Localiza la ultima vocal tonica de una palabra.

`extract_consonant_rhyme(...)`

- Extrae la rima consonante desde la vocal tonica hasta el final.

`extract_verse_rhyme(...)`

- Extrae la rima consonante de la ultima palabra de un verso.

`extract_rhymes(...)`

- Extrae rimas de una lista de versos.

`_scheme_letter(...)`

- Genera letras A, B, C... para esquemas de rima.

`build_rhyme_scheme(...)`

- Construye un esquema de rima a partir de rimas detectadas.

`describe_rhyme_errors(...)`

- Genera mensajes de error cuando la rima no coincide con el esquema esperado.

`evaluate_rhyme_scheme(...)`

- Funcion historica no central en el flujo actual.
- No es la evaluacion principal actual.

### Evaluacion de silabas

`describe_syllable_errors(...)`

- Genera mensajes para versos que no tienen 11 silabas.

`evaluate_syllable_count(...)`

- Funcion generica de evaluacion silabica.

`_clamp_score(...)`

- Acota una puntuacion al rango valido.

### Funciones heredadas no usadas por el flujo principal

`build_sonnet_feedback(...)`

- Construye feedback para el evaluador historico basado en 14 versos.

`evaluate_sonnet(...)`

- Evalua una estructura poetica de 14 versos.
- Se conserva por compatibilidad historica, pero no es el flujo principal actual.

### Evaluacion actual de estrofa ABBA

`evaluate_stanza_verse_count(...)`

- Comprueba que la estrofa tenga exactamente 4 versos.

`describe_stanza_syllable_errors(...)`

- Genera mensajes especificos para errores metricos de una estrofa.

`evaluate_stanza_syllable_count(...)`

- Evalua si los 4 versos son endecasilabos.

`evaluate_stanza_rhyme_scheme(...)`

- Evalua si la estrofa cumple rima ABBA.

`diagnose_stanza_outer_rhyme(...)`

- Diagnostica la pareja exterior A:
  - verso 1;
  - verso 4.

`diagnose_stanza_inner_rhyme(...)`

- Diagnostica la pareja interior B:
  - verso 2;
  - verso 3.

`build_stanza_feedback(...)`

- Construye feedback formal legible para la estrofa.

`evaluate_stanza_abba(...)`

- Funcion principal de evaluacion actual.
- Integra:
  - numero de versos;
  - metrica;
  - rima ABBA;
  - score normalizado;
  - score sobre 20;
  - feedback formal.

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

## Limitaciones conocidas

- El conteo metrico es aproximado.
- La sinalefa se modela de forma basica.
- La deteccion de vocal tonica en palabras sin tilde es heuristica.
- El LLM puede incumplir formato JSON o restricciones poeticas.
- La rima B suele ser mas dificil que la rima A.
- Aumentar reparaciones y variantes mejora la busqueda, pero incrementa el tiempo de ejecucion.

## Estado actual

El sistema actual permite:

- generar estrofas completas de 4 versos;
- evaluar formalmente metrica y rima ABBA;
- ejecutar Beam Search con LangGraph;
- activar o desactivar reparaciones para comparar configuraciones;
- preservar buenos beams mediante elitismo;
- guardar resultados finales y trazas detalladas.

El codigo ya no contiene:

- generacion de estructuras de 14 versos como objetivo principal;
- listas fijas de palabras finales para forzar rimas;
- reparacion B condicionada por palabras predefinidas.
