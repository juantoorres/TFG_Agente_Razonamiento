# Contexto del proyecto

Este documento resume el estado actual del sistema de generación automática de sonetos con LLMs, LangGraph, Ollama y Beam Search. Sirve como contexto base para continuar el trabajo en un nuevo chat.

## 1. Propósito de cada modulo

### `src/sonnet_metrics.py`

- Implementa la evaluación formal objetiva de sonetos en español.
- Resuelve tres aspectos principales:
  - extensión formal: exactamente 14 versos,
  - métrica aproximada: versos endecasílabos,
  - rima consonante: patrón `ABBA ABBA CDC CDC`.
- Devuelve puntuaciones, errores y feedback textual para guiar nuevas generaciones.
- Es el módulo de scoring formal que usa el agente.

### `src/langgraph_beam_ollama.py`

- Implementa el agente con LangGraph y Beam Search.
- Genera sonetos candidatos usando Ollama.
- Limpia problemas superficiales del formato generado por el LLM.
- Evalua cada candidato con `evaluate_sonnet`.
- Conserva los mejores beams y usa el feedback formal para pedir versiones corregidas.
- Exporta resultados finales y trazas completas a la carpeta `outputs`.

## 2. Funciones principales

### Funciones principales en `sonnet_metrics.py`

#### Limpieza y normalización

- `normalize_text(text: str) -> str`
  - Convierte a minúsculas, elimina espacios externos y compacta espacios internos.
  - Conserva tildes, diéresis y `ñ`.

- `remove_punctuation(text: str) -> str`
  - Elimina puntuacion común.
  - Conserva letras acentuadas, `ü`, `ñ` y espacios.

- `clean_verse(verse: str) -> str`
  - Aplica normalizacion y retirada de puntuación.
  - Se usa como base para dividir palabras y analizar versos.

- `split_words(verse: str) -> list[str]`
  - Devuelve las palabras limpias de un verso.
  - Alimenta conteo silábico, rima y extracción de última palabra.

- `normalize_verses_input(sonnet: str | list[str]) -> list[str]`
  - Acepta un soneto como texto completo o lista.
  - Devuelve una lista de versos sin líneas vacias.

#### Última palabra y acentuación

- `get_last_word(verse: str) -> str | None`
  - Devuelve la última palabra útil de un verso.
  - Se usa para ajuste métrico final y rima.

- `get_last_words(verses: list[str]) -> list[str | None]`
  - Aplica `get_last_word` a varios versos.

- `has_written_accent(word: str) -> bool`
  - Detecta si una palabra tiene tilde escrita.

- `get_stressed_vowel_index(word: str) -> int | None`
  - Devuelve el índice de la vocal con tilde si existe.

- `get_word_stress_type(word: str) -> str`
  - Clasifica una palabra como `aguda`, `llana` o `esdrujula`.
  - Usa reglas generales del español y `pyphen` cuando hay tilde.

- `get_final_word_syllable_adjustment(word: str) -> int`
  - Devuelve `+1` para aguda, `0` para llana y `-1` para esdrújula.
  - Se aplica al conteo métrico del verso.

#### Sílabas y sinalefa

- `split_word_syllables(word: str) -> list[str]`
  - Separa una palabra en sílabas aproximadas usando `pyphen.Pyphen(lang="es_ES")`.

- `count_word_syllables(word: str) -> int`
  - Cuenta sílabas aproximadas de una palabra.

- `count_raw_verse_syllables(verse: str) -> int`
  - Suma sílabas de palabras sin ajuste final ni sinalefa.

- `count_verse_syllables_without_sinalefa(verse: str) -> int`
  - Aplica el ajuste de palabra final, pero no sinalefa.

- `is_vowel_sound_start(word: str) -> bool`
  - Detecta si una palabra empieza por sonido vocálico.
  - Considera vocales y `h` muda seguida de vocal.

- `is_vowel_sound_end(word: str) -> bool`
  - Detecta si una palabra termina en sonido vocálico.

- `get_sinalefa_pairs(words: list[str]) -> list[tuple[str, str]]`
  - Devuelve pares de palabras consecutivas donde hay sinalefa.

- `count_sinalefas(words: list[str]) -> int`
  - Cuenta sinalefas básicas.

- `count_verse_syllables(verse: str) -> int`
  - Función principal de conteo métrico de verso.
  - Cuenta sílabas, aplica ajuste final y resta sinalefas.

- `analyze_verse_syllables(verse: str) -> dict[str, object]`
  - Devuelve una traza detallada del conteo de un verso.

#### Rima

- `strip_accents(text: str) -> str`
  - Elimina tildes de vocales y convierte a minúsculas.
  - Conserva `ñ`.

- `get_last_stressed_vowel_position(word: str) -> int | None`
  - Localiza la vocal tónica de una palabra.

- `extract_consonant_rhyme(word: str) -> str`
  - Extrae la rima consonante desde la vocal tónica hasta el final.

- `extract_verse_rhyme(verse: str) -> str | None`
  - Extrae la rima consonante de la última palabra del verso.

- `extract_rhymes(verses: list[str]) -> list[str | None]`
  - Devuelve la rima de cada verso.

- `build_rhyme_scheme(rhymes: list[str | None]) -> list[str | None]`
  - Convierte rimas reales en letras: `A`, `B`, `C`, etc.

- `evaluate_rhyme_scheme(verses: list[str]) -> dict[str, object]`
  - Compara el esquema real con `ABBA ABBA CDC CDC`.

#### Evaluación formal

- `evaluate_verse_count(sonnet: str | list[str]) -> dict[str, object]`
  - Evalúa si hay exactamente 14 versos.

- `evaluate_syllable_count(verses: list[str]) -> dict[str, object]`
  - Evalúa cuantos versos tienen 11 sílabas métricas.

- `describe_syllable_errors(verses: list[str], counts: list[int]) -> list[str]`
  - Genera errores legibles para versos no endecasílabos.

- `describe_rhyme_errors(actual_scheme, expected_pattern) -> list[str]`
  - Genera errores legibles del esquema de rima.

- `build_sonnet_feedback(evaluation: dict[str, object]) -> str`
  - Construye feedback en españool para el LLM generador.

- `evaluate_sonnet(sonnet: str | list[str]) -> dict[str, object]`
  - Función principal del módulo.
  - Combina extensión, sílabas y rima.
  - Devuelve `score`, `score_20`, métricas parciales, errores y feedback.

### Funciones principales en `langgraph_beam_ollama.py`

#### Comunicación con Ollama

- `chat_ollama(model, messages, temperature, num_predict) -> str`
  - Llama a Ollama mediante HTTP.
  - Fuerza salida JSON con `format: "json"`.

- `generate_sonnet_candidates_with_ollama(question, previous_sonnet, feedback, num_candidates=3) -> list[dict[str, Any]]`
  - Genera candidatos de soneto.
  - Si no hay soneto previo, genera desde cero.
  - Si hay soneto previo, genera versiones corregidas usando feedback formal.
  - Tiene retry con temperatura menor y prompt más estricto.

- `_build_sonnet_generation_messages(...) -> list[dict[str, str]]`
  - Construye el prompt de generación.
  - Incluye restricciones formales, estrategia de rima, longitud de verso y formato JSON obligatorio.

- `format_numbered_verses(verses: list[str]) -> str`
  - Formatea un soneto previo con numeración para el prompt.

- `summarize_feedback_for_prompt(feedback: str, max_chars=2500) -> str`
  - Recorta feedback largo para no saturar el prompt.

#### Parseo y limpieza de salida LLM

- `clean_generated_verse(verse: str) -> str`
  - Limpia numeración inicial, comillas exteriores y espacios duplicados.

- `clean_generated_sonnet(verses: list[str]) -> list[str]`
  - Limpia todos los versos y elimina vacíos.

- `parse_sonnet_response(raw_response: str) -> list[str]`
  - Parsea una respuesta JSON con clave `verses`.

- `parse_sonnet_candidates_response(raw_response: str, num_candidates: int) -> list[dict[str, Any]]`
  - Parsea el JSON global con `candidates`.
  - Devuelve candidatos con `sonnet` y `generation_reasoning`.

#### Beam Search y LangGraph

- `BeamSearchState`
  - Estado compartido por LangGraph.
  - Campos: `question`, `beams`, `candidates`, `trace`, `step`, `max_steps`, `k`.

- `expand_node(state: BeamSearchState) -> dict`
  - Genera candidatos nuevos para cada beam.
  - Usa soneto previo y feedback si existen.
  - Registra fase `expand` en la traza.

- `score_node(state: BeamSearchState) -> dict`
  - Evalúa cada candidato con `evaluate_sonnet`.
  - Actualiza `score_history`.
  - Calcula score global agregado con `aggregate_scores`.
  - Guarda feedback y métricas completas.
  - Registra fase `score` en la traza.

- `prune_node(state: BeamSearchState) -> dict`
  - Ordena candidatos por score global.
  - Selecciona los `k` mejores beams.
  - Registra fase `prune` en la traza.

- `should_continue(state: BeamSearchState) -> str`
  - Decide si el grafo continúa o termina según `step` y `max_steps`.

- `aggregate_scores(score_history: list[float], alpha=ALPHA) -> float`
  - Agrega el historial de puntuaciones con media geométrica normalizada.

#### Logs, resumen y exportación

- `summarize_metrics(metrics: dict[str, Any]) -> str`
  - Resume métricas en una linea.
  - Ejemplo: `versos=14/14 | endecasilabos=5/14 | rima=6/14 | score=8.57/20`.

- `build_trace_entry(item: dict[str, Any]) -> dict[str, Any]`
  - Crea entradas compactas para la traza.

- `save_final_result(best_beam, trace, output_dir="outputs") -> None`
  - Guarda tres archivos con timestamp:
    - soneto final en `.txt`,
    - métricas completas en `.json`,
    - traza completa en `.json`.

- `run_cleaning_smoke_tests() -> None`
  - Prueba manual de limpieza de versos con `--test-cleaning`.

- `main()`
  - Construye el grafo LangGraph.
  - Define el estado inicial.
  - Ejecuta Beam Search.
  - Imprime resultado final.
  - Exporta resultados a `outputs`.

## 3. Flujo de ejecución

1. `main()` crea un `StateGraph` con tres nodos:
   - `expand`,
   - `score`,
   - `prune`.

2. El estado inicial contiene:
   - una tarea de generación de soneto clásico,
   - un beam inicial sin soneto,
   - `score = 0.0`,
   - feedback inicial,
   - traza vacía.

3. `expand_node` genera candidatos:
   - si el beam no tiene soneto, pide una generación inicial;
   - si el beam ya tiene soneto, pide una versión corregida usando feedback.

4. Las respuestas de Ollama se parsean como JSON:
   - se espera una lista de candidatos,
   - cada candidato tiene `verses` y `generation_reasoning`,
   - se limpian numeraciones, comillas y espacios.

5. `score_node` evalua cada candidato con `evaluate_sonnet`.

6. `score_node` actualiza:
   - `step_score`,
   - `score_history`,
   - `score` agregado,
   - `feedback`,
   - `metrics`.

7. `prune_node` selecciona los `k` mejores candidatos según score global.

8. El ciclo continua hasta `max_steps`.

9. Al final:
   - se imprime el mejor beam,
   - se guarda el soneto,
   - se guardan métricas,
   - se guarda la traza completa.

## 4. Sistema de evaluacion (`sonnet_metrics`)

### Metricas implementadas

- Número de versos:
  - objetivo: 14.
  - score: `1.0` si hay 14, `0.0` si no.

- Computo silabico:
  - objetivo: 11 sílabas métricas por verso.
  - usa `pyphen` para sílabas aproximadas.
  - aplica regla de palabra final:
    - aguda suma 1,
    - llana suma 0,
    - esdrújula resta 1.
  - aplica sinalefa básica entre palabras consecutivas.

- Rima consonante:
  - extrae rima desde la vocal tónica de la última palabra.
  - elimina tildes para comparar.
  - construye esquema de letras.
  - compara con `ABBA ABBA CDC CDC`.

### Combinacion de scores

`evaluate_sonnet` usa pesos:

- número de versos: `0.2`,
- sílabas: `0.4`,
- rima: `0.4`.

Devuelve:

- `score`: valor entre 0 y 1,
- `score_20`: score sobre 20,
- métricas parciales,
- errores agregados,
- feedback textual.

## 5. Integración actual entre ambos módulos

### Ya conectado

- `langgraph_beam_ollama.py` importa `evaluate_sonnet` desde `sonnet_metrics.py`.
- `score_node` usa `evaluate_sonnet(candidate["sonnet"])`.
- El feedback producido por `evaluate_sonnet` se guarda en el beam.
- En el siguiente `expand_node`, ese feedback se pasa al prompt de Ollama.
- Las métricas completas quedan guardadas en `metrics`.
- El resumen compacto usa los campos de `metrics`.
- La exportación guarda métricas y traza.

### Aún no conectado o mejorable

- `score_candidate_with_ollama` sigue existiendo como función antigua orientada a razonamiento.
- `SCORING_MODEL` queda reservado para una futura fase de LLM-as-a-judge.
- No hay evaluación subjetiva de belleza, coherencia poética o riqueza léxica.
- No hay reparacion automática programatica de métrica/rima; solo feedback textual al LLM.

## 6. Decisiones de diseñoo importantes

- Beam Search opera sobre diccionarios de sonetos, no sobre clases.
- Cada beam contiene:
  - `sonnet`,
  - `score`,
  - `step_score`,
  - `score_history`,
  - `feedback`,
  - `metrics`,
  - `generation_reasoning`.

- El scoring actual es objetivo y formal.
- La generación se hace con Ollama y prompt JSON estricto.
- La salida de Ollama se limpia superficialmente antes de evaluar.
- Los errores formales se convierten en feedback para el siguiente ciclo.
- La traza se guarda para poder justificar la evolución experimental.
- Los archivos de salida usan timestamps y no sobrescriben resultados previos.
- No se eliminan automáticamente resultados anteriores.

## 7. Estado actual del proyecto

### Funciona correctamente

- Limpieza básica de versos y palabras.
- Conteo silábico aproximado con `pyphen`.
- Sinalefa básica.
- Extracción de rima consonante.
- Evaluación formal global.
- Generación de candidatos con Ollama en formato JSON.
- Uso de feedback formal en iteraciones posteriores.
- Beam Search con `expand`, `score` y `prune`.
- Resumen compacto de métricas en terminal.
- Exportación de resultados finales.
- Exportación de traza completa del proceso.

### En desarrollo

- Calidad real de los sonetos generados.
- Afinamiento del prompt para lograr versos endecasílabos.
- Mejora de rima consonante en generaciones sucesivas.
- Posible integración futura de evaluación subjetiva.

### Problemas conocidos

- `pyphen` es aproximado y no resuelve todos los casos métricos del español poético.
- La sinalefa es básica:
  - no trata sinalefa triple,
  - hiatos poéticos deliberados,
  - sinéresis,
  - diéresis.

- El conteo de palabras como `río` o diptongos/hiatos puede no coincidir siempre con criterio poético.
- La rima depende de una aproximación para localizar la vocal tónica en palabras sin tilde.
- Los LLMs locales pueden ignorar restricciones estrictas de métrica.
- El prompt puede mejorar, pero no conviene sobrediseñarlo en exceso.
- Hay funciones antiguas orientadas a razonamiento matemático que aun permanecen por compatibilidad o reutilización futura.
- Puede aparecer un warning externo de `langchain_core` con Python 3.14.

## 8. Recomendaciones para continuar

### Siguientes pasos concretos

1. Ejecutar varias pruebas reales con Ollama y revisar archivos en `outputs`.
2. Comparar trazas para ver si el feedback mejora o empeora los candidatos.
3. Reducir `max_steps` o `k` si el tiempo de ejecucion es alto.
4. Ajustar el prompt solo con cambios pequeños y medibles.
5. Revisar casos donde `evaluate_sonnet` penaliza demasiado por limitaciones de métrica.
6. Crear una tabla experimental con:
   - numero de ejecucion,
   - modelo,
   - `k`,
   - `max_steps`,
   - mejor score,
   - observaciones.

### Cambios mínimos recomendados

- Añadir una opción de configuración simple para `max_steps`, `k` y modelo.
- Guardar también parametros de ejecución en el JSON final.
- Añadir una función que imprima los errores principales del mejor beam.
- Revisar si conviene bajar o subir `temperature` según resultados reales.
- Mantener la evaluación formal separada de la generación.

### Evitar por ahora

- No introducir una arquitectura demasiado compleja.
- No añadir LLM-as-a-judge hasta estabilizar la evaluación formal.
- No intentar resolver toda la métrica española avanzada.
- No mezclar reparaciones programáticas fuertes con generación LLM todavía.
- No borrar outputs antiguos: el historico con timestamps es útil para la memoria.

### Prioridad recomendada para el TFG

1. Consolidar pipeline funcional completo.
2. Ejecutar experimentos repetibles.
3. Documentar limitaciones de métricas y LLM local.
4. Mostrar trazas como evidencia del razonamiento iterativo.
5. Solo después, valorar mejoras subjetivas o métricas más avanzadas.
