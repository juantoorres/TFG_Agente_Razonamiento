# Paso 7: reparacion de rima interior B reescribiendo solo el verso 3

## Objetivo del paso

El objetivo de este paso es anadir reparacion de la rima interior B siguiendo la misma metodologia que ya se usa para la rima exterior A.

La rima exterior A ya funciona de forma razonable:

```text
Verso 1 = ancla
Verso 4 = verso reparable
```

Para la rima interior B se debe aplicar el mismo criterio:

```text
Verso 2 = ancla
Verso 3 = verso reparable
```

Es decir:

```text
1. Se diagnostica si el verso 3 rima consonantemente con el verso 2.
2. Si la rima B ya es correcta, no se hace nada.
3. Si la rima B falla, se pide al LLM que reescriba solo el verso 3.
4. El verso 3 debe rimar con el verso 2.
5. Si existen palabras finales candidatas, se obliga a terminar en una de ellas.
6. Se evita repetir exactamente la palabra final del verso 2 cuando haya alternativas.
7. Se acepta una variante solo si corrige B y no empeora la distancia metrica del verso 3.
```

Este paso no debe introducir una metodologia nueva. Debe ser una replica controlada de la reparacion A, adaptada a la pareja interior B.

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

No debe tocar:

- `src/sonnet_metrics.py`;
- la extraccion de rima;
- el contador de silabas;
- la reparacion metrica;
- la reparacion de rima exterior A;
- `evaluate_stanza_abba`;
- `aggregate_scores(...)`;
- los pesos de scoring;
- el Beam Search;
- `k`;
- `max_steps`;
- `alpha`;
- la medicion de tiempo del Paso 6.

Este paso debe afectar solo a:

```text
Rima interior B: verso 2 con verso 3
```

No debe modificar la estrategia de:

```text
Rima exterior A: verso 1 con verso 4
```

## Punto de partida actual

Ya existe diagnostico de rima interior B mediante:

```python
diagnose_stanza_inner_rhyme(...)
```

Esta funcion devuelve, entre otros campos:

```python
{
    "target_pattern": "XBBY",
    "required_pair": [2, 3],
    "has_enough_verses": ...,
    "is_valid": ...,
    "verse_2": {
        "text": ...,
        "last_word": ...,
        "rhyme": ...,
    },
    "verse_3": {
        "text": ...,
        "last_word": ...,
        "rhyme": ...,
    },
    "target_rhyme": ...,
    "current_rhyme": ...,
    "errors": ...,
    "feedback": ...,
}
```

Tambien existen ya funciones auxiliares usadas por la reparacion A:

- `get_candidate_final_words_for_rhyme(...)`
- `get_last_word(...)`
- `count_verse_syllables(...)`
- `syllable_distance_to_target(...)`
- `summarize_inner_rhyme_diagnosis(...)`
- `format_numbered_verses(...)`
- `clean_generated_verse(...)`
- `chat_ollama(...)`

Estas funciones se pueden reutilizar en la reparacion B.

## Principio de reutilizacion

Este paso debe reutilizar todo lo que ya se ha implementado durante la reparacion de rima A y durante el Paso 5, siempre que sea posible.

No se debe duplicar codigo con el unico cambio de sustituir:

```text
verso 1 -> verso 2
verso 4 -> verso 3
```

Si una funcion nueva tendria practicamente el mismo cuerpo que una funcion ya existente de rima A, se debe preferir una de estas opciones:

```text
1. Convertir la funcion existente en una funcion generica parametrizable.
2. Crear una pequena funcion auxiliar comun y hacer que A y B la usen.
3. Crear solo una envoltura minima si hace falta conservar claridad de nombres.
```

El objetivo es que la reparacion B siga la misma metodologia de A, pero sin copiar bloques enteros de logica.

### Protecciones ya existentes que deben reutilizarse

No se deben volver a implementar las protecciones del Paso 5.

Ya existen mecanismos globales para:

- conservar mejores beams mediante elitismo;
- marcar beams con `preserved_by_elitism`;
- proteger restricciones ya satisfechas en el prompt general;
- conservar versos asociados a rimas ya correctas cuando el diagnostico lo indica.

La reparacion B debe integrarse con esos mecanismos, no duplicarlos.

En particular, `build_constraint_protection_prompt(...)` ya contempla:

```python
inner_rhyme_diagnosis["is_valid"]
```

Por tanto, cuando la rima interior B sea correcta, el prompt general de correccion ya debe proteger los versos 2 y 3. El Paso 7 no debe crear una segunda proteccion paralela para eso.

## Nuevas constantes

Anadir en la zona de constantes globales, junto a las constantes de reparacion A:

```python
# Fase experimental opcional: si la rima interior B falla, intenta
# reescribir solo el verso 3 para que rime con el verso 2.
ENABLE_INNER_RHYME_REPAIR = True
INNER_RHYME_REPAIR_VARIANTS = 5
INNER_RHYME_REPAIR_TEMPERATURE = 0.3
INNER_RHYME_REPAIR_NUM_PREDICT = 200
```

La configuracion debe ser paralela a la de rima A:

```python
ENABLE_OUTER_RHYME_REPAIR = True
OUTER_RHYME_REPAIR_VARIANTS = 5
OUTER_RHYME_REPAIR_TEMPERATURE = 0.3
OUTER_RHYME_REPAIR_NUM_PREDICT = 200
```

## Funciones nuevas o refactorizadas

Crear o refactorizar las siguientes funciones en `src/langgraph_beam_stanza.py`, preferiblemente en la misma zona donde esta la reparacion de rima exterior A.

### 1. Parser comun de variantes de reparacion de rima

La funcion actual:

```python
parse_outer_rhyme_repair_variants_response(...)
```

parsea variantes JSON para reparar un unico verso.

Ese comportamiento sera identico para A y B. Por tanto, no se debe duplicar el parser.

Refactor recomendado:

```python
def parse_rhyme_repair_variants_response(
    raw_response: str,
    max_variants: int,
) -> list[str]:
    ...
```

Debe contener la logica comun que ahora esta en:

```python
parse_outer_rhyme_repair_variants_response(...)
```

Es decir:

1. Parsear JSON.
2. Exigir un objeto JSON.
3. Leer el campo `"variants"`.
4. Exigir que `"variants"` sea una lista.
5. Limpiar cada variante con `clean_generated_verse(...)`.
6. Eliminar variantes vacias.
7. Eliminar duplicados.
8. Limitar a `max_variants`.
9. Lanzar `ValueError` si no hay ninguna variante valida.

Despues, la reparacion A y la reparacion B deben usar esta funcion comun.

Opcionalmente, si se quiere mantener compatibilidad interna de nombres, puede dejarse:

```python
def parse_outer_rhyme_repair_variants_response(...):
    return parse_rhyme_repair_variants_response(...)
```

pero no debe duplicarse la logica.

El formato esperado del LLM debe ser:

```json
{
  "variants": [
    "nuevo verso 3",
    "nuevo verso 3 alternativo"
  ]
}
```

### 2. Ayuda textual generica de rima objetivo

Actualmente existe:

```python
build_rhyme_target_hint(...)
```

Esa funcion ya contiene la logica de:

- explicar que la rima objetivo es una terminacion;
- usar palabras candidatas;
- evitar repetir la palabra ancla si hay alternativas.

No se debe crear una copia completa llamada `build_inner_rhyme_target_hint(...)` si solo cambia el numero de verso.

Refactor recomendado:

```python
def build_rhyme_target_hint(
    anchor_word: str | None,
    target_rhyme: str | None,
    anchor_verse_label: str = "verso 1",
    repaired_verse_label: str = "verso 4",
) -> str:
    ...
```

Con los valores por defecto, la reparacion A debe seguir funcionando igual.

Para B se llamaria con:

```python
build_rhyme_target_hint(
    anchor_word=verse_2_last_word,
    target_rhyme=target_rhyme,
    anchor_verse_label="verso 2",
    repaired_verse_label="verso 3",
)
```

El texto resultante debe quedar adaptado a:

```text
Verso 2 = ancla
Verso 3 = verso reparable
```

Debe explicar que:

- la rima objetivo es una terminacion de rima, no una palabra literal;
- el verso 3 debe terminar en una palabra real;
- el verso 3 debe rimar consonantemente con el verso 2;
- si existen palabras candidatas distintas de la palabra final del verso 2, deben priorizarse;
- debe evitarse que el verso 3 termine con la misma palabra final del verso 2 si existe alternativa.

Debe reutilizar:

```python
get_candidate_final_words_for_rhyme(...)
```

Igual que en A, si hay candidatas distintas del ancla, la ayuda debe mostrarlas.

Si solo existe la palabra ancla como referencia, debe pedir que se busque otra palabra real con la misma rima, si es posible.

### 3. Construccion del prompt de reparacion B

Crear:

```python
def _build_inner_rhyme_repair_messages(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    num_variants: int,
) -> List[Dict[str, str]]:
    ...
```

Esta funcion si debe existir porque el prompt de B tiene versos distintos y nombres distintos.

Sin embargo, debe reutilizar las funciones auxiliares comunes:

- `build_rhyme_target_hint(...)` parametrizada;
- `get_candidate_final_words_for_rhyme(...)`;
- cualquier helper comun que se haya extraido para construir listas de palabras candidatas.

Debe ser equivalente a `_build_outer_rhyme_repair_messages(...)`, pero sin duplicar logica auxiliar innecesaria y con:

```text
Verso 2 como ancla.
Verso 3 como verso a reparar.
```

Debe leer del diagnostico:

```python
verse_2 = inner_rhyme_diagnosis.get("verse_2", {})
verse_3 = inner_rhyme_diagnosis.get("verse_3", {})
target_rhyme = inner_rhyme_diagnosis.get("target_rhyme", "")
current_rhyme = inner_rhyme_diagnosis.get("current_rhyme", "")
```

Y construir:

```python
verse_2_text
verse_2_last_word
current_verse_3
```

Debe calcular candidatas:

```python
candidate_final_words = get_candidate_final_words_for_rhyme(
    anchor_word=verse_2_last_word,
    target_rhyme=target_rhyme,
)
non_anchor_candidate_final_words = [
    word for word in candidate_final_words if word != verse_2_last_word
]
enforced_candidate_final_words = non_anchor_candidate_final_words
```

Si `enforced_candidate_final_words` no esta vacia, el prompt debe incluir una seccion:

```text
Palabras finales candidatas para el nuevo verso 3:
- palabra1
- palabra2
...

Cada variante debe terminar exactamente en una de esas palabras.
No uses otras palabras finales en este paso.
No escribas texto despues de la palabra final.
```

Si no hay candidatas distintas, pero existe palabra final del verso 2, debe incluir:

```text
No hay palabras finales candidatas distintas de la palabra final del verso 2 ("...") en la lista actual.
Busca una palabra real distinta que comparta la misma rima consonante, si es posible.
```

### Prompt del reparador B

El `system_prompt` debe decir:

```text
Eres un asistente de reparacion de rima consonante para poesia espanola.
Tu tarea es proponer variantes de un unico verso: el verso 3.
No reescribas la estrofa completa.
No evalues belleza poetica ni expliques el resultado.
Devuelve solamente JSON valido.
```

El `user_prompt` debe incluir:

```text
Estrofa completa como contexto:
...

Verso 2 como ancla:
...
Palabra final del verso 2: ...
Rima consonante objetivo del verso 2: ...

Orientacion para la palabra final del verso 3:
...

Verso 3 actual:
...
Rima actual del verso 3: ...

Genera exactamente N variantes para reescribir solo el verso 3.
```

Restricciones obligatorias:

```text
- Reescribe solo el verso 3.
- No modifiques el verso 1.
- No modifiques el verso 2.
- No modifiques el verso 4.
- El nuevo verso 3 debe rimar consonantemente con el verso 2.
- La rima objetivo es: ...
- Evita que el verso 3 termine con la misma palabra final que el verso 2 si existe alternativa.
- La rima objetivo es una terminacion de rima, no una palabra obligatoria.
- No copies literalmente la rima objetivo como palabra final si no es una palabra natural.
- Termina el verso 3 con una palabra real que tenga esa rima consonante.
- La palabra final del verso 3 debe ser una palabra completa y natural en espanol.
- No fuerces terminaciones artificiales.
- No termines el verso con fragmentos como "eva", "argos", "ido" o similares si no funcionan como palabra real.
- Intenta que el nuevo verso 3 tenga 11 silabas metricas.
- Devuelve solo variantes del verso 3, no la estrofa completa.
- No anadas titulo.
- No numeres las variantes.
- Responde solo con JSON valido.
```

Si existen palabras finales candidatas, anadir tambien:

```text
- Cada variante debe terminar exactamente en una de las palabras finales candidatas.
- No uses una palabra final distinta de la lista.
- No escribas nada despues de la palabra final candidata.
- La ultima palabra de cada variante debe ser una palabra de la lista.
```

### 4. `generate_inner_rhyme_repair_variants_with_ollama(...)`

Crear:

```python
def generate_inner_rhyme_repair_variants_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    num_variants: int = INNER_RHYME_REPAIR_VARIANTS,
) -> list[str]:
    ...
```

Debe ser equivalente a:

```python
generate_outer_rhyme_repair_variants_with_ollama(...)
```

Pero debe usar:

```python
_build_inner_rhyme_repair_messages(...)
INNER_RHYME_REPAIR_TEMPERATURE
INNER_RHYME_REPAIR_NUM_PREDICT
parse_rhyme_repair_variants_response(...)
```

### 5. `build_inner_rhyme_repair_report(...)`

Crear:

```python
def build_inner_rhyme_repair_report(
    enabled: bool,
    stanza: list[str],
) -> dict[str, Any]:
    ...
```

Debe devolver una estructura paralela a `build_outer_rhyme_repair_report(...)`, pero adaptada a B:

```python
{
    "enabled": enabled,
    "attempted": False,
    "changed": False,
    "reason": None,
    "original_stanza": stanza.copy(),
    "repaired_stanza": stanza.copy(),
    "target_rhyme": None,
    "current_rhyme": None,
    "verse_2": None,
    "candidate_final_words": [],
    "enforced_candidate_final_words": [],
    "original_verse_3": None,
    "original_verse_3_syllables": None,
    "selected_variant": None,
    "selected_variant_syllables": None,
    "variants": [],
    "error": None,
}
```

### 6. `repair_stanza_inner_rhyme_with_ollama(...)`

Crear:

```python
def repair_stanza_inner_rhyme_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    ...
```

Debe ser equivalente a `repair_stanza_outer_rhyme_with_ollama(...)`, pero:

```text
Ancla: verso 2
Verso reparable: verso 3
Indice Python reparable: 2
```

### Flujo interno esperado

1. Crear `report` con `build_inner_rhyme_repair_report(...)`.
2. Copiar `stanza` en `repaired_stanza`.
3. Guardar en el reporte:

```python
report["target_rhyme"] = inner_rhyme_diagnosis.get("target_rhyme")
report["current_rhyme"] = inner_rhyme_diagnosis.get("current_rhyme")
report["verse_2"] = inner_rhyme_diagnosis.get("verse_2")
```

4. Si `ENABLE_INNER_RHYME_REPAIR` esta en `False`, no reparar.
5. Si la estrofa no tiene 4 versos, no reparar.
6. Calcular silabas originales del verso 3:

```python
original_verse_3 = stanza[2]
original_verse_3_syllables = count_verse_syllables(original_verse_3)
original_distance = syllable_distance_to_target(original_verse_3_syllables)
```

7. Si `inner_rhyme_diagnosis["is_valid"]` es `True`, no reparar.
8. Si no hay `target_rhyme`, no reparar.
9. Obtener `anchor_word` desde:

```python
verse_2 = inner_rhyme_diagnosis.get("verse_2", {})
anchor_word = verse_2.get("last_word") if isinstance(verse_2, dict) else None
```

10. Calcular:

```python
candidate_final_words = get_candidate_final_words_for_rhyme(
    anchor_word=anchor_word,
    target_rhyme=target_rhyme,
)
enforced_candidate_final_words = [
    word for word in candidate_final_words if word != anchor_word
]
```

11. Guardar ambas listas en el reporte.
12. Marcar:

```python
report["attempted"] = True
```

13. Pedir variantes con `generate_inner_rhyme_repair_variants_with_ollama(...)`.
14. Para cada variante:

```python
candidate_stanza = stanza.copy()
candidate_stanza[2] = variant
variant_diagnosis = diagnose_stanza_inner_rhyme(candidate_stanza)
variant_syllables = count_verse_syllables(variant)
variant_distance = syllable_distance_to_target(variant_syllables)
inner_rhyme_valid = bool(variant_diagnosis.get("is_valid", False))
metric_not_worse = variant_distance <= original_distance
variant_final_word = get_last_word(variant)
final_word_matches_anchor = bool(anchor_word and variant_final_word == anchor_word)
uses_candidate_final_word = (
    not enforced_candidate_final_words
    or variant_final_word in enforced_candidate_final_words
)
```

15. Aceptar solo si:

```python
is_acceptable = (
    inner_rhyme_valid
    and metric_not_worse
    and uses_candidate_final_word
)
```

16. Guardar en cada variante:

```python
{
    "verse": variant,
    "syllables": variant_syllables,
    "final_word": variant_final_word,
    "final_word_matches_anchor": final_word_matches_anchor,
    "uses_candidate_final_word": uses_candidate_final_word,
    "inner_rhyme_valid": inner_rhyme_valid,
    "inner_rhyme_summary": summarize_inner_rhyme_diagnosis(variant_diagnosis),
    "accepted": False,
    "rejection_reason": None,
}
```

17. Motivos de rechazo, en este orden:

```text
1. No termina en una palabra final candidata distinta del verso 2.
2. No corrige la rima interior B.
3. Empeora la distancia metrica del verso 3.
```

18. Si hay variantes aceptables, seleccionar con el mismo criterio que A:

```python
selected = min(
    acceptable_variants,
    key=lambda item: (
        item["final_word_matches_anchor"],
        item["distance"],
        item["index"],
    ),
)
```

19. Reemplazar solo:

```python
repaired_stanza[2] = selected["verse"]
```

20. Marcar:

```python
report["changed"] = True
report["reason"] = "Se acepto una variante que corrige la rima interior B."
report["selected_variant"] = selected["verse"]
report["selected_variant_syllables"] = selected["syllables"]
report["repaired_stanza"] = repaired_stanza.copy()
```

### Motivos de salida recomendados

Usar mensajes paralelos a la rima A:

```text
Reparacion de rima B desactivada.
La estrofa no tiene 4 versos.
La rima interior B ya es correcta.
No hay rima objetivo para reparar la rima B.
Fallo generando variantes de rima interior B.
Ninguna variante corrige B sin empeorar metrica.
Se acepto una variante que corrige la rima interior B.
```

### 7. `summarize_inner_rhyme_repair_report(...)`

Crear:

```python
def summarize_inner_rhyme_repair_report(report: dict[str, Any]) -> str:
    ...
```

Debe ser equivalente a `summarize_outer_rhyme_repair_report(...)`, pero para B:

```text
reparacion rima B = sin informe
reparacion rima B = desactivada
reparacion rima B = no necesaria
reparacion rima B = activa | intento = True | cambio = True/False
```

## Cambios en `score_node(...)`

Actualmente el flujo debe ser aproximadamente:

```python
stanza, meter_repair_report = repair_stanza_meter_with_ollama(...)
outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
stanza, outer_rhyme_repair_report = repair_stanza_outer_rhyme_with_ollama(...)
evaluation = evaluate_stanza_abba(stanza)
outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
```

Debe pasar a:

```python
stanza, meter_repair_report = repair_stanza_meter_with_ollama(...)

outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
stanza, outer_rhyme_repair_report = repair_stanza_outer_rhyme_with_ollama(...)

inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
stanza, inner_rhyme_repair_report = repair_stanza_inner_rhyme_with_ollama(...)

evaluation = evaluate_stanza_abba(stanza)
outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
```

Orden importante:

```text
1. Reparacion metrica.
2. Reparacion rima exterior A.
3. Reparacion rima interior B.
4. Evaluacion final.
```

No debe anadirse una nueva reparacion metrica despues de B en este paso.

En la salida por terminal de `score_node(...)`, anadir:

```python
print(
    "Reparacion rima B: "
    f"{summarize_inner_rhyme_repair_report(inner_rhyme_repair_report)}"
)
```

El `scored_candidate` debe guardar:

```python
"inner_rhyme_repair_report": inner_rhyme_repair_report,
```

## Cambios en candidatos, beams y trazas

### `expand_node(...)`

Al crear cada candidato, anadir:

```python
"inner_rhyme_repair_report": {},
```

### Estado inicial en `main(...)`

En el beam inicial, anadir:

```python
"inner_rhyme_repair_report": {},
```

### `build_trace_entry(...)`

Leer:

```python
inner_rhyme_repair_report = item.get("inner_rhyme_repair_report", {})
```

Validar que sea dict.

Incluir en la entrada de traza:

```python
"inner_rhyme_repair_summary": summarize_inner_rhyme_repair_report(
    inner_rhyme_repair_report
),
"inner_rhyme_repair_report": inner_rhyme_repair_report,
```

No eliminar ni modificar:

```python
"inner_rhyme_summary"
"inner_rhyme_diagnosis"
```

### `prune_node(...)`

Al imprimir cada beam seleccionado, anadir:

```python
print(
    "Reparacion rima B: "
    f"{summarize_inner_rhyme_repair_report(beam.get('inner_rhyme_repair_report', {}))}"
)
```

## Cambios en guardado final

### `save_final_result(...)`

Leer del mejor beam:

```python
inner_rhyme_repair_report = best_beam.get("inner_rhyme_repair_report", {})
```

Validar que sea dict.

En `text_lines`, anadir despues de `Resumen reparacion rima A`:

```python
(
    "Resumen reparacion rima B: "
    f"{summarize_inner_rhyme_repair_report(inner_rhyme_repair_report)}"
),
```

En `json_payload`, anadir:

```python
"inner_rhyme_repair_report": inner_rhyme_repair_report,
```

### `main(...)`

En `run_parameters`, anadir:

```python
"enable_inner_rhyme_repair": ENABLE_INNER_RHYME_REPAIR,
"inner_rhyme_repair_variants": INNER_RHYME_REPAIR_VARIANTS,
"inner_rhyme_repair_temperature": INNER_RHYME_REPAIR_TEMPERATURE,
"inner_rhyme_repair_num_predict": INNER_RHYME_REPAIR_NUM_PREDICT,
```

En la salida final por terminal, anadir:

```python
print(
    "Reparacion rima B: "
    f"{summarize_inner_rhyme_repair_report(beam.get('inner_rhyme_repair_report', {}))}"
)
```

En el `.txt` final, anadir tambien las lineas de parametros:

```text
Reparacion rima B activa: True
Variantes para reparacion rima B: 5
Temperatura reparacion rima B: 0.3
Num predict reparacion rima B: 200
```

## Relacion con la rima exterior A

Este paso no debe cambiar la reparacion A.

La unica excepcion aceptable es un refactor interno para extraer logica comun que permita reutilizar codigo en B sin alterar el comportamiento observable de A.

Ejemplos aceptables:

```text
- convertir el parser de variantes en parser generico comun;
- parametrizar `build_rhyme_target_hint(...)` para que sirva para A y B;
- crear helpers comunes para listas de palabras candidatas.
```

Ejemplos no aceptables:

```text
- cambiar el orden de reparacion A;
- cambiar el criterio de aceptacion de A;
- cambiar las constantes de A;
- cambiar que A reescriba solo el verso 4.
```

La reparacion B debe ocurrir despues de A.

Como B solo reescribe el verso 3, no debe modificar:

```text
Verso 1
Verso 2
Verso 4
```

Por tanto, no deberia romper la rima exterior A, salvo que haya un error de implementacion.

La seleccion de variantes debe construir siempre:

```python
candidate_stanza = stanza.copy()
candidate_stanza[2] = variant
```

No usar ningun otro indice.

## Que NO hacer en este paso

No implementar:

- reparacion simultanea de versos 2 y 3;
- reescritura del verso 2;
- cambio de la rima exterior A;
- duplicacion literal de funciones ya existentes de A cuando pueda extraerse un helper comun;
- nueva reparacion metrica posterior a B;
- nuevos pesos de scoring;
- cambios en `evaluate_stanza_abba`;
- cambios en `sonnet_metrics.py`;
- cambios en Beam Search;
- cambios en `k`, `max_steps` o `alpha`;
- nuevas librerias;
- diccionarios externos;
- generacion de soneto completo.

Este paso debe limitarse a replicar la reparacion A para la pareja B, usando el verso 2 como ancla y reescribiendo solo el verso 3.

## Pruebas minimas recomendadas

### 1. Probar parser de variantes B

Ejecutar una prueba con:

```python
parse_rhyme_repair_variants_response(
    '{"variants": ["Verso nuevo perdido", "Verso nuevo dormido"]}',
    5,
)
```

Debe devolver una lista de variantes limpias.

### 2. Probar prompt sin Ollama

Usar una estrofa donde el verso 2 termine en una rima con ejemplos conocidos:

```python
stanza = [
    "El tiempo se queda en silencio callado",
    "La memoria descansa en sueno dormido",
    "La tristeza se abre en mi corazon",
    "Y vuelve por la tarde lo olvidado",
]
```

Diagnosticar:

```python
d = diagnose_stanza_inner_rhyme(stanza)
```

Construir prompt:

```python
msgs = _build_inner_rhyme_repair_messages("test", stanza, d, 5)
prompt = msgs[1]["content"]
```

El prompt debe contener:

```text
Verso 2 como ancla
Reescribe solo el verso 3
No modifiques el verso 1
No modifiques el verso 2
No modifiques el verso 4
Palabras finales candidatas para el nuevo verso 3
perdido
olvido
```

Y no debe forzar `dormido` si existen alternativas.

### 3. Probar reparacion B simulada con monkeypatch

Sustituir temporalmente:

```python
generate_inner_rhyme_repair_variants_with_ollama
```

por una funcion que devuelva:

```python
[
    "La tristeza camina sin dormido",
    "La tristeza regresa hacia el olvido",
]
```

Si el verso 2 termina en `dormido`, la variante terminada en `dormido` debe rechazarse si existen alternativas candidatas.

La variante terminada en `olvido` debe aceptarse si:

- corrige la rima interior B;
- no empeora la distancia metrica del verso 3.

El reporte debe registrar:

```python
"candidate_final_words"
"enforced_candidate_final_words"
"final_word"
"final_word_matches_anchor"
"uses_candidate_final_word"
"inner_rhyme_valid"
```

### 4. Probar que solo cambia el verso 3

Despues de una reparacion aceptada, comprobar:

```python
repaired_stanza[0] == original_stanza[0]
repaired_stanza[1] == original_stanza[1]
repaired_stanza[3] == original_stanza[3]
repaired_stanza[2] != original_stanza[2]
```

### 5. Probar integracion en salida

Ejecutar el flujo completo y comprobar que aparecen:

```text
Reparacion rima B activa: True
Resumen reparacion rima B: ...
```

Tanto en terminal como en `final_stanza_*.txt`.

En `final_stanza_metrics_*.json` debe aparecer:

```json
"inner_rhyme_repair_report": {...}
```

En `final_stanza_trace_*.json` debe aparecer dentro de las entradas:

```json
"inner_rhyme_repair_summary": "..."
"inner_rhyme_repair_report": {...}
```

### 6. Probar compilacion

Ejecutar:

```powershell
python -m compileall src
```

Debe compilar sin errores.

## Criterio de exito del Paso 7

El Paso 7 estara completo cuando:

1. Existan las constantes de reparacion B.
2. Exista un parser comun reutilizable para variantes de reparacion de rima.
3. `build_rhyme_target_hint(...)` pueda usarse tanto para A como para B sin duplicar codigo.
4. Exista `_build_inner_rhyme_repair_messages(...)`.
5. Exista `generate_inner_rhyme_repair_variants_with_ollama(...)`.
6. Exista `build_inner_rhyme_repair_report(...)`.
7. Exista `repair_stanza_inner_rhyme_with_ollama(...)`.
8. Exista `summarize_inner_rhyme_repair_report(...)`.
9. `score_node(...)` aplique la reparacion B despues de la reparacion A y antes de `evaluate_stanza_abba(...)`.
10. La reparacion B reescriba solo el verso 3.
11. El verso 2 se use como ancla y no se modifique.
12. Las variantes se acepten solo si corrigen B y no empeoran la distancia metrica del verso 3.
13. Si existen candidatas distintas del ancla, se rechacen variantes que repiten la palabra final del verso 2.
14. El reporte B aparezca en trazas, JSON de metricas, TXT final y terminal.
15. No se haya modificado `sonnet_metrics.py`.
16. No se haya modificado el comportamiento observable de la reparacion A, salvo refactor interno para reutilizar codigo.
17. No se haya cambiado scoring, Beam Search, `k`, `max_steps` ni `alpha`.
18. No se hayan duplicado bloques de logica que ya existian para A si podian convertirse en helpers comunes.

## Resultado esperado

Si antes se obtenia una estrofa como:

```text
1. El tiempo avanza como un rio en invierno helado
2. Dejan atras recuerdos sin verano calido
3. En la tristeza de mi corazon
4. Vuelve a florecer el amor olvidado
```

Con:

```text
endecasilabos = 4/4
rima exterior A = correcta
rima interior B = incorrecta
```

El nuevo reparador B deberia intentar reescribir solo el verso 3 para que rime con `calido`, manteniendo intactos los versos 1, 2 y 4.

El objetivo experimental es pasar de:

```text
rima = 3/4
```

a:

```text
rima = 4/4
```

cuando el LLM sea capaz de generar una variante valida del verso 3 sin empeorar la metrica.
