# Paso 10: reparacion metrica posterior a la reparacion de rima B

## Objetivo del paso

El objetivo de este paso es mejorar la reparacion de la rima interior B cuando el modelo consigue una palabra final que rima con el verso 2, pero el verso 3 resultante no queda metricamente bien.

La idea es:

```text
1. La reparacion B propone un nuevo verso 3.
2. Ese verso 3 corrige la rima con el verso 2.
3. Si ese verso 3 empeora la metrica o no tiene 11 silabas, no se rechaza inmediatamente.
4. Antes de rechazarlo, se intenta reparar solo la metrica de ese mismo verso 3.
5. La reparacion metrica posterior debe conservar exactamente la palabra final que habia conseguido la rima B.
6. Si tras esa reparacion el verso 3 sigue rimando con el verso 2 y no empeora la metrica, se acepta.
```

Este paso no pretende cambiar la forma de generar palabras candidatas para la rima B. Eso ya se hizo en el Paso 9. Este paso solo anade una segunda oportunidad metrica para variantes que ya han conseguido la rima B.

## Motivacion

Actualmente, la reparacion B acepta una variante solo si cumple simultaneamente:

```python
is_acceptable = (
    inner_rhyme_valid
    and metric_not_worse
    and uses_candidate_final_word
)
```

Esto es prudente, pero puede descartar variantes valiosas.

Ejemplo conceptual:

```text
Verso 2 termina en: pasado
Rima B objetivo: ado

Variante del verso 3:
En mi pecho vuelve el sueno olvidado
```

Puede ocurrir que:

```text
- La palabra final "olvidado" corrige la rima B.
- La palabra final es una candidata valida.
- Pero el verso tiene 12 silabas.
```

Con la logica actual, esa variante se rechaza directamente por metrica.

En este paso se quiere intentar:

```text
En mi pecho vuelve el sueno olvidado
        |
        v
Mi pecho guarda el sueno olvidado
```

manteniendo obligatoriamente:

```text
palabra final = olvidado
```

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

No debe tocar:

- `src/sonnet_metrics.py`;
- la extraccion de rima;
- el contador de silabas;
- `evaluate_stanza_abba(...)`;
- `aggregate_scores(...)`;
- el Beam Search;
- `k`;
- `max_steps`;
- `alpha`;
- la medicion de tiempo;
- la reparacion de rima exterior A;
- la reparacion metrica inicial general.

Este paso debe afectar solo a:

```text
Reparacion de rima interior B, concretamente al tratamiento de variantes del verso 3 que corrigen la rima pero fallan en metrica.
```

No se debe implementar todavia:

- reparacion metrica posterior a la rima exterior A;
- reescritura del verso 2;
- reescritura simultanea de versos 2 y 3;
- reescritura del verso 4;
- cambios de scoring;
- nuevos diccionarios externos;
- nuevas librerias;
- una estrategia distinta para buscar palabras candidatas.

## Punto de partida actual

Ya existen las constantes de reparacion metrica inicial:

```python
ENABLE_LOCAL_METER_REPAIR = True
METER_REPAIR_VARIANTS_PER_VERSE = 5
METER_REPAIR_TEMPERATURE = 0.4
METER_REPAIR_NUM_PREDICT = 200
```

Ya existen las constantes de reparacion B:

```python
ENABLE_INNER_RHYME_REPAIR = True
INNER_RHYME_REPAIR_VARIANTS = 5
INNER_RHYME_REPAIR_TEMPERATURE = 0.3
INNER_RHYME_REPAIR_NUM_PREDICT = 200
INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE = True
INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE = 2
INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS = 4
```

Tambien existen:

- `parse_meter_repair_variants_response(...)`;
- `_build_meter_repair_messages(...)`;
- `generate_meter_repair_variants_with_ollama(...)`;
- `repair_stanza_meter_with_ollama(...)`;
- `repair_stanza_inner_rhyme_with_ollama(...)`;
- `diagnose_stanza_inner_rhyme(...)`;
- `count_verse_syllables(...)`;
- `get_last_word(...)`;
- `syllable_distance_to_target(...)`.

El Paso 10 debe reutilizar estas piezas siempre que tenga sentido, pero no debe modificar el comportamiento de la reparacion metrica inicial.

## Nueva estrategia

### Comportamiento actual

Dentro de `repair_stanza_inner_rhyme_with_ollama(...)`, cada variante B se evalua asi:

```python
candidate_stanza = stanza.copy()
candidate_stanza[2] = variant

variant_diagnosis = diagnose_stanza_inner_rhyme(candidate_stanza)
variant_syllables = count_verse_syllables(variant)
variant_distance = syllable_distance_to_target(variant_syllables)
inner_rhyme_valid = bool(variant_diagnosis.get("is_valid", False))
metric_not_worse = variant_distance <= original_distance
variant_final_word = get_last_word(variant)
uses_candidate_final_word = (
    not enforced_candidate_final_words
    or variant_final_word in enforced_candidate_final_words
)
```

Y solo se acepta si:

```python
inner_rhyme_valid and metric_not_worse and uses_candidate_final_word
```

### Comportamiento nuevo

El filtro principal debe seguir existiendo.

Pero si una variante cumple:

```python
inner_rhyme_valid and uses_candidate_final_word
```

y falla solo por metrica:

```python
not metric_not_worse
```

entonces debe intentarse una reparacion metrica posterior sobre el verso 3.

Esa reparacion posterior debe:

- reescribir solo el verso 3;
- conservar exactamente la palabra final de la variante B;
- no tocar los versos 1, 2 ni 4;
- intentar conseguir 11 silabas metricas;
- comprobar despues que la rima B sigue siendo correcta;
- aceptar el resultado solo si la metrica no queda peor que la del verso 3 original.

## Nuevas constantes

Anadir junto a las constantes de reparacion B:

```python
ENABLE_POST_B_RHYME_METER_REPAIR = True
POST_B_RHYME_METER_REPAIR_VARIANTS = 5
POST_B_RHYME_METER_REPAIR_TEMPERATURE = 0.4
POST_B_RHYME_METER_REPAIR_NUM_PREDICT = 200
```

### Significado

`ENABLE_POST_B_RHYME_METER_REPAIR`

- activa o desactiva esta nueva reparacion metrica posterior a B;
- si esta en `False`, la reparacion B debe comportarse como antes.

`POST_B_RHYME_METER_REPAIR_VARIANTS`

- numero de variantes metricas que se piden para corregir el verso 3 tras una variante B rimada;
- para este paso debe ser `5`.

`POST_B_RHYME_METER_REPAIR_TEMPERATURE`

- temperatura usada en esta reparacion metrica posterior;
- para este paso debe ser `0.4`, igual que la reparacion metrica inicial.

`POST_B_RHYME_METER_REPAIR_NUM_PREDICT`

- limite de tokens de la llamada de reparacion metrica posterior;
- para este paso debe ser `200`.

## Nueva funcion: `_build_meter_repair_messages_preserving_final_word(...)`

Crear:

```python
def _build_meter_repair_messages_preserving_final_word(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    required_final_word: str,
    num_variants: int,
) -> List[Dict[str, str]]:
    ...
```

Esta funcion debe construir un prompt parecido a `_build_meter_repair_messages(...)`, pero con una restriccion mas fuerte:

```text
La palabra final del verso reparado debe conservarse exactamente.
```

### Restricciones obligatorias del prompt

El prompt debe indicar de forma clara:

```text
- Reescribe solo el verso indicado.
- No devuelvas la estrofa completa.
- No modifiques ningun otro verso.
- El objetivo es acercar el verso a 11 silabas metricas.
- La palabra final obligatoria es: required_final_word.
- Todas las variantes deben terminar exactamente en required_final_word.
- No uses una forma derivada, flexionada o parecida de required_final_word.
- No escribas ninguna palabra despues de required_final_word.
- No cambies la palabra final aunque eso limite la calidad poetica.
- Responde solo con JSON valido.
```

Para el caso del Paso 10, se llamara siempre con:

```python
verse_index = 2
```

porque solo se quiere reparar el verso 3.

### Formato JSON

El formato debe ser el mismo que en la reparacion metrica actual:

```json
{
  "variants": [
    "variante 1",
    "variante 2"
  ]
}
```

No se debe crear un parser nuevo.

Debe reutilizarse:

```python
parse_meter_repair_variants_response(...)
```

## Nueva funcion: `generate_meter_repair_variants_preserving_final_word_with_ollama(...)`

Crear:

```python
def generate_meter_repair_variants_preserving_final_word_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    required_final_word: str,
    num_variants: int = POST_B_RHYME_METER_REPAIR_VARIANTS,
) -> list[str]:
    ...
```

Debe:

1. Construir mensajes con `_build_meter_repair_messages_preserving_final_word(...)`.
2. Llamar a `chat_ollama(...)`.
3. Usar:

```python
temperature=POST_B_RHYME_METER_REPAIR_TEMPERATURE
num_predict=POST_B_RHYME_METER_REPAIR_NUM_PREDICT
```

4. Parsear con:

```python
parse_meter_repair_variants_response(...)
```

## Nueva funcion: `build_post_b_meter_repair_report(...)`

Crear:

```python
def build_post_b_meter_repair_report(
    enabled: bool,
    stanza: list[str],
    verse_index: int,
    required_final_word: str,
) -> dict[str, Any]:
    ...
```

Debe devolver una estructura base como:

```python
{
    "enabled": enabled,
    "attempted": False,
    "changed": False,
    "reason": None,
    "original_stanza": stanza.copy(),
    "repaired_stanza": stanza.copy(),
    "verse_number": verse_index + 1,
    "required_final_word": required_final_word,
    "original_verse": stanza[verse_index],
    "original_syllables": None,
    "original_distance": None,
    "selected_variant": None,
    "selected_syllables": None,
    "selected_distance": None,
    "variants": [],
    "error": None,
}
```

## Nueva funcion: `repair_verse_meter_preserving_final_word_with_ollama(...)`

Crear:

```python
def repair_verse_meter_preserving_final_word_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    required_final_word: str,
) -> tuple[list[str], dict[str, Any]]:
    ...
```

Aunque esta funcion sea generica, en el Paso 10 se usara solo para:

```python
verse_index = 2
```

### Comportamiento

La funcion debe:

1. Crear el informe con `build_post_b_meter_repair_report(...)`.
2. Si `ENABLE_POST_B_RHYME_METER_REPAIR` esta en `False`, devolver la estrofa sin cambios.
3. Si la estrofa no tiene ese indice de verso, devolver la estrofa sin cambios con motivo claro.
4. Calcular:

```python
original_syllables = count_verse_syllables(original_verse)
original_distance = syllable_distance_to_target(original_syllables)
```

5. Si `original_distance == 0`, no debe llamar a Ollama. Debe devolver:

```text
El verso ya tiene 11 silabas metricas.
```

6. Generar variantes metricas que conserven la palabra final obligatoria.
7. Evaluar cada variante.
8. Seleccionar la variante que mas reduzca la distancia al objetivo.

### Filtro de variantes

Cada variante debe registrarse en el informe con:

```python
{
    "verse": variant,
    "syllables": variant_syllables,
    "distance": variant_distance,
    "final_word": variant_final_word,
    "preserves_final_word": preserves_final_word,
    "improved": improved,
    "accepted": False,
    "rejection_reason": None,
}
```

Una variante solo puede considerarse candidata si:

```python
preserves_final_word and variant_distance < original_distance
```

Es decir:

- debe conservar exactamente la palabra final;
- debe mejorar estrictamente la metrica respecto al verso recibido por esta funcion.

Si ninguna variante mejora, se conserva el verso recibido.

## Integracion en `repair_stanza_inner_rhyme_with_ollama(...)`

La integracion debe hacerse dentro del bucle que evalua variantes B.

### Caso 1: variante aceptable como hasta ahora

Si la variante cumple:

```python
inner_rhyme_valid and metric_not_worse and uses_candidate_final_word
```

debe aceptarse como antes.

No hace falta llamar a la reparacion metrica posterior.

### Caso 2: variante corrige B pero falla por metrica

Si la variante cumple:

```python
inner_rhyme_valid and uses_candidate_final_word
```

pero:

```python
not metric_not_worse
```

y:

```python
ENABLE_POST_B_RHYME_METER_REPAIR
```

entonces debe intentarse:

```python
candidate_stanza_for_meter = stanza.copy()
candidate_stanza_for_meter[2] = variant

post_meter_stanza, post_meter_report = (
    repair_verse_meter_preserving_final_word_with_ollama(
        question=question,
        stanza=candidate_stanza_for_meter,
        verse_index=2,
        required_final_word=variant_final_word,
    )
)
```

Despues hay que comprobar el resultado:

```python
post_meter_verse = post_meter_stanza[2]
post_meter_syllables = count_verse_syllables(post_meter_verse)
post_meter_distance = syllable_distance_to_target(post_meter_syllables)
post_meter_final_word = get_last_word(post_meter_verse)
post_meter_diagnosis = diagnose_stanza_inner_rhyme(post_meter_stanza)
post_meter_inner_rhyme_valid = bool(post_meter_diagnosis.get("is_valid", False))
post_meter_preserves_final_word = post_meter_final_word == variant_final_word
post_meter_metric_not_worse = post_meter_distance <= original_distance
```

La variante reparada solo podra entrar en `acceptable_variants` si:

```python
post_meter_inner_rhyme_valid
and post_meter_preserves_final_word
and post_meter_metric_not_worse
and uses_candidate_final_word
```

Importante:

```text
La comparacion final de metrica debe hacerse contra el verso 3 original de la estrofa antes de la reparacion B, no solo contra la variante B intermedia.
```

Asi evitamos aceptar una reparacion que mejora un verso B muy malo, pero deja el resultado peor que el verso 3 original.

### Caso 3: variante no corrige B

Si la variante no corrige la rima B:

```python
not inner_rhyme_valid
```

no se debe intentar reparacion metrica posterior.

La reparacion metrica posterior no sirve para arreglar la rima; solo sirve para conservar una rima B ya conseguida.

### Caso 4: variante no termina en palabra candidata

Si:

```python
not uses_candidate_final_word
```

no se debe intentar reparacion metrica posterior.

Esa variante no respeta la estrategia de palabra candidata y debe rechazarse igual que ahora.

## Cambios en `variant_info`

Dentro de cada entrada de `report["variants"]`, anadir campos nuevos:

```python
"post_b_meter_repair_attempted": False,
"post_b_meter_repair_report": None,
"post_b_meter_repaired_verse": None,
"post_b_meter_repaired_syllables": None,
"post_b_meter_repaired_distance": None,
"post_b_meter_repair_accepted": False,
```

Cuando se intente la reparacion metrica posterior:

- `post_b_meter_repair_attempted` debe ser `True`;
- `post_b_meter_repair_report` debe contener el informe completo;
- si hay variante reparada, guardar texto, silabas y distancia.

Si la variante final aceptada viene de la reparacion metrica posterior:

```python
post_b_meter_repair_accepted = True
accepted = True
```

Y el elemento anadido a `acceptable_variants` debe usar el verso reparado, no la variante B intermedia.

Ejemplo:

```python
acceptable_variants.append(
    {
        "index": variant_index,
        "verse": post_meter_verse,
        "syllables": post_meter_syllables,
        "distance": post_meter_distance,
        "final_word_matches_anchor": final_word_matches_anchor,
        "variant_info": variant_info,
    }
)
```

## Cambios en `build_inner_rhyme_repair_report(...)`

Anadir campos generales:

```python
"post_b_meter_repair_enabled": ENABLE_POST_B_RHYME_METER_REPAIR,
"post_b_meter_repair_attempts": 0,
"post_b_meter_repair_successes": 0,
```

Significado:

`post_b_meter_repair_enabled`

- indica si la nueva fase estaba activada.

`post_b_meter_repair_attempts`

- numero de variantes B para las que se intento la reparacion metrica posterior.

`post_b_meter_repair_successes`

- numero de variantes B que acabaron siendo aceptables gracias a esa reparacion posterior.

## Cambios en seleccion de variante final

La seleccion final debe seguir usando:

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

No cambiar esta prioridad.

La unica diferencia es que `acceptable_variants` podra contener:

- variantes B aceptadas directamente;
- variantes B que fueron metricamente reparadas despues.

## Cambios en resumen de reparacion B

Actualizar `summarize_inner_rhyme_repair_report(...)` para que, cuando la reparacion B este activa, pueda incluir de forma compacta los intentos de reparacion metrica posterior.

Formato sugerido:

```text
reparacion rima B = activa | intento = True | cambio = True | post-metrica B = 2/1
```

Donde:

```text
2 = intentos de reparacion metrica posterior
1 = exitos de reparacion metrica posterior
```

Si no hubo intentos:

```text
reparacion rima B = activa | intento = True | cambio = False | post-metrica B = 0/0
```

No hace falta crear una linea nueva en terminal. Basta con ampliar el resumen ya existente.

## Cambios en trazas y JSON

No hace falta cambiar `build_trace_entry(...)` si ya guarda:

```python
"inner_rhyme_repair_report": inner_rhyme_repair_report
```

Los nuevos campos apareceran automaticamente en:

- `final_stanza_metrics_*.json`;
- `final_stanza_trace_*.json`.

## Cambios en `run_parameters`

En `main(...)`, anadir:

```python
"enable_post_b_rhyme_meter_repair": ENABLE_POST_B_RHYME_METER_REPAIR,
"post_b_rhyme_meter_repair_variants": POST_B_RHYME_METER_REPAIR_VARIANTS,
"post_b_rhyme_meter_repair_temperature": POST_B_RHYME_METER_REPAIR_TEMPERATURE,
"post_b_rhyme_meter_repair_num_predict": POST_B_RHYME_METER_REPAIR_NUM_PREDICT,
```

En `save_final_result(...)`, anadir al `.txt` final, junto a los parametros de reparacion B:

```text
Reparacion metrica posterior a rima B activa: True
Variantes para reparacion metrica posterior a rima B: 5
Temperatura reparacion metrica posterior a rima B: 0.4
Num predict reparacion metrica posterior a rima B: 200
```

Esto es importante para poder comparar tiempos de ejecucion con distintas configuraciones.

## Que NO hacer en este paso

No implementar:

- reparacion metrica posterior a la rima A;
- cambios sobre el verso 4;
- cambios sobre el verso 2;
- reparacion conjunta de versos 2 y 3;
- nuevos criterios de puntuacion;
- cambios en `evaluate_stanza_abba(...)`;
- cambios en `sonnet_metrics.py`;
- cambios en Beam Search;
- eliminacion de la reparacion metrica inicial;
- eliminacion de la reparacion B condicionada por palabra candidata;
- nuevas dependencias externas.

## Pruebas minimas recomendadas

### 1. Probar prompt con palabra final obligatoria

Construir:

```python
msgs = _build_meter_repair_messages_preserving_final_word(
    question="test",
    stanza=stanza,
    verse_index=2,
    original_syllables=12,
    required_final_word="olvidado",
    num_variants=5,
)
prompt = msgs[1]["content"]
```

El prompt debe contener:

```text
La palabra final obligatoria es: olvidado
Todas las variantes deben terminar exactamente en olvidado
No escribas ninguna palabra despues de olvidado
```

### 2. Probar que no cambia la palabra final

Con monkeypatch sobre:

```python
generate_meter_repair_variants_preserving_final_word_with_ollama(...)
```

devolver:

```python
[
    "Verso que cambia la palabra final perdida",
    "Verso correcto con final olvidado",
]
```

La funcion `repair_verse_meter_preserving_final_word_with_ollama(...)` debe rechazar la primera si la palabra final no es exactamente `olvidado`.

### 3. Probar que solo se toca el verso 3

Dada una estrofa de 4 versos, llamar a:

```python
repair_verse_meter_preserving_final_word_with_ollama(
    question=question,
    stanza=stanza,
    verse_index=2,
    required_final_word="olvidado",
)
```

Comprobar:

```python
repaired_stanza[0] == stanza[0]
repaired_stanza[1] == stanza[1]
repaired_stanza[3] == stanza[3]
```

Solo puede cambiar:

```python
repaired_stanza[2]
```

### 4. Probar integracion en B

Simular una variante B que:

```text
- corrige la rima interior B;
- termina en palabra candidata;
- empeora la metrica;
```

Y simular que la reparacion metrica posterior devuelve otra variante que:

```text
- conserva la misma palabra final;
- mantiene la rima B;
- no empeora la distancia metrica respecto al verso 3 original.
```

Debe aceptarse la variante reparada.

### 5. Probar que no se intenta post-metrica si no hay rima B

Si una variante no corrige la rima B, `post_b_meter_repair_attempted` debe seguir en `False`.

### 6. Probar que no se intenta post-metrica si no usa candidata

Si una variante no termina en una palabra final candidata valida, `post_b_meter_repair_attempted` debe seguir en `False`.

### 7. Probar que se conserva el comportamiento anterior si la constante esta desactivada

Con:

```python
ENABLE_POST_B_RHYME_METER_REPAIR = False
```

la reparacion B debe comportarse como antes del Paso 10.

### 8. Probar compilacion

Ejecutar:

```powershell
python -m compileall src
```

Debe compilar sin errores.

## Criterio de exito del Paso 10

El Paso 10 estara completo cuando:

1. Existan las constantes:
   - `ENABLE_POST_B_RHYME_METER_REPAIR`;
   - `POST_B_RHYME_METER_REPAIR_VARIANTS`;
   - `POST_B_RHYME_METER_REPAIR_TEMPERATURE`;
   - `POST_B_RHYME_METER_REPAIR_NUM_PREDICT`.
2. Exista `_build_meter_repair_messages_preserving_final_word(...)`.
3. Exista `generate_meter_repair_variants_preserving_final_word_with_ollama(...)`.
4. Exista `build_post_b_meter_repair_report(...)`.
5. Exista `repair_verse_meter_preserving_final_word_with_ollama(...)`.
6. `repair_stanza_inner_rhyme_with_ollama(...)` intente la reparacion metrica posterior solo cuando una variante:
   - corrige la rima B;
   - usa una palabra final candidata;
   - falla por metrica.
7. La reparacion metrica posterior conserve exactamente la palabra final del verso 3.
8. La reparacion metrica posterior no modifique versos 1, 2 ni 4.
9. Una variante reparada solo se acepte si:
   - mantiene la rima B;
   - conserva la palabra final;
   - no empeora la metrica frente al verso 3 original.
10. El reporte B incluya:
   - `post_b_meter_repair_enabled`;
   - `post_b_meter_repair_attempts`;
   - `post_b_meter_repair_successes`;
   - informacion detallada por variante.
11. Los parametros aparezcan en `final_stanza_*.txt`.
12. No se haya tocado `sonnet_metrics.py`.
13. No se haya cambiado la rima A.
14. No se haya cambiado el Beam Search ni el scoring.

## Resultado esperado

Con este paso, algunas variantes que antes se rechazaban por metrica podran rescatarse.

Antes:

```text
Variante B:
En mi pecho vuelve el sueno olvidado

Resultado:
Rechazada porque empeora la metrica.
```

Despues:

```text
Variante B:
En mi pecho vuelve el sueno olvidado

Reparacion metrica posterior:
Mi pecho guarda el sueno olvidado

Resultado:
Aceptada si mantiene rima B y no empeora metrica frente al verso 3 original.
```

Este paso no garantiza que la rima B se repare siempre, pero deberia aumentar el numero de casos en los que una buena palabra final rimada no se pierde solo porque el primer verso generado no encaja metricamente.
