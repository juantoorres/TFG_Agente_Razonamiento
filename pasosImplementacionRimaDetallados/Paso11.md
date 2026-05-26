# Paso 11: reparacion metrica posterior tambien para la rima exterior A

## Objetivo del paso

El objetivo de este paso es generalizar la reparacion metrica posterior a la reparacion de rima para que no se aplique solo a la rima interior B, sino tambien a la rima exterior A.

Actualmente, tras el Paso 10, existe una segunda oportunidad metrica para el verso 3:

```text
Rima B:
verso 2 = ancla
verso 3 = verso reparable
```

Si una variante del verso 3 consigue rimar con el verso 2 pero falla por metrica, se intenta reparar solo la metrica del verso 3 conservando exactamente la palabra final que habia conseguido la rima.

En este Paso 11 se quiere aplicar la misma idea a la rima A:

```text
Rima A:
verso 1 = ancla
verso 4 = verso reparable
```

Es decir:

```text
1. La reparacion A propone un nuevo verso 4.
2. Ese verso 4 corrige la rima con el verso 1.
3. Si ese verso 4 falla por metrica, no se rechaza inmediatamente.
4. Antes de rechazarlo, se intenta reparar solo la metrica del verso 4.
5. La reparacion metrica posterior debe conservar exactamente la palabra final que habia conseguido la rima A.
6. Si tras esa reparacion el verso 4 sigue rimando con el verso 1 y no empeora la metrica, se acepta.
```

## Motivacion

La reparacion B posterior a la rima ha mostrado que puede rescatar variantes que ya tienen una buena palabra final rimada, pero que inicialmente fallan por metrica.

La rima A tiene el mismo problema potencial. Una variante del verso 4 puede terminar en una palabra correcta para rimar con el verso 1, pero tener 10, 12 o mas silabas.

Ejemplo conceptual:

```text
Verso 1 termina en: helado
Rima A objetivo: ado

Variante del verso 4:
Y vuelve hacia mi pecho ya olvidado
```

Puede ocurrir que:

```text
- La palabra final "olvidado" corrige la rima A.
- La palabra final es una candidata valida.
- Pero el verso 4 no queda en 11 silabas.
```

En lugar de rechazarla directamente, este paso debe intentar:

```text
Y vuelve hacia mi pecho ya olvidado
        |
        v
Vuelve a mi pecho el amor olvidado
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
- la generacion inicial de estrofas;
- la reparacion metrica inicial general.

Este paso debe afectar a:

```text
Reparacion de rima exterior A, concretamente al tratamiento de variantes del verso 4 que corrigen la rima pero fallan en metrica.
```

Tambien puede afectar de forma limitada a:

```text
Funciones auxiliares comunes de reparacion metrica posterior conservando palabra final.
```

pero solo si es necesario para reutilizarlas tanto en A como en B sin duplicar logica.

## Punto de partida actual

### Reparacion A actual

La reparacion exterior A se hace en:

```python
repair_stanza_outer_rhyme_with_ollama(...)
```

Actualmente:

```text
verso 1 = ancla
verso 4 = verso reparable
```

El filtro de aceptacion de variantes A es:

```python
is_acceptable = (
    outer_rhyme_valid
    and metric_not_worse
    and uses_candidate_final_word
)
```

Si una variante corrige la rima A pero empeora la metrica del verso 4, se rechaza con:

```text
Empeora la distancia metrica del verso 4.
```

### Reparacion B posterior ya existente

Tras el Paso 10 ya existen funciones para reparar la metrica conservando palabra final:

```python
_build_meter_repair_messages_preserving_final_word(...)
generate_meter_repair_variants_preserving_final_word_with_ollama(...)
repair_verse_meter_preserving_final_word_with_ollama(...)
```

Tambien existen campos especificos de B:

```python
post_b_meter_repair_enabled
post_b_meter_repair_attempts
post_b_meter_repair_successes
```

Y campos por variante:

```python
post_b_meter_repair_attempted
post_b_meter_repair_report
post_b_meter_repaired_verse
post_b_meter_repaired_syllables
post_b_meter_repaired_distance
post_b_meter_repair_accepted
```

## Decision de diseno

El Paso 11 debe reutilizar la metodologia del Paso 10.

No se debe crear una estrategia distinta para A.

La nueva reparacion posterior A debe cumplir exactamente el mismo principio:

```text
La rima se consigue primero.
La metrica se repara despues.
La palabra final que consiguio la rima se bloquea.
```

## Reutilizacion de funciones

No se deben duplicar funciones grandes.

La funcion:

```python
repair_verse_meter_preserving_final_word_with_ollama(...)
```

ya es conceptualmente generica porque recibe:

```python
verse_index
required_final_word
```

Por tanto, debe reutilizarse tambien para el verso 4:

```python
repair_verse_meter_preserving_final_word_with_ollama(
    question=question,
    stanza=candidate_stanza_for_meter,
    verse_index=3,
    required_final_word=variant_final_word,
)
```

### Ajuste recomendado para evitar nombres demasiado especificos de B

Actualmente algunos nombres internos dicen `post_b`.

Para este paso hay dos opciones aceptables:

1. Mantener la funcion generica existente y usarla tambien para A, aunque internamente el informe base se llame `build_post_b_meter_repair_report(...)`.
2. Refactorizar de forma minima los nombres comunes para que dejen de ser especificos de B.

La opcion preferida es la 2, siempre que no cambie el comportamiento de B.

Refactor sugerido:

```python
build_post_rhyme_meter_repair_report(...)
```

en lugar de:

```python
build_post_b_meter_repair_report(...)
```

La funcion:

```python
repair_verse_meter_preserving_final_word_with_ollama(...)
```

puede conservar su nombre porque ya es generica.

Si se refactoriza el nombre del builder, deben actualizarse sus llamadas. No hace falta mantener alias antiguo salvo que se quiera evitar tocar mas codigo.

## Nuevas constantes para A

Anadir junto a las constantes de reparacion A:

```python
ENABLE_POST_A_RHYME_METER_REPAIR = True
POST_A_RHYME_METER_REPAIR_VARIANTS = 5
POST_A_RHYME_METER_REPAIR_TEMPERATURE = 0.4
POST_A_RHYME_METER_REPAIR_NUM_PREDICT = 200
```

### Significado

`ENABLE_POST_A_RHYME_METER_REPAIR`

- activa o desactiva la nueva reparacion metrica posterior a A;
- si esta en `False`, la reparacion A debe comportarse como antes.

`POST_A_RHYME_METER_REPAIR_VARIANTS`

- numero de variantes metricas que se piden para corregir el verso 4 tras una variante A rimada;
- para este paso debe ser `5`.

`POST_A_RHYME_METER_REPAIR_TEMPERATURE`

- temperatura usada en esta reparacion metrica posterior;
- para este paso debe ser `0.4`, igual que la reparacion metrica inicial y la post-metrica B.

`POST_A_RHYME_METER_REPAIR_NUM_PREDICT`

- limite de tokens de la llamada de reparacion metrica posterior;
- para este paso debe ser `200`.

## Ajuste recomendado en funciones genericas

Actualmente:

```python
generate_meter_repair_variants_preserving_final_word_with_ollama(...)
```

usa constantes de B:

```python
POST_B_RHYME_METER_REPAIR_TEMPERATURE
POST_B_RHYME_METER_REPAIR_NUM_PREDICT
```

Para poder reutilizarla en A sin duplicar codigo, modificarla para aceptar parametros opcionales:

```python
def generate_meter_repair_variants_preserving_final_word_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    required_final_word: str,
    num_variants: int,
    temperature: float,
    num_predict: int,
) -> list[str]:
    ...
```

O, si se prefiere mantener defaults, usar:

```python
def generate_meter_repair_variants_preserving_final_word_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    required_final_word: str,
    num_variants: int = POST_B_RHYME_METER_REPAIR_VARIANTS,
    temperature: float = POST_B_RHYME_METER_REPAIR_TEMPERATURE,
    num_predict: int = POST_B_RHYME_METER_REPAIR_NUM_PREDICT,
) -> list[str]:
    ...
```

La llamada desde B debe seguir usando los valores B.

La llamada desde A debe usar:

```python
num_variants=POST_A_RHYME_METER_REPAIR_VARIANTS
temperature=POST_A_RHYME_METER_REPAIR_TEMPERATURE
num_predict=POST_A_RHYME_METER_REPAIR_NUM_PREDICT
```

Tambien conviene ajustar:

```python
repair_verse_meter_preserving_final_word_with_ollama(...)
```

para recibir:

```python
enabled: bool
num_variants: int
temperature: float
num_predict: int
repair_label: str
```

Ejemplo:

```python
def repair_verse_meter_preserving_final_word_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    required_final_word: str,
    enabled: bool,
    num_variants: int,
    temperature: float,
    num_predict: int,
    repair_label: str,
) -> tuple[list[str], dict[str, Any]]:
    ...
```

Valores de `repair_label`:

```python
"post_a"
"post_b"
```

Este campo sirve solo para que el informe tenga motivos claros. No debe afectar a la logica.

## Cambios en `build_outer_rhyme_repair_report(...)`

Anadir campos generales:

```python
"post_a_meter_repair_enabled": ENABLE_POST_A_RHYME_METER_REPAIR,
"post_a_meter_repair_attempts": 0,
"post_a_meter_repair_successes": 0,
```

Significado:

`post_a_meter_repair_enabled`

- indica si la nueva fase estaba activada.

`post_a_meter_repair_attempts`

- numero de variantes A para las que se intento la reparacion metrica posterior.

`post_a_meter_repair_successes`

- numero de variantes A que acabaron siendo aceptables gracias a esa reparacion posterior.

## Cambios en `repair_stanza_outer_rhyme_with_ollama(...)`

La integracion debe hacerse dentro del bucle que evalua variantes A.

### Caso 1: variante aceptable como hasta ahora

Si la variante cumple:

```python
outer_rhyme_valid and metric_not_worse and uses_candidate_final_word
```

debe aceptarse como antes.

No hace falta llamar a la reparacion metrica posterior.

### Caso 2: variante corrige A pero falla por metrica

Si la variante cumple:

```python
outer_rhyme_valid and uses_candidate_final_word
```

pero:

```python
not metric_not_worse
```

y:

```python
ENABLE_POST_A_RHYME_METER_REPAIR
```

entonces debe intentarse:

```python
report["post_a_meter_repair_attempts"] += 1
variant_info["post_a_meter_repair_attempted"] = True

candidate_stanza_for_meter = stanza.copy()
candidate_stanza_for_meter[3] = variant

post_meter_stanza, post_meter_report = (
    repair_verse_meter_preserving_final_word_with_ollama(
        question=question,
        stanza=candidate_stanza_for_meter,
        verse_index=3,
        required_final_word=variant_final_word,
        enabled=ENABLE_POST_A_RHYME_METER_REPAIR,
        num_variants=POST_A_RHYME_METER_REPAIR_VARIANTS,
        temperature=POST_A_RHYME_METER_REPAIR_TEMPERATURE,
        num_predict=POST_A_RHYME_METER_REPAIR_NUM_PREDICT,
        repair_label="post_a",
    )
)
```

Despues hay que comprobar el resultado:

```python
post_meter_verse = post_meter_stanza[3]
post_meter_syllables = count_verse_syllables(post_meter_verse)
post_meter_distance = syllable_distance_to_target(post_meter_syllables)
post_meter_final_word = get_last_word(post_meter_verse)
post_meter_diagnosis = diagnose_stanza_outer_rhyme(post_meter_stanza)
post_meter_outer_rhyme_valid = bool(post_meter_diagnosis.get("is_valid", False))
post_meter_preserves_final_word = post_meter_final_word == variant_final_word
post_meter_metric_not_worse = post_meter_distance <= original_distance
```

La variante reparada solo podra entrar en `acceptable_variants` si:

```python
post_meter_outer_rhyme_valid
and post_meter_preserves_final_word
and post_meter_metric_not_worse
and uses_candidate_final_word
```

Importante:

```text
La comparacion final de metrica debe hacerse contra el verso 4 original de la estrofa antes de la reparacion A, no solo contra la variante A intermedia.
```

Asi se evita aceptar una reparacion que mejora una variante A muy mala, pero deja el resultado peor que el verso 4 original.

### Caso 3: variante no corrige A

Si la variante no corrige la rima A:

```python
not outer_rhyme_valid
```

no se debe intentar reparacion metrica posterior.

La post-metrica sirve para conservar una rima ya lograda, no para arreglar una rima que sigue mal.

### Caso 4: variante no termina en palabra candidata

Si:

```python
not uses_candidate_final_word
```

no se debe intentar reparacion metrica posterior.

Esa variante debe rechazarse igual que ahora.

## Cambios en `variant_info` de A

Dentro de cada entrada de `report["variants"]` de A, anadir:

```python
"post_a_meter_repair_attempted": False,
"post_a_meter_repair_report": None,
"post_a_meter_repaired_verse": None,
"post_a_meter_repaired_syllables": None,
"post_a_meter_repaired_distance": None,
"post_a_meter_repair_accepted": False,
```

Cuando se intente la reparacion metrica posterior:

- `post_a_meter_repair_attempted` debe ser `True`;
- `post_a_meter_repair_report` debe contener el informe completo;
- si hay variante reparada, guardar texto, silabas y distancia.

Si la variante final aceptada viene de la reparacion metrica posterior:

```python
post_a_meter_repair_accepted = True
accepted = True
```

Y el elemento anadido a `acceptable_variants` debe usar el verso reparado, no la variante A intermedia.

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

## Cambios en B para mantener compatibilidad

Si se modifica la firma de:

```python
repair_verse_meter_preserving_final_word_with_ollama(...)
```

hay que actualizar la llamada existente desde B para pasar explicitamente:

```python
enabled=ENABLE_POST_B_RHYME_METER_REPAIR
num_variants=POST_B_RHYME_METER_REPAIR_VARIANTS
temperature=POST_B_RHYME_METER_REPAIR_TEMPERATURE
num_predict=POST_B_RHYME_METER_REPAIR_NUM_PREDICT
repair_label="post_b"
```

La logica de B no debe cambiar.

El comportamiento esperado para B tras este paso debe ser el mismo que en el Paso 10.

## Cambios en seleccion de variante final A

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

- variantes A aceptadas directamente;
- variantes A que fueron metricamente reparadas despues.

## Cambios en resumen de reparacion A

Actualizar:

```python
summarize_outer_rhyme_repair_report(...)
```

para que incluya los intentos de reparacion metrica posterior A.

Formato sugerido:

```text
reparacion rima A = activa | intento = True | cambio = True | post-metrica A = 2/1
```

Donde:

```text
2 = intentos de reparacion metrica posterior A
1 = exitos de reparacion metrica posterior A
```

Si no hubo intentos:

```text
reparacion rima A = activa | intento = True | cambio = False | post-metrica A = 0/0
```

## Cambios en trazas y JSON

No hace falta cambiar `build_trace_entry(...)` si ya guarda:

```python
"outer_rhyme_repair_report": outer_rhyme_repair_report
```

Los nuevos campos apareceran automaticamente en:

- `final_stanza_metrics_*.json`;
- `final_stanza_trace_*.json`.

## Cambios en `run_parameters`

En `main(...)`, anadir:

```python
"enable_post_a_rhyme_meter_repair": ENABLE_POST_A_RHYME_METER_REPAIR,
"post_a_rhyme_meter_repair_variants": POST_A_RHYME_METER_REPAIR_VARIANTS,
"post_a_rhyme_meter_repair_temperature": POST_A_RHYME_METER_REPAIR_TEMPERATURE,
"post_a_rhyme_meter_repair_num_predict": POST_A_RHYME_METER_REPAIR_NUM_PREDICT,
```

En `save_final_result(...)`, anadir al `.txt` final, junto a los parametros de reparacion A:

```text
Reparacion metrica posterior a rima A activa: True
Variantes para reparacion metrica posterior a rima A: 5
Temperatura reparacion metrica posterior a rima A: 0.4
Num predict reparacion metrica posterior a rima A: 200
```

Mantener tambien los parametros ya existentes de post-metrica B.

## Que NO hacer en este paso

No implementar:

- cambios en `sonnet_metrics.py`;
- cambios en el contador de silabas;
- cambios en la extraccion de rima;
- cambios en el Beam Search;
- cambios en scoring;
- cambios en `k`, `max_steps` o `alpha`;
- reparacion conjunta de versos 1 y 4;
- reescritura del verso 1;
- reescritura del verso 2;
- reescritura simultanea de versos 2 y 3;
- nuevas dependencias externas;
- nuevos diccionarios externos;
- generacion de palabras candidatas con otra llamada al LLM.

Tampoco se debe relajar la aceptacion final.

Una variante post-reparada debe seguir cumpliendo:

```text
rima correcta + palabra final conservada + metrica no peor que la original
```

## Pruebas minimas recomendadas

### 1. Probar que A intenta post-metrica solo cuando procede

Simular una variante A que:

```text
- corrige la rima exterior A;
- termina en palabra candidata;
- empeora la metrica.
```

Debe activar:

```python
post_a_meter_repair_attempted = True
post_a_meter_repair_attempts == 1
```

### 2. Probar que A no intenta post-metrica si no hay rima

Si una variante del verso 4 no corrige la rima A:

```python
outer_rhyme_valid = False
```

entonces:

```python
post_a_meter_repair_attempted = False
```

### 3. Probar que A no intenta post-metrica si no usa candidata

Si una variante no termina en palabra final candidata valida:

```python
uses_candidate_final_word = False
```

entonces:

```python
post_a_meter_repair_attempted = False
```

### 4. Probar que solo se toca el verso 4

Con una reparacion A post-metrica simulada:

```python
repaired_stanza[0] == stanza[0]
repaired_stanza[1] == stanza[1]
repaired_stanza[2] == stanza[2]
```

Solo puede cambiar:

```python
repaired_stanza[3]
```

### 5. Probar que conserva palabra final

Si la variante A acaba en:

```text
olvidado
```

la variante post-metrica aceptada debe acabar tambien exactamente en:

```text
olvidado
```

No debe aceptarse:

```text
olvidada
olvido
olvidando
```

### 6. Probar que mantiene rima A tras la post-metrica

Tras reparar metricamente el verso 4, volver a ejecutar:

```python
diagnose_stanza_outer_rhyme(post_meter_stanza)
```

Debe cumplirse:

```python
post_meter_outer_rhyme_valid is True
```

### 7. Probar que B no se rompe

Ejecutar una prueba equivalente a la del Paso 10 para comprobar que la post-metrica B sigue funcionando igual:

```text
verso 3 corrige B + falla metrica -> post-metrica B -> acepta si queda valida
```

### 8. Probar compilacion

Ejecutar:

```powershell
python -m compileall src
```

Debe compilar sin errores.

## Criterio de exito del Paso 11

El Paso 11 estara completo cuando:

1. Existan las constantes:
   - `ENABLE_POST_A_RHYME_METER_REPAIR`;
   - `POST_A_RHYME_METER_REPAIR_VARIANTS`;
   - `POST_A_RHYME_METER_REPAIR_TEMPERATURE`;
   - `POST_A_RHYME_METER_REPAIR_NUM_PREDICT`.
2. La reparacion A intente post-metrica solo cuando una variante:
   - corrige la rima A;
   - usa una palabra final candidata;
   - falla por metrica.
3. La post-metrica A repare solo el verso 4.
4. La post-metrica A conserve exactamente la palabra final del verso 4.
5. Una variante post-metrica A solo se acepte si:
   - mantiene la rima A;
   - conserva la palabra final;
   - no empeora la metrica frente al verso 4 original.
6. El reporte A incluya:
   - `post_a_meter_repair_enabled`;
   - `post_a_meter_repair_attempts`;
   - `post_a_meter_repair_successes`;
   - informacion detallada por variante.
7. El resumen de reparacion A muestre `post-metrica A = intentos/exitos`.
8. Los parametros nuevos aparezcan en `final_stanza_*.txt`.
9. La post-metrica B siga funcionando igual que antes.
10. No se haya tocado `sonnet_metrics.py`.
11. No se haya cambiado Beam Search ni scoring.

## Resultado esperado

Antes:

```text
Variante A:
Vuelve a mi pecho todo lo olvidado

Resultado:
Rechazada porque empeora la metrica.
```

Despues:

```text
Variante A:
Vuelve a mi pecho todo lo olvidado

Reparacion metrica posterior:
Vuelve a mi pecho el amor olvidado

Resultado:
Aceptada si mantiene rima A y no empeora metrica frente al verso 4 original.
```

Con este paso, tanto la rima A como la rima B podran beneficiarse del mismo enfoque:

```text
primero asegurar la palabra final rimada;
despues ajustar la metrica sin cambiar esa palabra final.
```
