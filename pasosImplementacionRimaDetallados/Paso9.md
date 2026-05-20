# Paso 9: reparacion B condicionada por palabra final candidata concreta

## Objetivo del paso

El objetivo de este paso es mejorar la reparacion de la rima interior B haciendo que el LLM genere variantes del verso 3 condicionadas por una palabra final candidata concreta.

Actualmente la reparacion B puede recibir una lista de palabras candidatas:

```text
llorar, pasar, mirar, soñar, callar, recordar
```

Pero en las trazas se ha observado que el modelo a veces no respeta la palabra exacta. Por ejemplo, para una rima objetivo `ar` y candidatas:

```text
llorar, pasar, mirar, soñar, callar, recordar
```

el modelo genero:

```text
En la tristeza de mi corazón llora
En la tristeza de mi corazón pasa
En la tristeza de mi corazón soña
```

Esas variantes son cercanas, pero no terminan exactamente en las palabras candidatas esperadas. Por tanto, son rechazadas por el filtro:

```text
No termina en una palabra final candidata distinta del verso 2.
```

La idea del Paso 9 es reducir aun mas la libertad del modelo:

```text
1. Elegir una palabra final candidata concreta.
2. Pedir variantes del verso 3 que terminen exactamente en esa palabra.
3. Repetir para varias palabras candidatas.
4. Evaluar todas las variantes generadas con el filtro B ya existente.
```

Ejemplo:

```text
Genera variantes del verso 3 terminando exactamente en "llorar".
Genera variantes del verso 3 terminando exactamente en "pasar".
Genera variantes del verso 3 terminando exactamente en "mirar".
```

Esto mantiene la metodologia actual:

```text
Verso 2 = ancla
Verso 3 = verso reparable
```

pero fuerza con mas claridad la palabra final del verso 3.

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
- el Beam Search;
- `k`;
- `max_steps`;
- `alpha`;
- la medicion de tiempo.

Este paso debe afectar solo a:

```text
Reparacion de rima interior B.
```

No se debe implementar todavia:

- reparacion metrica posterior a B;
- reescritura del verso 2;
- reparacion simultanea de versos 2 y 3;
- cambios de scoring;
- nuevos diccionarios externos;
- nuevas librerias.

## Punto de partida actual

Ya existen:

```python
ENABLE_INNER_RHYME_REPAIR = True
INNER_RHYME_REPAIR_VARIANTS = 5
INNER_RHYME_REPAIR_TEMPERATURE = 0.3
INNER_RHYME_REPAIR_NUM_PREDICT = 200
```

Tambien existen:

- `_build_inner_rhyme_repair_messages(...)`
- `generate_inner_rhyme_repair_variants_with_ollama(...)`
- `build_inner_rhyme_repair_report(...)`
- `repair_stanza_inner_rhyme_with_ollama(...)`
- `parse_rhyme_repair_variants_response(...)`
- `get_candidate_final_words_for_rhyme(...)`

La funcion principal de reparacion B ya calcula:

```python
candidate_final_words = get_candidate_final_words_for_rhyme(
    anchor_word=anchor_word,
    target_rhyme=target_rhyme,
)
enforced_candidate_final_words = [
    word for word in candidate_final_words if word != anchor_word
]
```

Y ya guarda:

```python
report["candidate_final_words"]
report["enforced_candidate_final_words"]
report["has_enforced_candidate_final_words"]
report["candidate_warning"]
```

El Paso 9 debe reutilizar todo esto.

## Nueva estrategia

### Comportamiento actual

Actualmente, si hay candidatas forzables, el prompt B dice:

```text
Cada variante debe terminar exactamente en una de esas palabras.
```

Esto a veces no es suficiente.

### Comportamiento nuevo

Si existen `enforced_candidate_final_words`, la reparacion B debe pedir variantes condicionadas por palabra final concreta.

Por ejemplo:

```python
enforced_candidate_final_words = [
    "llorar",
    "pasar",
    "mirar",
    "soñar",
    "callar",
    "recordar",
]
```

La generacion debe hacerse como:

```text
Genera variantes terminando exactamente en "llorar".
Genera variantes terminando exactamente en "pasar".
Genera variantes terminando exactamente en "mirar".
...
```

Despues se unen todas las variantes y se evalua con la logica actual.

Si `enforced_candidate_final_words` esta vacia, la reparacion B debe seguir usando el metodo actual:

```python
generate_inner_rhyme_repair_variants_with_ollama(...)
```

Es decir, este paso no debe bloquear casos sin candidatas.

## Nuevas constantes

Anadir junto a las constantes de reparacion B:

```python
INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE = True
INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE = 2
INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS = 4
```

### Significado

`INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE`

- activa o desactiva esta nueva estrategia;
- si esta en `False`, B debe comportarse como antes.

`INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE`

- numero de variantes que se piden para cada palabra final candidata concreta;
- para este paso debe ser `2`.

`INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS`

- maximo de palabras candidatas que se usan para no disparar el numero de llamadas a Ollama;
- para este paso debe ser `4`;
- se deben usar las primeras palabras de `enforced_candidate_final_words`, conservando el orden.

Ejemplo:

```python
enforced_candidate_final_words = [
    "llorar",
    "pasar",
    "mirar",
    "soñar",
    "callar",
    "recordar",
]
```

Con `INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS = 4`, se usan:

```python
["llorar", "pasar", "mirar", "soñar"]
```

## Nueva funcion: `_build_inner_rhyme_repair_messages_for_final_word(...)`

Crear:

```python
def _build_inner_rhyme_repair_messages_for_final_word(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    required_final_word: str,
    num_variants: int,
) -> List[Dict[str, str]]:
    ...
```

Debe construir un prompt parecido a `_build_inner_rhyme_repair_messages(...)`, pero con una palabra final obligatoria.

### Diferencia principal

El prompt debe decir de forma muy clara:

```text
Palabra final obligatoria para todas las variantes: llorar

Todas las variantes deben terminar exactamente en "llorar".
La ultima palabra debe ser exactamente "llorar".
No escribas nada despues de "llorar".
No uses "llora", "llorando", "lloré" ni otra forma derivada.
```

El texto debe adaptarse a la palabra recibida en `required_final_word`.

### Formato JSON

El formato debe seguir siendo:

```json
{
  "variants": [
    "nuevo verso 3",
    "nuevo verso 3 alternativo"
  ]
}
```

No debe cambiar el parser.

### Restricciones obligatorias

Debe incluir:

```text
- Reescribe solo el verso 3.
- No modifiques el verso 1.
- No modifiques el verso 2.
- No modifiques el verso 4.
- El nuevo verso 3 debe rimar consonantemente con el verso 2.
- La palabra final obligatoria es: required_final_word.
- Todas las variantes deben terminar exactamente en required_final_word.
- No uses una forma flexionada o derivada de required_final_word.
- No escribas nada despues de required_final_word.
- Intenta que el nuevo verso 3 tenga 11 silabas metricas.
- Devuelve solo variantes del verso 3, no la estrofa completa.
- Responde solo con JSON valido.
```

### Contexto incluido

Debe incluir, igual que el prompt B actual:

```text
Estrofa completa como contexto.
Verso 2 como ancla.
Palabra final del verso 2.
Rima consonante objetivo del verso 2.
Verso 3 actual.
Rima actual del verso 3.
```

No hace falta incluir la lista completa de candidatas, porque este prompt trabaja con una sola palabra final obligatoria.

## Nueva funcion: `generate_inner_rhyme_repair_variants_for_final_word_with_ollama(...)`

Crear:

```python
def generate_inner_rhyme_repair_variants_for_final_word_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    required_final_word: str,
    num_variants: int = INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE,
) -> list[str]:
    ...
```

Debe:

1. Construir mensajes con `_build_inner_rhyme_repair_messages_for_final_word(...)`.
2. Llamar a `chat_ollama(...)`.
3. Usar:

```python
temperature=INNER_RHYME_REPAIR_TEMPERATURE
num_predict=INNER_RHYME_REPAIR_NUM_PREDICT
```

4. Parsear con:

```python
parse_rhyme_repair_variants_response(...)
```

## Nueva funcion auxiliar: `generate_conditioned_inner_rhyme_repair_variants_with_ollama(...)`

Crear:

```python
def generate_conditioned_inner_rhyme_repair_variants_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    enforced_candidate_final_words: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    ...
```

Debe:

1. Tomar como maximo `INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS` candidatas.
2. Para cada candidata, llamar a:

```python
generate_inner_rhyme_repair_variants_for_final_word_with_ollama(...)
```

3. Unir las variantes.
4. Eliminar duplicados conservando orden.
5. Devolver:

```python
(
    variants,
    conditioned_generation_reports,
)
```

### Estructura de `conditioned_generation_reports`

Cada entrada debe tener:

```python
{
    "required_final_word": word,
    "attempted": True,
    "variants": [...],
    "error": None,
}
```

Si una llamada falla, no debe abortar toda la reparacion B. Debe registrar:

```python
{
    "required_final_word": word,
    "attempted": True,
    "variants": [],
    "error": str(exc),
}
```

y continuar con la siguiente palabra candidata.

Si todas las llamadas fallan o no devuelven variantes validas, la lista total de variantes quedara vacia.

En ese caso, `repair_stanza_inner_rhyme_with_ollama(...)` debera comportarse igual que ante un fallo de generacion:

```text
Fallo generando variantes de rima interior B.
```

o usar un motivo equivalente:

```text
No se generaron variantes condicionadas validas para la rima interior B.
```

## Cambios en `build_inner_rhyme_repair_report(...)`

Anadir campos nuevos:

```python
"conditioned_by_candidate": False,
"conditioned_candidate_words": [],
"conditioned_generation_reports": [],
```

Significado:

`conditioned_by_candidate`

- `True` si se uso la estrategia nueva condicionada por palabra candidata.

`conditioned_candidate_words`

- lista de palabras candidatas usadas en las llamadas condicionadas;
- debe respetar el limite `INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS`.

`conditioned_generation_reports`

- informes de cada llamada condicionada.

## Cambios en `repair_stanza_inner_rhyme_with_ollama(...)`

La funcion actual llama siempre a:

```python
variants = generate_inner_rhyme_repair_variants_with_ollama(...)
```

Debe cambiarse a:

```python
if (
    INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE
    and enforced_candidate_final_words
):
    variants, conditioned_generation_reports = (
        generate_conditioned_inner_rhyme_repair_variants_with_ollama(
            question=question,
            stanza=stanza,
            inner_rhyme_diagnosis=inner_rhyme_diagnosis,
            enforced_candidate_final_words=enforced_candidate_final_words,
        )
    )
    report["conditioned_by_candidate"] = True
    report["conditioned_candidate_words"] = enforced_candidate_final_words[
        :INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS
    ]
    report["conditioned_generation_reports"] = conditioned_generation_reports
else:
    variants = generate_inner_rhyme_repair_variants_with_ollama(...)
```

Si se usa la estrategia condicionada y no se obtiene ninguna variante, asignar:

```python
report["reason"] = "No se generaron variantes condicionadas validas para la rima interior B."
report["repaired_stanza"] = repaired_stanza.copy()
return repaired_stanza, report
```

### Importante

No cambiar el bucle de evaluacion de variantes.

El filtro actual debe seguir igual:

```python
is_acceptable = (
    inner_rhyme_valid
    and metric_not_worse
    and uses_candidate_final_word
)
```

Este paso solo cambia como se generan las variantes, no como se aceptan.

## Cambios en trazas y JSON

No hace falta cambiar `build_trace_entry(...)` si ya guarda el reporte completo:

```python
"inner_rhyme_repair_report": inner_rhyme_repair_report
```

Los nuevos campos apareceran automaticamente en:

- `final_stanza_metrics_*.json`;
- `final_stanza_trace_*.json`.

No hace falta cambiar el resumen de terminal.

## Cambios en `run_parameters`

En `main(...)`, anadir:

```python
"inner_rhyme_repair_conditioned_by_candidate": INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE,
"inner_rhyme_repair_variants_per_candidate": INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE,
"inner_rhyme_repair_max_candidate_words": INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS,
```

En `save_final_result(...)`, anadir al `.txt` final, junto a los parametros de reparacion B:

```text
Reparacion rima B condicionada por candidata: True
Variantes por candidata para reparacion rima B: 2
Max candidatas usadas para reparacion rima B: 4
```

Esto servira para documentar el coste temporal de esta estrategia.

## Que NO hacer en este paso

No implementar:

- reparacion metrica posterior a B;
- cambios en scoring;
- cambios en Beam Search;
- cambios en la reparacion A;
- cambios en `sonnet_metrics.py`;
- generacion de palabras candidatas mediante una llamada separada al LLM;
- reescritura del verso 2;
- aceptacion de variantes que no cumplan el filtro actual.

## Pruebas minimas recomendadas

### 1. Probar prompt condicionado por palabra final

Usar:

```python
msgs = _build_inner_rhyme_repair_messages_for_final_word(
    question="test",
    stanza=stanza,
    inner_rhyme_diagnosis=d,
    required_final_word="llorar",
    num_variants=2,
)
prompt = msgs[1]["content"]
```

El prompt debe contener:

```text
Palabra final obligatoria para todas las variantes: llorar
Todas las variantes deben terminar exactamente en "llorar"
No uses "llora"
No escribas nada despues de "llorar"
```

### 2. Probar generacion condicionada con monkeypatch

Sustituir temporalmente:

```python
generate_inner_rhyme_repair_variants_for_final_word_with_ollama
```

por una funcion que devuelva:

```python
[
    f"verso terminado en {required_final_word}"
]
```

Llamar a:

```python
generate_conditioned_inner_rhyme_repair_variants_with_ollama(...)
```

con:

```python
["llorar", "pasar", "mirar", "soñar", "callar"]
```

Debe usar solo las primeras 4 candidatas y devolver 4 variantes.

### 3. Probar reporte condicionado

Con una reparacion B simulada donde `enforced_candidate_final_words` no esta vacia, comprobar:

```python
report["conditioned_by_candidate"] is True
report["conditioned_candidate_words"] == primeras candidatas usadas
report["conditioned_generation_reports"] contiene una entrada por candidata usada
```

### 4. Probar fallback cuando no hay candidatas

Si `enforced_candidate_final_words` esta vacia, debe seguir usandose:

```python
generate_inner_rhyme_repair_variants_with_ollama(...)
```

Y:

```python
report["conditioned_by_candidate"] is False
```

### 5. Probar que el filtro de aceptacion no cambia

Con monkeypatch, generar una variante que termina en la candidata correcta pero no corrige rima o empeora metrica.

Debe seguir rechazandose.

### 6. Probar compilacion

Ejecutar:

```powershell
python -m compileall src
```

Debe compilar sin errores.

## Criterio de exito del Paso 9

El Paso 9 estara completo cuando:

1. Existan las constantes:
   - `INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE`;
   - `INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE`;
   - `INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS`.
2. Exista `_build_inner_rhyme_repair_messages_for_final_word(...)`.
3. Exista `generate_inner_rhyme_repair_variants_for_final_word_with_ollama(...)`.
4. Exista `generate_conditioned_inner_rhyme_repair_variants_with_ollama(...)`.
5. `repair_stanza_inner_rhyme_with_ollama(...)` use la estrategia condicionada cuando haya candidatas forzables.
6. Si no hay candidatas forzables, B siga usando el metodo anterior.
7. El reporte B incluya:
   - `conditioned_by_candidate`;
   - `conditioned_candidate_words`;
   - `conditioned_generation_reports`.
8. El filtro de aceptacion de B no haya cambiado.
9. No se haya implementado reparacion metrica posterior a B.
10. No se haya cambiado scoring, Beam Search, `k`, `max_steps` ni `alpha`.
11. No se haya tocado `sonnet_metrics.py`.

## Resultado esperado

En casos como:

```text
Rima objetivo B: ar
Candidatas: llorar, pasar, mirar, soñar
```

el LLM deberia producir con mas frecuencia versos que terminen exactamente en:

```text
llorar
pasar
mirar
soñar
```

en lugar de variantes incorrectas como:

```text
llora
pasa
soña
```

Esto no garantiza que la reparacion B siempre sea aceptada, porque la metrica puede seguir fallando. Pero deberia aumentar la proporcion de variantes que al menos cumplen la palabra final y la rima.
