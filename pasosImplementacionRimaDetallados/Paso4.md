# Paso 4: reparacion de rima exterior A con palabra final candidata fijada

## Objetivo del paso

El objetivo de este paso es cambiar la estrategia de reparacion de la rima exterior A.

Los pasos anteriores han mostrado que pedir al LLM:

```text
Reescribe el verso 4 para que rime con la rima objetivo "eva"
```

no es suficiente. Incluso tras mejorar el prompt, el modelo local sigue sin corregir de forma fiable la rima exterior A.

La nueva estrategia sera mas controlada:

```text
1. Obtener la rima objetivo del verso 1.
2. Obtener una lista de palabras finales candidatas que tengan esa rima.
3. Pedir al LLM que genere variantes del verso 4 terminando exactamente en una de esas palabras.
4. Evaluar las variantes con el filtro existente.
```

Ejemplo:

```text
Verso 1: El tiempo, un rastro que el viento lleva
Rima objetivo: eva
Palabras finales candidatas: lleva, nueva, eleva, conlleva, nieva
```

En vez de pedir al LLM que "rime con eva", se le pedira que genere versos 4 que terminen exactamente en una de esas palabras.

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

No debe tocar:

- `src/sonnet_metrics.py`;
- la extraccion de rima;
- la reparacion metrica;
- la reparacion de rima interior B;
- los pesos de scoring;
- el Beam Search;
- `evaluate_stanza_abba`;
- el criterio general de scoring.

Este paso sigue afectando solo a:

```text
Rima exterior A: verso 1 con verso 4
```

No se debe implementar todavia la reparacion de:

```text
Rima interior B: verso 2 con verso 3
```

## Problema que se quiere resolver

Actualmente el reparador de rima A genera variantes libres del verso 4. Aunque el prompt incluye la rima objetivo y ejemplos, el modelo no consigue producir de forma fiable versos que acaben en una palabra compatible.

El fallo observado es que:

- a veces copia literalmente la terminacion de rima;
- a veces produce una palabra final no compatible;
- a veces produce un verso con metrica aceptable pero sin rima;
- el filtro rechaza todas las variantes y `changed` queda en `False`.

La solucion propuesta es reducir la libertad del modelo:

```text
Elige una de estas palabras finales: nueva, eleva, nieva...
Genera variantes del verso 4 que terminen exactamente en una de ellas.
```

## Funciones existentes relevantes

En `src/langgraph_beam_stanza.py` ya existen:

- `RHYME_HINT_EXAMPLES`
  - Mapa de rimas objetivo a ejemplos de palabras compatibles.

- `build_rhyme_target_hint(...)`
  - Genera ayuda textual para el prompt.

- `_build_outer_rhyme_repair_messages(...)`
  - Construye el prompt actual de reparacion A.

- `generate_outer_rhyme_repair_variants_with_ollama(...)`
  - Llama a Ollama para generar variantes del verso 4.

- `repair_stanza_outer_rhyme_with_ollama(...)`
  - Funcion principal que decide si reparar A, pide variantes, las evalua y acepta una.

- `parse_outer_rhyme_repair_variants_response(...)`
  - Parsea variantes JSON.

- `diagnose_stanza_outer_rhyme(...)`
  - Evalua si la variante corrige AXYA.

- `count_verse_syllables(...)`
  - Evalua si la variante no empeora la metrica del verso 4.

## Nueva funcion para obtener palabras finales candidatas

Crear en `src/langgraph_beam_stanza.py`:

```python
def get_candidate_final_words_for_rhyme(
    anchor_word: str | None,
    target_rhyme: str | None,
) -> list[str]:
    ...
```

### Comportamiento

La funcion debe:

1. Normalizar `anchor_word` y `target_rhyme` con `str(...).strip()`.
2. Buscar `target_rhyme` en `RHYME_HINT_EXAMPLES`.
3. Construir una lista de palabras candidatas.
4. Incluir primero las palabras de `RHYME_HINT_EXAMPLES[target_rhyme]`, si existen.
5. Incluir tambien `anchor_word` si existe y no esta ya en la lista.
6. Eliminar vacios.
7. Eliminar duplicados conservando orden.
8. Devolver la lista.

### Ejemplos esperados

```python
get_candidate_final_words_for_rhyme("lleva", "eva")
```

debe devolver algo como:

```python
["lleva", "nueva", "eleva", "conlleva", "nieva"]
```

```python
get_candidate_final_words_for_rhyme("amargos", "argos")
```

debe devolver:

```python
["amargos", "largos"]
```

Si no hay ejemplos:

```python
get_candidate_final_words_for_rhyme("dolor", "or")
```

debe devolver:

```python
["dolor"]
```

Si no hay ni ejemplos ni palabra ancla:

```python
[]
```

## Cambios en `build_rhyme_target_hint`

`build_rhyme_target_hint(...)` puede reutilizar la nueva funcion:

```python
candidate_words = get_candidate_final_words_for_rhyme(
    anchor_word=anchor_word,
    target_rhyme=target_rhyme,
)
```

El texto de ayuda debe seguir explicando que:

- la rima objetivo es una terminacion;
- el verso 4 debe acabar en una palabra real;
- no hay que copiar literalmente la rima objetivo.

Pero ahora debe hablar de "palabras finales candidatas" de forma mas explicita.

Ejemplo:

```text
Palabras finales candidatas para el verso 4: lleva, nueva, eleva, conlleva, nieva.
El nuevo verso 4 debe terminar exactamente en una de esas palabras.
```

## Cambios en `_build_outer_rhyme_repair_messages`

Esta funcion debe calcular las palabras finales candidatas:

```python
candidate_final_words = get_candidate_final_words_for_rhyme(
    anchor_word=verse_1_last_word,
    target_rhyme=target_rhyme,
)
```

Y debe incluirlas en el prompt.

### Nueva informacion que debe aparecer en el prompt

El prompt debe incluir una seccion clara:

```text
Palabras finales candidatas para el nuevo verso 4:
- lleva
- nueva
- eleva
- conlleva
- nieva
```

Y despues:

```text
Cada variante debe terminar exactamente en una de esas palabras.
No uses otras palabras finales en este paso.
No escribas texto despues de la palabra final.
```

### Reglas nuevas obligatorias en el prompt

Anadir a las restricciones:

```text
- Cada variante debe terminar exactamente en una de las palabras finales candidatas.
- No uses una palabra final distinta de la lista.
- No escribas nada despues de la palabra final candidata.
- La ultima palabra de cada variante debe ser una palabra de la lista.
```

Mantener tambien las reglas previas:

```text
- Reescribe solo el verso 4.
- No modifiques el verso 1.
- No modifiques los versos 2 y 3.
- Intenta que el nuevo verso 4 tenga 11 silabas metricas.
- Devuelve solo variantes del verso 4, no la estrofa completa.
- Responde solo con JSON valido.
```

## Comportamiento si no hay palabras candidatas

Si `candidate_final_words` esta vacia:

- el prompt debe seguir funcionando con la ayuda textual anterior;
- no debe fallar la construccion del prompt;
- `repair_stanza_outer_rhyme_with_ollama(...)` puede seguir intentando la reparacion como hasta ahora.

Pero si hay palabras candidatas, deben usarse como restriccion fuerte.

## Cambios recomendados en el reporte

Actualizar `outer_rhyme_repair_report` para incluir:

```python
"candidate_final_words": [...],
```

En cada variante registrada, incluir:

```python
"final_word": "...",
"uses_candidate_final_word": True | False,
```

Esto permitira ver si el LLM obedecio la instruccion de terminar en una palabra candidata.

## Funcion auxiliar para extraer ultima palabra de variante

Para registrar la palabra final usada por cada variante, se puede importar:

```python
get_last_word
```

desde `sonnet_metrics.py`.

Pero antes de hacerlo hay que valorar si compensa tocar imports.

Opcion recomendada:

```python
from sonnet_metrics import (
    TARGET_SYLLABLES_PER_VERSE,
    count_verse_syllables,
    diagnose_stanza_inner_rhyme,
    diagnose_stanza_outer_rhyme,
    evaluate_stanza_abba,
    get_last_word,
)
```

No se debe reimplementar una limpieza manual de ultima palabra si ya existe `get_last_word(...)`.

## Cambios en `repair_stanza_outer_rhyme_with_ollama`

Dentro de `repair_stanza_outer_rhyme_with_ollama(...)`, despues de obtener `target_rhyme`, calcular:

```python
verse_1 = outer_rhyme_diagnosis.get("verse_1", {})
anchor_word = verse_1.get("last_word") if isinstance(verse_1, dict) else None
candidate_final_words = get_candidate_final_words_for_rhyme(
    anchor_word=anchor_word,
    target_rhyme=target_rhyme,
)
report["candidate_final_words"] = candidate_final_words
```

Al evaluar cada variante:

```python
variant_final_word = get_last_word(variant)
uses_candidate_final_word = (
    not candidate_final_words
    or variant_final_word in candidate_final_words
)
```

Y guardar en `variant_info`:

```python
"final_word": variant_final_word,
"uses_candidate_final_word": uses_candidate_final_word,
```

### Criterio de aceptacion

Actualizar el criterio de aceptacion para que, si hay palabras candidatas, tambien exija:

```python
uses_candidate_final_word is True
```

Es decir:

```python
is_acceptable = (
    outer_rhyme_valid
    and metric_not_worse
    and uses_candidate_final_word
)
```

Si no hay palabras candidatas, no debe bloquear por este criterio.

### Rejection reason

Si la variante no usa palabra candidata, registrar:

```text
No termina en una palabra final candidata.
```

El orden recomendado para motivos de rechazo:

1. no usa palabra final candidata;
2. no corrige la rima exterior AXYA;
3. empeora la distancia metrica del verso 4.

## Cambios en el reporte base

Actualizar `build_outer_rhyme_repair_report(...)` para incluir:

```python
"candidate_final_words": [],
```

## Cambios en trazas y exportacion

No hace falta cambiar la estructura general de trazas ni exportacion, porque `outer_rhyme_repair_report` ya se guarda completo.

Al anadir:

```python
"candidate_final_words"
"final_word"
"uses_candidate_final_word"
```

estos campos apareceran automaticamente en:

- traza;
- JSON final.

## Pruebas minimas recomendadas

### 1. Probar palabras candidatas

```powershell
$env:PYTHONPATH='.\src'
python -c "from langgraph_beam_stanza import get_candidate_final_words_for_rhyme; print(get_candidate_final_words_for_rhyme('lleva', 'eva'))"
```

Salida esperada:

```python
['lleva', 'nueva', 'eleva', 'conlleva', 'nieva']
```

### 2. Probar prompt sin Ollama

```powershell
$env:PYTHONPATH='.\src'
python -c "from sonnet_metrics import diagnose_stanza_outer_rhyme; from langgraph_beam_stanza import _build_outer_rhyme_repair_messages; stanza=['El tiempo, un rastro que el viento lleva','En la memoria, un recuerdo amargo dulce','Corazon, triste profundidad alli','Que nunca se olvidara jamas mas']; d=diagnose_stanza_outer_rhyme(stanza); msgs=_build_outer_rhyme_repair_messages('test', stanza, d, 5); print(msgs[1]['content'])"
```

El prompt debe contener:

```text
Palabras finales candidatas
lleva
nueva
eleva
Cada variante debe terminar exactamente en una de esas palabras
```

### 3. Probar evaluacion de variante simulada

Con monkeypatch o sustitucion temporal de:

```python
generate_outer_rhyme_repair_variants_with_ollama
```

probar que:

- una variante que termina en `nueva` se acepta si corrige rima y no empeora metrica;
- una variante que no termina en palabra candidata se rechaza aunque tenga una rima dudosa;
- se registra `final_word`;
- se registra `uses_candidate_final_word`.

### 4. Prueba real con Ollama

Ejecutar el flujo completo y revisar en la traza:

```json
"candidate_final_words": [...]
```

y en cada variante:

```json
"final_word": "...",
"uses_candidate_final_word": true
```

Se espera que el modelo produzca versos que terminen en palabras candidatas, no en fragmentos sueltos.

## Criterio de exito del Paso 4

El Paso 4 estara completo cuando:

1. Exista `get_candidate_final_words_for_rhyme(...)`.
2. El prompt de reparacion A muestre claramente las palabras finales candidatas.
3. El prompt obligue a terminar cada variante exactamente en una palabra candidata cuando existan candidatas.
4. El reporte guarde `candidate_final_words`.
5. Cada variante guarde su `final_word`.
6. Cada variante indique si usa una palabra candidata.
7. El filtro rechace variantes que no terminen en palabra candidata cuando hay lista disponible.
8. No se haya implementado reparacion de rima interior B.
9. No se haya tocado la reparacion metrica.
10. No se haya modificado `sonnet_metrics.py`, salvo para importar `get_last_word` desde ahi.

## Que NO hacer en este paso

No implementar todavia:

- reparacion de rima interior B;
- cambios en scoring;
- cambios en Beam Search;
- nuevos pesos;
- diccionarios externos;
- librerias nuevas;
- reparacion metrica posterior a rima;
- generacion de soneto completo;
- evaluacion subjetiva.

Este paso debe limitarse a hacer mas controlada la reparacion de la rima exterior A imponiendo palabras finales candidatas para el verso 4.
