# Paso 3c: mejorar el prompt de reparacion A para no copiar literalmente la rima objetivo

## Objetivo del paso

El objetivo de este paso es corregir un problema detectado tras el Paso 3b.

La extraccion de rima ya devuelve una rima mas razonable. Por ejemplo:

```text
lleva -> eva
amargos -> argos
```

Sin embargo, el reparador de rima exterior A esta usando esa rima objetivo de forma demasiado literal. En vez de terminar el verso 4 con una palabra real que tenga esa rima, el modelo esta escribiendo directamente la cadena de rima como si fuese una palabra.

Ejemplo observado:

```text
Verso 1: El tiempo, un rastro que el viento lleva
Rima objetivo: eva
```

Variantes generadas:

```text
Que nunca mas se olvidara eva
Que jamas mas el olvido no se eva
Que el olvido jamas mas se deja eva
```

Esto es incorrecto. `eva` es una terminacion de rima, no una palabra final necesariamente valida.

Este paso debe mejorar el prompt de reparacion A para que el LLM entienda que debe terminar el verso 4 con una palabra real que comparta esa rima consonante.

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

Funcion principal a modificar:

```python
_build_outer_rhyme_repair_messages(...)
```

Opcionalmente, se puede crear una funcion auxiliar pequena dentro del mismo archivo para construir una ayuda textual sobre la rima objetivo.

Este paso NO debe tocar:

- `src/sonnet_metrics.py`;
- `diagnose_stanza_outer_rhyme(...)`;
- `diagnose_stanza_inner_rhyme(...)`;
- `extract_consonant_rhyme(...)`;
- la reparacion metrica;
- el criterio de aceptacion de variantes;
- la reparacion de rima interior B;
- los pesos de scoring;
- el Beam Search.

## Problema observado en la traza

Fragmento relevante:

```json
{
  "target_rhyme": "eva",
  "current_rhyme": "as",
  "verse_1": {
    "text": "El tiempo, un rastro que el viento lleva",
    "last_word": "lleva",
    "rhyme": "eva"
  },
  "variants": [
    {
      "verse": "Que nunca mas se olvidara eva",
      "outer_rhyme_valid": false
    },
    {
      "verse": "Que jamas mas el olvido no se eva",
      "outer_rhyme_valid": false
    }
  ]
}
```

El sistema ahora extrae bien la rima objetivo:

```text
lleva -> eva
```

Pero el prompt no explica suficientemente que el modelo debe buscar una palabra real compatible, por ejemplo:

```text
nueva
eleva
conlleva
nieva
```

No debe escribir literalmente:

```text
eva
```

## Cambio principal a realizar

Modificar `_build_outer_rhyme_repair_messages(...)` para que el prompt incluya una explicacion mas clara de la diferencia entre:

- palabra final del verso 1;
- rima consonante objetivo;
- palabra final real que debe usar el verso 4.

Actualmente el prompt dice:

```text
Rima consonante objetivo del verso 1: eva
...
- La rima objetivo es: eva.
```

Esto debe cambiarse o ampliarse para que diga de forma clara:

```text
La rima objetivo "eva" es una terminacion de rima, no una palabra que debas copiar literalmente.
El nuevo verso 4 debe terminar en una palabra real cuya rima consonante sea "eva".
No termines el verso con la cadena aislada "eva" salvo que sea una palabra real y natural en el verso.
```

## Reglas nuevas que deben aparecer en el prompt

Dentro de las restricciones del prompt de reparacion A deben aparecer reglas equivalentes a estas:

```text
- La rima objetivo es una terminacion de rima, no una palabra obligatoria.
- No copies literalmente la rima objetivo como palabra final si no es una palabra natural.
- Termina el verso 4 con una palabra real que tenga esa rima consonante.
- La palabra final del verso 4 debe ser una palabra completa y natural en espanol.
- No fuerces terminaciones artificiales.
- No termines el verso con fragmentos como "eva", "argos", "ido" o similares si no funcionan como palabra real.
```

Estas reglas deben añadirse sin eliminar las reglas ya existentes:

```text
- Reescribe solo el verso 4.
- No modifiques el verso 1.
- No modifiques los versos 2 y 3.
- El nuevo verso 4 debe rimar consonantemente con el verso 1.
- Intenta que el nuevo verso 4 tenga 11 silabas metricas.
- Devuelve solo variantes del verso 4, no la estrofa completa.
- No anadas titulo.
- No numeres las variantes.
- Responde solo con JSON valido.
```

## Usar mas la palabra ancla

El prompt debe apoyarse mas en la palabra final del verso 1.

Actualmente ya se incluye:

```python
verse_1_last_word = verse_1.get("last_word", "")
target_rhyme = outer_rhyme_diagnosis.get("target_rhyme", "")
```

El prompt debe explicar:

```text
El verso 1 termina en "lleva".
El verso 4 no debe terminar necesariamente en "lleva", pero debe terminar en una palabra que rime consonantemente con "lleva".
```

Esto ayuda al modelo a buscar una palabra completa que rime con la palabra ancla, no solo con la cadena `eva`.

## Ayuda opcional con ejemplos de palabra final

Se puede crear una funcion auxiliar pequena en `src/langgraph_beam_stanza.py`:

```python
def build_rhyme_target_hint(
    anchor_word: str | None,
    target_rhyme: str | None,
) -> str:
    ...
```

Esta funcion debe devolver una ayuda textual para el prompt.

No debe usar diccionarios externos ni librerias nuevas.

Puede funcionar con reglas simples y casos concretos observados:

```python
RHYME_HINT_EXAMPLES = {
    "eva": ["lleva", "nueva", "eleva", "conlleva", "nieva"],
    "argos": ["amargos", "largos"],
    "argo": ["amargo", "largo"],
    "oria": ["memoria", "historia", "gloria"],
    "ido": ["dormido", "perdido", "olvido"],
    "ida": ["vida", "herida", "partida"],
}
```

Uso esperado:

```python
hint = build_rhyme_target_hint(verse_1_last_word, target_rhyme)
```

Ejemplo de salida:

```text
Ejemplos de palabras finales compatibles con la rima "eva": lleva, nueva, eleva, conlleva, nieva.
Usa una palabra real de este tipo o una equivalente; no escribas solo "eva".
```

Si no hay ejemplos para una rima:

```text
Busca una palabra real que comparta la rima consonante "..." con "..."; no copies solo la terminacion.
```

Esta funcion es opcional, pero recomendable porque el caso `eva` ya ha fallado de forma clara.

## Cambios concretos en `_build_outer_rhyme_repair_messages`

Dentro de `_build_outer_rhyme_repair_messages(...)`, despues de obtener:

```python
verse_1_text = verse_1.get("text", "")
verse_1_last_word = verse_1.get("last_word", "")
target_rhyme = outer_rhyme_diagnosis.get("target_rhyme", "")
current_rhyme = outer_rhyme_diagnosis.get("current_rhyme", "")
```

anadir:

```python
rhyme_target_hint = build_rhyme_target_hint(
    anchor_word=verse_1_last_word,
    target_rhyme=target_rhyme,
)
```

Luego incluir `rhyme_target_hint` en el `user_prompt`, cerca de la informacion de rima.

Ejemplo:

```python
f"Orientacion para la palabra final del verso 4:\n{rhyme_target_hint}\n\n"
```

## Prompt esperado tras el cambio

El prompt final de reparacion A debe contener ideas equivalentes a:

```text
Verso 1 como ancla:
El tiempo, un rastro que el viento lleva
Palabra final del verso 1: lleva
Rima consonante objetivo del verso 1: eva

Orientacion para la palabra final del verso 4:
La rima objetivo "eva" es una terminacion de rima, no una palabra que debas copiar literalmente.
El verso 4 debe terminar en una palabra real que tenga esa rima consonante.
Ejemplos de palabras finales compatibles: lleva, nueva, eleva, conlleva, nieva.
No escribas solo "eva" como palabra final.
```

Y en restricciones:

```text
- Termina el verso 4 con una palabra real que tenga esa rima consonante.
- No copies literalmente la rima objetivo como palabra final si no es una palabra natural.
- No termines el verso con fragmentos artificiales.
```

## Salida esperada tras el cambio

Para una rima objetivo:

```text
eva
```

Las variantes deberian tender a terminar en palabras como:

```text
nueva
eleva
lleva
conlleva
nieva
```

Ejemplos aceptables en cuanto a rima:

```text
Que en mi memoria lentamente nieva
Y al viejo corazon la sombra eleva
La tarde vuelve con su luz nueva
```

Ejemplos que deben evitarse:

```text
Que nunca mas se olvidara eva
Que el olvido jamas mas sea eva
```

El filtro de aceptacion debe seguir encargandose de comprobar:

- si AXYA queda correcta;
- si la metrica del verso 4 no empeora.

## Pruebas minimas recomendadas

### 1. Probar la funcion de ayuda, si se implementa

```powershell
$env:PYTHONPATH='.\src'
python -c "from langgraph_beam_stanza import build_rhyme_target_hint; print(build_rhyme_target_hint('lleva', 'eva'))"
```

Debe aparecer un texto que diga que:

- `eva` es terminacion, no palabra obligatoria;
- debe usarse una palabra real;
- se sugieren palabras como `lleva`, `nueva`, `eleva`.

### 2. Probar el prompt generado

Sin llamar a Ollama, inspeccionar:

```powershell
$env:PYTHONPATH='.\src'
python -c "from sonnet_metrics import diagnose_stanza_outer_rhyme; from langgraph_beam_stanza import _build_outer_rhyme_repair_messages; stanza=['El tiempo, un rastro que el viento lleva','En la memoria, un recuerdo amargo dulce','Corazon, triste profundidad alli','Que nunca se olvidara jamas mas']; d=diagnose_stanza_outer_rhyme(stanza); msgs=_build_outer_rhyme_repair_messages('test', stanza, d, 5); print(msgs[1]['content'])"
```

El texto debe contener:

```text
No escribas solo "eva" como palabra final
palabra real
lleva, nueva, eleva
```

### 3. Ejecutar una prueba real con Ollama

Tras implementar el cambio, ejecutar el flujo completo y revisar la traza.

Se espera que las variantes ya no terminen en:

```text
eva
```

sino en palabras reales compatibles.

## Criterio de exito del Paso 3c

El Paso 3c estara completo cuando:

1. El prompt de reparacion A explique que la rima objetivo es una terminacion, no una palabra a copiar.
2. El prompt indique explicitamente que el verso 4 debe terminar en una palabra real.
3. El prompt prohiba terminar con fragmentos artificiales como `eva` cuando no sean naturales.
4. Para la rima `eva`, el prompt sugiera ejemplos como `lleva`, `nueva`, `eleva`, `conlleva`, `nieva`.
5. No se haya cambiado el criterio de aceptacion de variantes.
6. No se haya implementado reparacion de la rima interior B.
7. No se haya tocado la reparacion metrica.

## Que NO hacer en este paso

No implementar todavia:

- reparacion de rima interior B;
- cambios en `repair_stanza_outer_rhyme_with_ollama(...)`;
- cambios en el criterio de aceptacion;
- cambios en `sonnet_metrics.py`;
- cambios en Beam Search;
- cambios en pesos de scoring;
- diccionarios externos de rima;
- librerias nuevas.

Este paso se limita a mejorar el prompt de reparacion A para que el LLM genere palabras finales reales compatibles con la rima objetivo.
