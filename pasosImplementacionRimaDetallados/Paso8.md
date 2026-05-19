# Paso 8: refuerzo de palabras candidatas para la reparacion de rima interior B

## Objetivo del paso

El objetivo de este paso es mejorar la reparacion de la rima interior B reforzando el mismo mecanismo que ya hizo funcionar razonablemente la reparacion de la rima exterior A:

```text
Usar palabras finales candidatas concretas para guiar al LLM.
```

La reparacion B ya esta implementada y se ejecuta, pero las trazas muestran que casi nunca acepta variantes. El motivo principal observado no parece ser que la reparacion B este mal conectada, sino que muchas veces no tiene palabras candidatas utiles.

Ejemplos observados en ejecuciones reales:

```text
target_rhyme = ombra
candidate_final_words = ["sombra"]
enforced_candidate_final_words = []
```

```text
target_rhyme = ende
candidate_final_words = ["extiende"]
enforced_candidate_final_words = []
```

```text
target_rhyme = eso
candidate_final_words = ["retroceso"]
enforced_candidate_final_words = []
```

Cuando `enforced_candidate_final_words` queda vacia, el prompt no puede imponer una lista concreta de palabras finales distintas de la palabra ancla. En ese caso, el LLM vuelve a generar versos bastante libres y normalmente no corrige B.

En cambio, cuando la rima objetivo si existe en `RHYME_HINT_EXAMPLES`, el sistema se comporta mejor. Por ejemplo:

```text
target_rhyme = eva
candidate_final_words = ["lleva", "nueva", "eleva", "conlleva", "nieva"]
enforced_candidate_final_words = ["lleva", "eleva", "conlleva", "nieva"]
```

En ese caso se ha observado que el modelo si puede llegar a generar una palabra final que corrige la rima, aunque luego pueda fallar por metrica.

Por tanto, este paso debe reforzar la lista de palabras candidatas para que la reparacion B tenga mas casos guiados.

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

No debe tocar:

- `src/sonnet_metrics.py`;
- la extraccion de rima;
- el contador de silabas;
- la reparacion metrica;
- la reparacion de rima exterior A como estrategia;
- la reparacion de rima interior B como estrategia;
- el Beam Search;
- `evaluate_stanza_abba`;
- `aggregate_scores(...)`;
- los pesos de scoring;
- `k`;
- `max_steps`;
- `alpha`;
- la medicion de tiempo.

Este paso debe limitarse a:

```text
1. ampliar y ordenar mejor las palabras candidatas de rima;
2. mejorar el diagnostico/reporting cuando no hay candidatas utiles para B;
3. mantener intacta la logica de aceptacion de variantes.
```

## Que NO se debe hacer en este paso

No implementar todavia:

- reparacion metrica posterior a la reparacion B;
- una segunda llamada al LLM para generar solo palabras candidatas;
- diccionarios externos;
- librerias nuevas;
- busqueda en internet;
- reparacion simultanea de versos 2 y 3;
- reescritura del verso 2;
- cambios en scoring;
- cambios en Beam Search;
- cambios en `sonnet_metrics.py`;
- cambios en el contador de silabas;
- cambios en la extraccion de rima.

La reparacion metrica posterior a B queda explicitamente fuera de este paso.

## Diagnostico del problema actual

La reparacion B usa:

```python
get_candidate_final_words_for_rhyme(
    anchor_word=anchor_word,
    target_rhyme=target_rhyme,
)
```

Esta funcion mira `RHYME_HINT_EXAMPLES`.

Si la rima objetivo existe en `RHYME_HINT_EXAMPLES`, devuelve ejemplos utiles mas la palabra ancla.

Si la rima objetivo no existe, devuelve solo la palabra ancla.

Despues, la reparacion B calcula:

```python
enforced_candidate_final_words = [
    word for word in candidate_final_words if word != anchor_word
]
```

Por tanto:

- si hay ejemplos distintos de la palabra ancla, se puede imponer una lista concreta;
- si solo esta la palabra ancla, no hay lista obligatoria;
- si no hay lista obligatoria, el LLM suele no corregir B.

El Paso 8 debe reducir el numero de casos en los que `enforced_candidate_final_words` queda vacia.

## Funcion afectada principal

La funcion principal afectada sera:

```python
get_candidate_final_words_for_rhyme(...)
```

Y, de forma indirecta, el mapa:

```python
RHYME_HINT_EXAMPLES
```

No se debe cambiar la firma de `get_candidate_final_words_for_rhyme(...)` en este paso.

No se debe cambiar su comportamiento general:

```text
1. ejemplos de RHYME_HINT_EXAMPLES primero;
2. palabra ancla si no esta ya;
3. eliminar vacios;
4. eliminar duplicados conservando orden.
```

Lo que se debe mejorar es el contenido de `RHYME_HINT_EXAMPLES` y el reporte cuando no hay candidatas distintas.

## Ampliacion de `RHYME_HINT_EXAMPLES`

Ampliar el diccionario `RHYME_HINT_EXAMPLES` con rimas que han aparecido en las ejecuciones reales y que actualmente dejan a B sin candidatas utiles.

### Rimas observadas que deben anadirse

Anadir, como minimo, las siguientes entradas:

```python
"ombra": ["sombra", "alfombra", "asombra", "nombra"],
"ende": ["extiende", "enciende", "defiende", "comprende", "aprende"],
"eso": ["retroceso", "proceso", "exceso", "regreso"],
"ar": ["llorar", "pasar", "mirar", "soñar", "callar", "recordar"],
"isima": ["finisima", "bellisima", "tristisima", "purisima"],
"iguo": ["antiguo", "contiguo"],
"igo": ["conmigo", "testigo", "abrigo", "amigo"],
"anca": ["arranca", "banca", "blanca"],
"anco": ["blanco", "franco", "banco"],
"ena": ["pena", "condena", "arena", "cadena"],
"ido": ["dormido", "perdido", "olvido", "herido", "partido"],
"ida": ["vida", "herida", "partida", "salida", "caida"],
"ado": ["helado", "olvidado", "callado", "pasado", "soñado", "amado"],
"ura": ["oscura", "ternura", "locura", "altura", "llanura"],
"or": ["dolor", "amor", "temor", "rumor", "ardor"],
"ente": ["presente", "ausente", "siente", "puente", "frente"],
"ero": ["sendero", "entero", "primero", "sincero"],
"ante": ["instante", "distante", "errante", "constante"],
```

### Notas sobre tildes

El sistema de extraccion de rima normaliza tildes para comparar rimas, pero las palabras finales candidatas deben ser palabras naturales.

Por simplicidad y consistencia con el codigo actual, las claves de `RHYME_HINT_EXAMPLES` deben estar en la forma de rima devuelta por `extract_verse_rhyme(...)`, normalmente en minusculas y sin tildes.

Las palabras candidatas pueden escribirse sin tilde si el codigo actual trabaja asi en prompts, pero es preferible usar palabras naturales cuando no introduzca problemas.

Para evitar inconsistencias en comparacion con `get_last_word(...)`, en este paso se recomienda usar candidatas sin tilde en los casos donde la clave tampoco la lleva:

```python
"isima": ["finisima", "bellisima", "tristisima", "purisima"]
```

No introducir todavia normalizacion adicional de candidatas.

## Cuidado con palabras poco naturales

No se deben anadir palabras inventadas solo para forzar la rima.

Las candidatas deben ser palabras reales y razonablemente utiles en contexto poetico.

Evitar listas excesivamente largas. El objetivo no es crear un diccionario completo, sino reforzar las rimas que estan apareciendo en el experimento.

Cada entrada deberia tener entre 2 y 6 palabras candidatas.

## Mejoras de reporte para falta de candidatas B

Actualmente el reporte B ya guarda:

```python
"candidate_final_words"
"enforced_candidate_final_words"
```

Este paso debe anadir una indicacion clara de si habia alternativas utiles distintas del ancla.

### Campo nuevo recomendado

En `build_inner_rhyme_repair_report(...)`, anadir:

```python
"has_enforced_candidate_final_words": False,
```

En `repair_stanza_inner_rhyme_with_ollama(...)`, despues de calcular:

```python
enforced_candidate_final_words = [
    word for word in candidate_final_words if word != anchor_word
]
```

guardar:

```python
report["has_enforced_candidate_final_words"] = bool(enforced_candidate_final_words)
```

Esto permitira ver rapidamente si B estaba guiada por una lista concreta o no.

### Motivo informativo cuando no hay candidatas

Si `enforced_candidate_final_words` esta vacia, no se debe abortar la reparacion.

Pero el reporte debe conservar una pista clara.

Anadir al reporte:

```python
"candidate_warning": None,
```

En `repair_stanza_inner_rhyme_with_ollama(...)`, si no hay candidatas forzables y existe `anchor_word`, asignar:

```python
report["candidate_warning"] = (
    "No hay palabras finales candidatas distintas de la palabra ancla para la rima B."
)
```

Si si hay candidatas, dejar:

```python
report["candidate_warning"] = None
```

No usar este warning como motivo de rechazo.

No cambiar `reason` por este warning si la reparacion sigue intentando variantes.

## Cambios opcionales en la rima A

Como `RHYME_HINT_EXAMPLES` es compartido por A y B, ampliar este diccionario tambien puede beneficiar a A.

Eso es aceptable.

Pero no se debe cambiar:

- el prompt de A;
- el criterio de aceptacion de A;
- los indices de A;
- el informe de A;
- el orden de reparacion A.

Solo se permite que A vea mas palabras candidatas porque el mapa compartido ha crecido.

## Cambios en `summarize_inner_rhyme_repair_report(...)`

No es obligatorio cambiar el resumen.

Si se cambia, debe ser muy breve y no romper el formato actual.

Opcion recomendada:

Mantenerlo como esta:

```text
reparacion rima B = activa | intento = True | cambio = False
```

El diagnostico detallado de falta de candidatas debe consultarse en el JSON, no en la linea resumen.

## Cambios en trazas y JSON

No hace falta cambiar `build_trace_entry(...)` si ya incluye completo:

```python
"inner_rhyme_repair_report": inner_rhyme_repair_report
```

Al anadir los nuevos campos al reporte:

```python
"has_enforced_candidate_final_words"
"candidate_warning"
```

apareceran automaticamente en:

- `final_stanza_metrics_*.json`;
- `final_stanza_trace_*.json`.

No es necesario anadir nuevas claves fuera del reporte.

## Pruebas minimas recomendadas

### 1. Probar nuevas candidatas directamente

Ejecutar:

```powershell
$env:PYTHONPATH='.\src'
python -c "from langgraph_beam_stanza import get_candidate_final_words_for_rhyme; print(get_candidate_final_words_for_rhyme('sombra', 'ombra')); print(get_candidate_final_words_for_rhyme('extiende', 'ende')); print(get_candidate_final_words_for_rhyme('retroceso', 'eso'))"
```

Salida esperada aproximada:

```text
['sombra', 'alfombra', 'asombra', 'nombra']
['extiende', 'enciende', 'defiende', 'comprende', 'aprende']
['retroceso', 'proceso', 'exceso', 'regreso']
```

Si la palabra ancla ya aparece en la lista, no debe duplicarse.

### 2. Probar que B tiene candidatas forzables

Con una estrofa como:

```python
stanza = [
    "El tiempo se pierde en la antigua sombra",
    "La memoria despierta bajo sombra",
    "El corazon se rompe y nunca vuelve",
    "Y la tarde se queda fria y asombra",
]
```

Diagnosticar B:

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
Palabras finales candidatas para el nuevo verso 3
alfombra
asombra
nombra
```

Y no debe forzar `sombra` si hay alternativas.

### 3. Probar reporte cuando hay candidatas

Usando una reparacion B simulada con monkeypatch, comprobar que el reporte incluye:

```python
"has_enforced_candidate_final_words": True
"candidate_warning": None
```

cuando la rima objetivo tiene candidatas distintas.

### 4. Probar reporte cuando no hay candidatas

Usar una rima objetivo que no exista en `RHYME_HINT_EXAMPLES`.

Comprobar que:

```python
"has_enforced_candidate_final_words": False
"candidate_warning": "No hay palabras finales candidatas distintas de la palabra ancla para la rima B."
```

La reparacion debe seguir intentandose.

### 5. Probar que no cambia la logica de aceptacion

Con monkeypatch de:

```python
generate_inner_rhyme_repair_variants_with_ollama
```

comprobar que una variante solo se acepta si:

```text
1. corrige la rima interior B;
2. no empeora la distancia metrica del verso 3;
3. termina en palabra candidata si hay lista forzable.
```

Este paso no debe relajar el filtro metrico.

### 6. Probar compilacion

Ejecutar:

```powershell
python -m compileall src
```

Debe compilar sin errores.

## Criterio de exito del Paso 8

El Paso 8 estara completo cuando:

1. `RHYME_HINT_EXAMPLES` incluya nuevas rimas frecuentes observadas en B.
2. `get_candidate_final_words_for_rhyme(...)` devuelva alternativas utiles para rimas como `ombra`, `ende`, `eso`, `ar`, `isima`, `iguo`, `ena`, etc.
3. El prompt B muestre candidatas distintas del ancla cuando existan.
4. El prompt B no fuerce la misma palabra final del verso 2 si hay alternativas.
5. `inner_rhyme_repair_report` incluya `has_enforced_candidate_final_words`.
6. `inner_rhyme_repair_report` incluya `candidate_warning`.
7. La reparacion B siga intentando variantes aunque no haya candidatas forzables.
8. No se haya implementado reparacion metrica posterior a B.
9. No se haya cambiado el criterio de aceptacion de B.
10. No se haya cambiado la reparacion A salvo por beneficiarse del diccionario compartido ampliado.
11. No se haya tocado `sonnet_metrics.py`.
12. No se haya cambiado scoring, Beam Search, `k`, `max_steps` ni `alpha`.

## Resultado esperado

Tras este paso, en mas casos la reparacion B deberia pasar de prompts libres como:

```text
Busca una palabra real distinta que comparta la misma rima consonante.
```

a prompts guiados como:

```text
Palabras finales candidatas para el nuevo verso 3:
- alfombra
- asombra
- nombra

Cada variante debe terminar exactamente en una de esas palabras.
```

Esto no garantiza que B se repare siempre, porque todavia puede fallar la metrica. Pero deberia reducir claramente los fallos causados por falta de palabras finales candidatas.
