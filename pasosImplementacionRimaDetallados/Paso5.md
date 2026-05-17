# Paso 5: conservacion de mejores beams y proteccion de rimas ya correctas

## Objetivo del paso

El objetivo de este paso es evitar que el sistema destruya una estrofa buena que ya habia encontrado en pasos anteriores del Beam Search.

Se ha observado un caso claro:

```text
Step 0:
- 4/4 versos endecasilabos.
- Rima exterior A correcta.
- Rima total 3/4.
- Score formal 18/20.

Step 1:
- El modelo reescribe la estrofa.
- Cambia versos que ya estaban bien.
- Rompe la rima exterior A.
- Baja a 14/20 o 12/20.
```

El problema no esta en la evaluacion formal, sino en la dinamica de busqueda:

```text
El Beam Search sustituye beams buenos por nuevas variantes peores.
```

Ademas, el prompt de correccion permite que el modelo modifique versos cuya metrica o rima ya estaban correctas.

Por tanto, este paso debe introducir dos mecanismos:

```text
1. Elitismo: conservar beams buenos ya encontrados.
2. Proteccion en prompt: indicar al modelo que no modifique versos asociados a restricciones ya correctas.
```

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

No debe tocar:

- `src/sonnet_metrics.py`;
- la extraccion de rima;
- la reparacion metrica;
- la reparacion de rima exterior A;
- la reparacion de rima interior B;
- `evaluate_stanza_abba`;
- los pesos internos de la evaluacion formal;
- la generacion de sonetos completos.

Este paso puede tocar el Beam Search solo en este sentido:

```text
Permitir que beams ya existentes sobrevivan al prune si siguen siendo mejores que las nuevas variantes.
```

No se debe cambiar todavia:

- `k`;
- `max_steps`;
- `alpha`;
- la formula de `aggregate_scores(...)`;
- el criterio formal de evaluacion.

## Problema que se quiere resolver

Actualmente el flujo es:

```text
1. `expand_node` genera nuevas estrofas a partir de cada beam.
2. `score_node` evalua solo esas nuevas estrofas.
3. `prune_node` selecciona los k mejores candidatos nuevos.
```

El beam original no compite contra sus descendientes.

Eso implica que una estrofa con score 0.900 puede desaparecer si sus variantes generadas tienen score 0.700 y 0.600.

El resultado es que el sistema puede empeorar con cada iteracion, incluso cuando ya habia encontrado una solucion muy cercana al objetivo.

## Solucion propuesta

### Parte A: elitismo en el Beam Search

El `prune_node(...)` debe seleccionar los mejores elementos a partir de:

```text
candidatos nuevos + beams previos conservables
```

Es decir, si un beam previo ya tenia una estrofa valida y buen score, debe poder sobrevivir al siguiente paso.

Ejemplo:

```text
Beam previo:
- score = 0.900

Nuevos candidatos:
- score = 0.700
- score = 0.600

Resultado esperado con elitismo:
- Se conserva el beam previo de 0.900.
- Se elige como segundo beam el mejor candidato nuevo, si k = 2.
```

### Parte B: proteccion de restricciones ya correctas en el prompt

Cuando se genera una correccion de una estrofa previa, el prompt debe informar al modelo de que algunas partes ya estan bien y no deben modificarse.

Ejemplo observado:

```text
Verso 1: El tiempo, un rio que pasa sin cesar
Verso 4: Como hojas en otono, sin regresar
```

Si la rima exterior A ya es correcta, el prompt debe decir explicitamente:

```text
La rima exterior A ya es correcta.
No modifiques el verso 1 ni el verso 4 salvo que sea imprescindible.
Conserva sus palabras finales.
```

Si la metrica de un verso ya es correcta, el prompt tambien puede indicarlo como informacion de conservacion:

```text
Los versos que ya tienen 11 silabas metricas deben conservarse siempre que sea posible.
```

El objetivo no es crear todavia una reparacion de rima interior B, sino evitar que el modelo rompa lo que ya esta correcto.

## Nuevas constantes

Anadir en la zona de constantes globales de `src/langgraph_beam_stanza.py`:

```python
ENABLE_BEAM_ELITISM = True
ELITE_BEAMS_TO_KEEP = 1
ENABLE_PROMPT_CONSTRAINT_PROTECTION = True
```

### Significado

`ENABLE_BEAM_ELITISM`:

- activa o desactiva la conservacion de beams previos durante el prune.

`ELITE_BEAMS_TO_KEEP`:

- numero maximo de beams previos que pueden conservarse como elites.
- para este paso debe ser `1`.

`ENABLE_PROMPT_CONSTRAINT_PROTECTION`:

- activa o desactiva las instrucciones de proteccion en el prompt de correccion.

## Funciones nuevas

### 1. `is_preservable_beam(...)`

Crear una funcion auxiliar:

```python
def is_preservable_beam(beam: dict[str, Any]) -> bool:
    ...
```

Debe devolver `True` solo si:

- el beam tiene una estrofa;
- la estrofa es una lista;
- la estrofa no esta vacia;
- el beam tiene historial de scores;
- el historial de scores no esta vacio.

Debe devolver `False` para el beam inicial vacio.

Motivo:

```text
No queremos conservar el estado inicial sin estrofa como elite.
```

### 2. `mark_beam_as_elite_candidate(...)`

Crear una funcion auxiliar:

```python
def mark_beam_as_elite_candidate(beam: dict[str, Any]) -> dict[str, Any]:
    ...
```

Debe:

- hacer una copia superficial del beam;
- copiar tambien la lista `stanza`, si existe;
- copiar tambien `score_history`, si existe;
- anadir:

```python
"preserved_by_elitism": True
```

- mantener el resto de informacion del beam:
  - `score`;
  - `step_score`;
  - `feedback`;
  - `metrics`;
  - `meter_repair_report`;
  - `outer_rhyme_repair_report`;
  - `outer_rhyme_diagnosis`;
  - `inner_rhyme_diagnosis`;
  - `generation_reasoning`.

No debe reevaluar la estrofa.

### 3. `deduplicate_beams_by_stanza(...)`

Crear una funcion auxiliar:

```python
def deduplicate_beams_by_stanza(beams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ...
```

Debe:

- eliminar duplicados basandose en la estrofa completa;
- conservar el primer beam encontrado para cada estrofa;
- mantener el orden recibido.

La clave puede construirse como:

```python
tuple(str(verse).strip() for verse in stanza)
```

Si el beam no tiene estrofa valida, puede usarse una clave unica para no eliminarlo accidentalmente.

## Cambios en `expand_node(...)`

En `expand_node(...)`, al crear cada candidato generado, anadir:

```python
"preserved_by_elitism": False
```

Esto permite distinguir:

```text
- candidatos nuevos generados por el LLM;
- beams previos conservados por elitismo.
```

No debe cambiarse la generacion de candidatos.

No se debe anadir aqui el beam previo como candidato. La conservacion debe hacerse en `prune_node(...)`.

## Cambios en `prune_node(...)`

Actualmente:

```python
sorted_candidates = sorted(
    state["candidates"],
    key=lambda beam: beam["score"],
    reverse=True,
)
top_k = sorted_candidates[: state["k"]]
```

Debe modificarse para construir antes una bolsa de seleccion:

```python
selection_pool = list(state["candidates"])
```

Si `ENABLE_BEAM_ELITISM` esta activo:

1. Obtener beams previos desde:

```python
previous_beams = state.get("beams", [])
```

2. Filtrar solo los preservables:

```python
preservable_beams = [
    mark_beam_as_elite_candidate(beam)
    for beam in previous_beams
    if is_preservable_beam(beam)
]
```

3. Ordenarlos por score descendente.

4. Tomar como maximo:

```python
ELITE_BEAMS_TO_KEEP
```

5. Anadirlos a `selection_pool`.

6. Deduplicar por estrofa:

```python
selection_pool = deduplicate_beams_by_stanza(selection_pool)
```

7. Ordenar `selection_pool` por score descendente.

8. Tomar los `k` mejores.

### Comportamiento esperado

Si un beam previo tiene score mayor que todos sus descendientes, debe conservarse.

Si un candidato nuevo mejora el score del beam previo, el candidato nuevo debe poder sustituirlo.

Si un candidato nuevo es igual textualmente al beam previo, debe evitarse duplicarlo.

## Cambios en trazas y salida por terminal

En `build_trace_entry(...)`, incluir:

```python
"preserved_by_elitism": bool(item.get("preserved_by_elitism", False))
```

En `prune_node(...)`, al imprimir cada beam seleccionado, anadir una linea:

```text
Conservado por elitismo: True/False
```

Esto permitira verificar rapidamente si un beam ha sobrevivido sin ser regenerado.

Tambien debe aparecer en las trazas JSON gracias a `build_trace_entry(...)`.

## Cambios en la proteccion del prompt

### Funcion nueva: `build_constraint_protection_prompt(...)`

Crear una funcion auxiliar cerca de las funciones de construccion de prompt:

```python
def build_constraint_protection_prompt(previous_beam: dict[str, Any] | None) -> str:
    ...
```

Debe devolver una cadena vacia si:

- `ENABLE_PROMPT_CONSTRAINT_PROTECTION` esta en `False`;
- no hay beam previo;
- no hay estrofa previa.

Si hay beam previo, debe usar la informacion ya disponible en el beam:

- `outer_rhyme_diagnosis`;
- `inner_rhyme_diagnosis`;
- `metrics`.

No debe recalcular metricas dentro de esta funcion.

### Contenido minimo del texto

Si `outer_rhyme_diagnosis["is_valid"]` es `True`, debe incluir:

```text
La rima exterior A ya es correcta.
Conserva el verso 1 y el verso 4 siempre que sea posible.
Conserva especialmente las palabras finales del verso 1 y del verso 4.
```

Si `inner_rhyme_diagnosis["is_valid"]` es `True`, debe incluir:

```text
La rima interior B ya es correcta.
Conserva el verso 2 y el verso 3 siempre que sea posible.
Conserva especialmente las palabras finales del verso 2 y del verso 3.
```

Si todos los versos son endecasilabos segun `metrics`, debe incluir:

```text
Todos los versos ya tienen 11 silabas metricas.
No alargues ni acortes versos que ya cumplen la metrica.
```

Si solo algunos versos son endecasilabos, puede incluir:

```text
Conserva los versos que ya tienen 11 silabas metricas siempre que sea posible.
```

La funcion debe limitarse a construir texto para el prompt.

No debe bloquear candidatos ni modificar estrofas.

## Cambio de firma en `_build_stanza_generation_messages(...)`

La funcion actual es:

```python
def _build_stanza_generation_messages(
    question: str,
    previous_stanza: list[str] | None,
    feedback: str | None,
    num_candidates: int,
    strict: bool = False,
) -> List[Dict[str, str]]:
```

Debe cambiarse a:

```python
def _build_stanza_generation_messages(
    question: str,
    previous_stanza: list[str] | None,
    feedback: str | None,
    num_candidates: int,
    strict: bool = False,
    previous_beam: dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
```

Dentro de la funcion, construir:

```python
constraint_protection = build_constraint_protection_prompt(previous_beam)
```

Si `previous_stanza` existe y `constraint_protection` no esta vacio, incluirlo en el prompt de correccion de estrofa previa bajo una seccion:

```text
RESTRICCIONES YA SATISFECHAS QUE DEBES CONSERVAR:
...
```

No debe incluirse esta seccion cuando se genera desde cero.

## Cambio de firma en `generate_stanza_candidates_with_ollama(...)`

Si esta funcion llama a `_build_stanza_generation_messages(...)`, debe aceptar tambien:

```python
previous_beam: dict[str, Any] | None = None
```

Y pasarlo a `_build_stanza_generation_messages(...)`.

## Cambio en `expand_node(...)` para pasar el beam previo

En la llamada:

```python
generated_stanzas = generate_stanza_candidates_with_ollama(
    question=state["question"],
    previous_stanza=previous_stanza,
    feedback=current_feedback,
    num_candidates=state["k"],
)
```

Anadir:

```python
previous_beam=beam,
```

Asi el constructor del prompt puede saber que restricciones ya estaban satisfechas.

## Que NO hacer en este paso

No implementar todavia:

- reparacion de rima interior B;
- nuevos prompts especificos para reparar B;
- nuevos pesos de scoring;
- cambios en `evaluate_stanza_abba`;
- cambios en `sonnet_metrics.py`;
- reparacion metrica posterior a la reparacion de rima;
- diccionarios externos;
- librerias nuevas;
- cambios de modelo;
- cambios de temperatura;
- cambios de `k`, `max_steps` o `alpha`.

Este paso no debe intentar mejorar la rima B.

Este paso solo debe evitar que el proceso pierda buenos beams y rompa restricciones ya correctas.

## Pruebas minimas recomendadas

### 1. Probar que el beam inicial vacio no se conserva

Crear o simular un beam inicial sin estrofa:

```python
{"stanza": [], "score_history": []}
```

`is_preservable_beam(...)` debe devolver:

```python
False
```

### 2. Probar que un beam con estrofa y score se conserva

Crear o simular:

```python
{
    "stanza": ["v1", "v2", "v3", "v4"],
    "score": 0.9,
    "score_history": [0.9],
}
```

`is_preservable_beam(...)` debe devolver:

```python
True
```

`mark_beam_as_elite_candidate(...)` debe anadir:

```python
"preserved_by_elitism": True
```

### 3. Probar seleccion con elitismo

Simular:

```text
previous_beam score = 0.900
candidate_1 score = 0.700
candidate_2 score = 0.600
k = 2
```

Resultado esperado:

```text
top_k contiene el previous_beam de 0.900
top_k contiene candidate_1 de 0.700
candidate_2 queda fuera
```

### 4. Probar prompt de proteccion

Simular un beam previo con:

```python
"outer_rhyme_diagnosis": {"is_valid": True}
```

El prompt generado en modo correccion debe contener:

```text
RESTRICCIONES YA SATISFECHAS QUE DEBES CONSERVAR
La rima exterior A ya es correcta
Conserva el verso 1 y el verso 4
Conserva especialmente las palabras finales
```

### 5. Probar ejecucion real

Ejecutar el flujo completo.

En la salida por terminal debe aparecer:

```text
Conservado por elitismo: True
```

cuando un beam previo sobreviva a una iteracion posterior.

En la traza JSON debe aparecer:

```json
"preserved_by_elitism": true
```

## Criterio de exito del Paso 5

El Paso 5 estara completo cuando:

1. Existan las constantes:
   - `ENABLE_BEAM_ELITISM`;
   - `ELITE_BEAMS_TO_KEEP`;
   - `ENABLE_PROMPT_CONSTRAINT_PROTECTION`.
2. Exista `is_preservable_beam(...)`.
3. Exista `mark_beam_as_elite_candidate(...)`.
4. Exista `deduplicate_beams_by_stanza(...)`.
5. `prune_node(...)` pueda seleccionar beams previos junto a candidatos nuevos.
6. El beam inicial vacio no pueda conservarse por elitismo.
7. Los candidatos nuevos tengan `preserved_by_elitism = False`.
8. Los beams preservados tengan `preserved_by_elitism = True`.
9. `build_trace_entry(...)` incluya `preserved_by_elitism`.
10. La terminal muestre si un beam fue conservado por elitismo.
11. Exista `build_constraint_protection_prompt(...)`.
12. El prompt de correccion incluya restricciones ya satisfechas cuando proceda.
13. No se haya implementado reparacion de rima interior B.
14. No se haya tocado `sonnet_metrics.py`.
15. No se haya cambiado la formula de scoring ni los pesos formales.

## Resultado esperado

Con este paso, si el sistema encuentra una estrofa como:

```text
Score formal: 18/20
Endecasilabos: 4/4
Rima exterior A: correcta
Rima interior B: incorrecta
```

esa estrofa no debe desaparecer simplemente porque las siguientes variantes generadas sean peores.

Ademas, cuando se pida una correccion de esa estrofa, el prompt debe advertir al modelo de que la rima exterior A ya esta correcta y que debe conservar los versos 1 y 4 siempre que sea posible.
