# Paso 3: refinamiento de la rima exterior A reescribiendo solo el verso 4

## Objetivo del paso

El objetivo de este tercer paso es implementar el primer refinamiento real de rima, pero de forma muy limitada y controlada.

Se va a trabajar solo la pareja exterior A:

```text
Verso 1 -> A
Verso 4 -> A
```

Si el diagnostico `outer_rhyme_diagnosis` indica que el verso 4 no rima con el verso 1, se pediran variantes al LLM para reescribir solo el verso 4.

En este paso NO se va a reparar todavia la pareja interior B:

```text
Verso 2 -> B
Verso 3 -> B
```

Tampoco se va a pedir al LLM que regenere la estrofa completa. La unidad de reparacion sera un unico verso: el verso 4.

## Resumen del comportamiento deseado

Para cada estrofa candidata:

1. Se mantiene el flujo actual de generacion y reparacion metrica.
2. Se calcula el diagnostico de rima exterior AXYA.
3. Si la rima exterior ya es correcta, no se hace nada.
4. Si la rima exterior falla, se pide al LLM que proponga variantes solo del verso 4.
5. Cada variante se inserta temporalmente en la estrofa.
6. Se comprueba si la nueva estrofa cumple la rima exterior AXYA.
7. Se comprueba tambien la metrica del nuevo verso 4.
8. Se acepta una variante solo si mejora la rima exterior y no empeora la metrica del verso 4.
9. Si ninguna variante es aceptable, se conserva la estrofa original.
10. Tras la reparacion, se recalculan las metricas y diagnosticos.

## Por que empezar por el verso 4

La rima exterior A tiene dos versos:

```text
Verso 1
Verso 4
```

El verso 1 se usara como ancla. Su rima consonante sera la rima objetivo.

Por tanto:

- no se debe modificar el verso 1;
- no se deben modificar los versos 2 y 3;
- solo se debe intentar reescribir el verso 4;
- el verso 4 debe terminar con la misma rima consonante que el verso 1.

Ejemplo:

```text
Verso 1: Vuelan anos, dejan huella en el alma
Rima objetivo: alma

Verso 4 actual: Pasa el tiempo, dejandonos sin vida
Rima actual: ida
```

El reparador debe intentar crear un nuevo verso 4 que rime con `alma`.

## Archivos implicados

En este paso deberia tocarse principalmente:

- `src/langgraph_beam_stanza.py`

Solo se deberia tocar `src/sonnet_metrics.py` si aparece una necesidad estricta de reutilizacion que no este cubierta por las funciones existentes.

En principio, NO hace falta modificar `sonnet_metrics.py`, porque ya existen:

- `diagnose_stanza_outer_rhyme(...)`;
- `count_verse_syllables(...)`;
- `TARGET_SYLLABLES_PER_VERSE`;
- `evaluate_stanza_abba(...)`.

## Funciones existentes que se van a reutilizar

En `src/langgraph_beam_stanza.py`:

- `chat_ollama(...)`
  - Para pedir variantes del verso 4 al modelo local.

- `clean_generated_verse(...)`
  - Para limpiar cada variante generada.

- `format_numbered_verses(...)`
  - Para incluir la estrofa completa como contexto en el prompt.

- `summarize_outer_rhyme_diagnosis(...)`
  - Para registrar resumenes legibles.

- `summarize_meter_repair_report(...)`
  - No se modifica, pero conviene mantener su informacion en trazas.

En `src/sonnet_metrics.py`:

- `diagnose_stanza_outer_rhyme(...)`
  - Para saber si la pareja A falla y para obtener la rima objetivo.

- `diagnose_stanza_inner_rhyme(...)`
  - Para recalcular el diagnostico interior despues de la reparacion exterior.

- `count_verse_syllables(...)`
  - Para comprobar la metrica del verso 4 original y de las variantes.

- `evaluate_stanza_abba(...)`
  - Para recalcular la evaluacion formal final de la estrofa tras la posible reparacion.

## Nuevas constantes recomendadas

Anadir en la zona de constantes de `src/langgraph_beam_stanza.py`:

```python
ENABLE_OUTER_RHYME_REPAIR = True
OUTER_RHYME_REPAIR_VARIANTS = 5
OUTER_RHYME_REPAIR_TEMPERATURE = 0.3
OUTER_RHYME_REPAIR_NUM_PREDICT = 200
```

Motivo:

- `ENABLE_OUTER_RHYME_REPAIR` permite activar o desactivar experimentalmente este nuevo refinamiento.
- `OUTER_RHYME_REPAIR_VARIANTS` controla cuantas variantes del verso 4 se piden.
- `OUTER_RHYME_REPAIR_TEMPERATURE` mantiene el refinamiento relativamente conservador.
- `OUTER_RHYME_REPAIR_NUM_PREDICT` limita la longitud de la respuesta.

No modificar en este paso las constantes de reparacion metrica.

## Nueva estructura de reporte

Cada candidato deberia guardar un reporte de reparacion de rima exterior:

```python
"outer_rhyme_repair_report": {
    "enabled": True,
    "attempted": True,
    "changed": False,
    "reason": "...",
    "original_stanza": [...],
    "repaired_stanza": [...],
    "target_rhyme": "...",
    "current_rhyme": "...",
    "verse_1": {...},
    "original_verse_4": "...",
    "original_verse_4_syllables": 10,
    "selected_variant": None,
    "selected_variant_syllables": None,
    "variants": [
        {
            "verse": "...",
            "syllables": 11,
            "outer_rhyme_valid": True,
            "outer_rhyme_summary": "...",
            "accepted": False,
            "rejection_reason": "..."
        }
    ],
    "error": None
}
```

Este reporte debe guardarse en:

- el beam/candidato puntuado;
- la traza;
- el JSON final.

Tambien debe aparecer un resumen en el `.txt` final.

## Nuevas funciones a implementar en `langgraph_beam_stanza.py`

### 1. `parse_outer_rhyme_repair_variants_response`

Crear:

```python
def parse_outer_rhyme_repair_variants_response(
    raw_response: str,
    max_variants: int,
) -> list[str]:
    ...
```

Debe parsear una respuesta JSON con este formato:

```json
{
  "variants": [
    "variante del verso 4",
    "otra variante del verso 4"
  ]
}
```

Comportamiento:

- usar `json.loads`;
- exigir que la respuesta sea un objeto JSON;
- exigir que exista una clave `"variants"` de tipo lista;
- limpiar cada variante con `clean_generated_verse`;
- descartar variantes vacias;
- descartar variantes duplicadas;
- devolver como maximo `max_variants`;
- si no hay variantes validas, lanzar `ValueError`.

Esta funcion debe ser equivalente en estilo a `parse_meter_repair_variants_response`, pero aplicada a rima exterior.

### 2. `_build_outer_rhyme_repair_messages`

Crear:

```python
def _build_outer_rhyme_repair_messages(
    question: str,
    stanza: list[str],
    outer_rhyme_diagnosis: dict[str, Any],
    num_variants: int,
) -> List[Dict[str, str]]:
    ...
```

Debe construir el prompt para pedir variantes del verso 4.

El prompt debe incluir:

- la tarea original;
- la estrofa completa numerada;
- el verso 1 como ancla;
- la palabra final del verso 1;
- la rima consonante objetivo del verso 1;
- el verso 4 actual;
- la rima actual del verso 4;
- la instruccion de reescribir solo el verso 4;
- la instruccion de intentar mantener 11 silabas metricas;
- el formato JSON obligatorio.

Restricciones que deben aparecer claramente en el prompt:

```text
- Reescribe solo el verso 4.
- No modifiques el verso 1.
- No modifiques los versos 2 y 3.
- El nuevo verso 4 debe rimar consonantemente con el verso 1.
- La rima objetivo es: ...
- Intenta que el nuevo verso 4 tenga 11 silabas metricas.
- Devuelve solo variantes del verso 4, no la estrofa completa.
- No anadas titulo.
- No numeres las variantes.
- Responde solo con JSON valido.
```

Formato JSON esperado:

```json
{
  "variants": [
    "nuevo verso 4",
    "nuevo verso 4 alternativo"
  ]
}
```

### 3. `generate_outer_rhyme_repair_variants_with_ollama`

Crear:

```python
def generate_outer_rhyme_repair_variants_with_ollama(
    question: str,
    stanza: list[str],
    outer_rhyme_diagnosis: dict[str, Any],
    num_variants: int = OUTER_RHYME_REPAIR_VARIANTS,
) -> list[str]:
    ...
```

Debe:

1. Construir mensajes con `_build_outer_rhyme_repair_messages`.
2. Llamar a `chat_ollama`.
3. Usar:

```python
model=GENERATION_MODEL
temperature=OUTER_RHYME_REPAIR_TEMPERATURE
num_predict=OUTER_RHYME_REPAIR_NUM_PREDICT
```

4. Parsear con `parse_outer_rhyme_repair_variants_response`.
5. Devolver lista de variantes limpias.

### 4. `build_outer_rhyme_repair_report`

Crear:

```python
def build_outer_rhyme_repair_report(
    enabled: bool,
    stanza: list[str],
) -> dict[str, Any]:
    ...
```

Debe devolver la estructura base del reporte.

Campos minimos:

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
    "verse_1": None,
    "original_verse_4": None,
    "original_verse_4_syllables": None,
    "selected_variant": None,
    "selected_variant_syllables": None,
    "variants": [],
    "error": None,
}
```

### 5. `repair_stanza_outer_rhyme_with_ollama`

Crear:

```python
def repair_stanza_outer_rhyme_with_ollama(
    question: str,
    stanza: list[str],
    outer_rhyme_diagnosis: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    ...
```

Esta sera la funcion principal del Paso 3.

Comportamiento obligatorio:

1. Crear reporte con `build_outer_rhyme_repair_report`.
2. Si `ENABLE_OUTER_RHYME_REPAIR` es `False`, devolver la estrofa sin cambios.
3. Si la estrofa no tiene 4 versos, devolver la estrofa sin cambios.
4. Si `outer_rhyme_diagnosis["is_valid"]` es `True`, devolver la estrofa sin cambios.
5. Si no hay `target_rhyme`, devolver la estrofa sin cambios.
6. Si se puede reparar, marcar `attempted = True`.
7. Pedir variantes del verso 4 con `generate_outer_rhyme_repair_variants_with_ollama`.
8. Para cada variante:
   - crear una copia de la estrofa;
   - sustituir solo el verso 4;
   - calcular `diagnose_stanza_outer_rhyme` sobre la copia;
   - contar silabas del nuevo verso 4 con `count_verse_syllables`;
   - registrar la variante en el reporte.
9. Aceptar una variante solo si:
   - la nueva rima exterior AXYA es valida;
   - el nuevo verso 4 no empeora la distancia metrica al objetivo de 11 silabas.
10. Si varias variantes cumplen, elegir preferentemente:
   - primero, una variante con 11 silabas;
   - despues, la de menor distancia a 11 silabas;
   - si hay empate, la primera recibida.
11. Si se acepta una variante:
   - sustituir el verso 4;
   - marcar `changed = True`;
   - guardar `selected_variant`;
   - guardar `selected_variant_syllables`.
12. Si ninguna variante cumple, conservar la estrofa original.
13. Devolver:

```python
return repaired_stanza, report
```

### 6. `summarize_outer_rhyme_repair_report`

Crear:

```python
def summarize_outer_rhyme_repair_report(report: dict[str, Any]) -> str:
    ...
```

Debe devolver frases compactas como:

```text
reparacion rima A = desactivada
```

```text
reparacion rima A = no necesaria
```

```text
reparacion rima A = activa | intento = True | cambio = True
```

```text
reparacion rima A = activa | intento = True | cambio = False
```

## Criterio de aceptacion de variantes

La reparacion de rima exterior no debe aceptar cualquier verso que rime.

Debe comprobar:

```python
original_verse_4_syllables = count_verse_syllables(original_verse_4)
variant_syllables = count_verse_syllables(variant)
```

Y comparar distancia al objetivo:

```python
original_distance = abs(original_verse_4_syllables - TARGET_SYLLABLES_PER_VERSE)
variant_distance = abs(variant_syllables - TARGET_SYLLABLES_PER_VERSE)
```

Una variante solo puede aceptarse si:

```python
outer_rhyme_valid is True
variant_distance <= original_distance
```

Motivo:

- este paso arregla rima exterior;
- pero no debe empeorar la metrica del verso 4;
- si puede dejar el verso 4 en 11 silabas, mejor.

## Integracion en `score_node`

Actualmente el flujo en `score_node` es:

```python
stanza, meter_repair_report = repair_stanza_meter_with_ollama(...)
evaluation = evaluate_stanza_abba(stanza)
outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
```

Tras este paso, el flujo deberia ser:

```python
stanza, meter_repair_report = repair_stanza_meter_with_ollama(...)

outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
stanza, outer_rhyme_repair_report = repair_stanza_outer_rhyme_with_ollama(
    question=state["question"],
    stanza=stanza,
    outer_rhyme_diagnosis=outer_rhyme_diagnosis,
)

evaluation = evaluate_stanza_abba(stanza)
outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
```

Notas importantes:

- Despues de la posible reparacion de rima, hay que recalcular `evaluation`.
- Despues de la posible reparacion de rima, hay que recalcular ambos diagnosticos.
- No hay que llamar de nuevo a `repair_stanza_meter_with_ollama` en este paso.
- La comprobacion metrica posterior se hace evaluando el verso 4 y la estrofa final, no ejecutando otra reparacion metrica completa.

## Cambios en candidatos y beams

Cada candidato puntuado debe incluir:

```python
"outer_rhyme_repair_report": outer_rhyme_repair_report,
```

Tambien debe inicializarse en candidatos generados:

```python
"outer_rhyme_repair_report": {},
```

Y en el estado inicial:

```python
"outer_rhyme_repair_report": {},
```

## Cambios en la traza

Actualizar `build_trace_entry(...)` para incluir:

```python
"outer_rhyme_repair_summary": summarize_outer_rhyme_repair_report(outer_rhyme_repair_report),
"outer_rhyme_repair_report": outer_rhyme_repair_report,
```

## Cambios en salida por terminal

En `score_node(...)` y/o `prune_node(...)`, imprimir:

```python
print(
    "Reparacion rima A: "
    f"{summarize_outer_rhyme_repair_report(outer_rhyme_repair_report)}"
)
```

Esto debe aparecer junto a:

- resumen de metricas;
- reparacion metrica;
- rima exterior AXYA;
- rima interior B.

## Cambios en exportacion

En `save_final_result(...)`, guardar el reporte del mejor beam en el JSON final:

```python
"outer_rhyme_repair_report": outer_rhyme_repair_report,
```

En el `.txt` final, anadir una linea resumen:

```text
Resumen reparacion rima A: reparacion rima A = activa | intento = True | cambio = True
```

Tambien conviene anadir estos parametros al bloque `run_parameters`:

```python
"enable_outer_rhyme_repair": ENABLE_OUTER_RHYME_REPAIR,
"outer_rhyme_repair_variants": OUTER_RHYME_REPAIR_VARIANTS,
"outer_rhyme_repair_temperature": OUTER_RHYME_REPAIR_TEMPERATURE,
"outer_rhyme_repair_num_predict": OUTER_RHYME_REPAIR_NUM_PREDICT,
```

## Pruebas minimas recomendadas

### 1. Prueba de parseo

Probar que:

```python
parse_outer_rhyme_repair_variants_response(
    '{"variants": ["Nuevo verso cuarto", "Otro verso cuarto"]}',
    5,
)
```

devuelve una lista de variantes limpias.

### 2. Prueba sin reparacion necesaria

Con una estrofa que ya cumple AXYA:

- `repair_stanza_outer_rhyme_with_ollama` debe devolver la misma estrofa;
- `attempted` debe ser `False`;
- `changed` debe ser `False`.

### 3. Prueba con estrofa de menos de 4 versos

Debe devolver la estrofa sin cambios y registrar el motivo.

### 4. Prueba real con Ollama

Ejecutar el flujo completo y observar:

- si se intento reparar la rima A;
- cuantas variantes se probaron;
- si alguna variante fue aceptada;
- si la rima exterior AXYA paso a correcta;
- si la metrica del verso 4 se mantuvo o mejoro.

## Criterio de exito del Paso 3

El Paso 3 estara completo cuando:

1. Exista una funcion que repare la rima exterior A reescribiendo solo el verso 4.
2. La reparacion solo se intente cuando `outer_rhyme_diagnosis["is_valid"]` sea `False`.
3. El LLM reciba un prompt especifico con la rima objetivo del verso 1.
4. El sistema evalue las variantes antes de aceptar ninguna.
5. Se acepte una variante solo si arregla AXYA y no empeora la metrica del verso 4.
6. Se conserven sin cambios los versos 1, 2 y 3.
7. Tras la reparacion, se recalculen:
   - `evaluate_stanza_abba`;
   - `diagnose_stanza_outer_rhyme`;
   - `diagnose_stanza_inner_rhyme`.
8. La traza guarde el reporte de reparacion de rima A.
9. El JSON final incluya el reporte.
10. El `.txt` final incluya un resumen de la reparacion de rima A.

## Que NO hacer en este paso

No implementar todavia:

- reparacion de la rima interior B;
- reescritura del verso 3;
- regeneracion completa de la estrofa para reparar rima;
- cambios en los pesos de scoring;
- cambios en el prompt principal de generacion de estrofas;
- cambios en la reparacion metrica local;
- segunda pasada automatica de reparacion metrica despues de reparar rima;
- evaluacion subjetiva de calidad poetica.

Este paso debe limitarse a una cosa:

```text
Si falla AXYA, pedir variantes del verso 4 y aceptar solo una variante que arregle la rima exterior sin empeorar la metrica del verso 4.
```

## Relacion con el Paso 4

Si este Paso 3 funciona, el siguiente paso natural sera observar resultados reales y decidir si:

- repetir el mismo patron para la rima interior B, reescribiendo solo el verso 3;
- o ajustar primero el criterio de aceptacion de variantes de rima exterior.

No se debe implementar la rima interior B hasta haber probado este refinamiento exterior A con varias ejecuciones.
