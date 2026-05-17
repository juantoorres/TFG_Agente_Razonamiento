# Paso 2: diagnostico explicito de la rima interior B

## Objetivo del paso

El objetivo de este segundo paso es completar el diagnostico progresivo de la rima ABBA, analizando ahora solo la pareja interior de la estrofa:

```text
Verso 1 -> X
Verso 2 -> B
Verso 3 -> B
Verso 4 -> Y
```

En el Paso 1 se diagnostico la pareja exterior:

```text
Verso 1 -> A
Verso 4 -> A
```

En este Paso 2 se debe diagnosticar la pareja interior:

```text
Verso 2 -> B
Verso 3 -> B
```

En este paso NO se va a reparar todavia la rima con el LLM. Tampoco se va a reescribir ningun verso.

Lo que hay que implementar es un diagnostico claro de la rima interior B:

- extraer la palabra final del verso 2;
- extraer la palabra final del verso 3;
- extraer la rima consonante del verso 2;
- extraer la rima consonante del verso 3;
- comprobar si ambas rimas coinciden;
- generar un mensaje de diagnostico util para el futuro refinamiento de rima.

Este diagnostico sera la base para que, en un paso posterior, el prompt pueda decir algo concreto como:

```text
El verso 2 termina en "recuerdo", cuya rima consonante es "erdo".
Reescribe el verso 3 para que termine tambien con rima consonante "erdo".
```

## Por que este paso va antes del refinamiento de rima

Ya existe diagnostico AXYA para la pareja exterior. Antes de implementar reparacion de rima, conviene saber si el sistema puede separar claramente los dos problemas:

```text
Pareja A: verso 1 con verso 4
Pareja B: verso 2 con verso 3
```

La salida experimental tras el Paso 1 mostro una situacion muy util:

```text
Rima exterior AXYA correcta
Rima ABBA global incompleta porque falla el verso 3
```

Eso indica que necesitamos diagnosticar tambien la pareja B de forma explicita, no solo depender del resumen global `rima=3/4`.

Este paso debe seguir la misma filosofia que el Paso 1:

1. Primero diagnosticar.
2. Despues observar resultados reales.
3. Solo entonces implementar refinamiento con LLM.

## Archivos implicados

En este paso deberian tocarse principalmente estos archivos:

- `src/sonnet_metrics.py`
- `src/langgraph_beam_stanza.py`

No deberia tocarse todavia:

- la reparacion metrica local;
- los prompts de generacion principal;
- la logica de reparacion de rima con LLM;
- los pesos del scoring;
- el Beam Search;
- la exportacion ya existente, salvo para anadir el nuevo diagnostico.

## Funciones existentes que se van a reutilizar

En `src/sonnet_metrics.py` ya existen funciones utiles:

- `normalize_verses_input(stanza: str | list[str]) -> list[str]`
  - Normaliza la entrada a una lista de versos sin lineas vacias.

- `get_last_word(verse: str) -> str | None`
  - Obtiene la ultima palabra util de un verso.

- `extract_verse_rhyme(verse: str) -> str | None`
  - Extrae la rima consonante de la ultima palabra del verso.

- `diagnose_stanza_outer_rhyme(stanza: str | list[str]) -> dict[str, object]`
  - Diagnostica la pareja exterior A, versos 1 y 4.
  - Debe mantenerse sin cambios salvo que sea estrictamente necesario.

- `evaluate_stanza_abba(stanza: str | list[str]) -> dict[str, object]`
  - Evalua la estrofa completa como ABBA. Se mantiene como evaluador final.

## Nueva funcionalidad a implementar

### 1. Nueva funcion en `sonnet_metrics.py`

Crear una funcion especifica para diagnosticar la pareja B:

```python
def diagnose_stanza_inner_rhyme(stanza: str | list[str]) -> dict[str, object]:
    ...
```

Nombre recomendado: `diagnose_stanza_inner_rhyme`.

Motivo del nombre:

- "stanza" porque trabaja sobre una estrofa;
- "inner" porque analiza los versos interiores, 2 y 3;
- "rhyme" porque solo diagnostica rima.

### 2. Entrada de la funcion

La funcion debe aceptar:

```python
stanza: str | list[str]
```

Debe usar internamente:

```python
verses = normalize_verses_input(stanza)
```

Asi se mantiene el mismo comportamiento que el resto del modulo.

### 3. Comportamiento esperado

La funcion debe:

1. Normalizar la entrada a lista de versos.
2. Comprobar si hay al menos 3 versos.
3. Tomar el verso 2 y el verso 3.
4. Extraer la ultima palabra de ambos versos.
5. Extraer la rima consonante de ambos versos.
6. Comparar las dos rimas.
7. Devolver un diccionario con diagnostico estructurado.

Nota importante:

- Para diagnosticar la pareja B solo hacen falta al menos 3 versos.
- Aun asi, en el flujo normal se espera trabajar con estrofas de 4 versos.

### 4. Estructura recomendada del diccionario

La funcion deberia devolver algo de este estilo:

```python
{
    "target_pattern": "XBBY",
    "required_pair": [2, 3],
    "has_enough_verses": True,
    "is_valid": False,
    "verse_2": {
        "text": "Imagen clara en la memoria recuerdo",
        "last_word": "recuerdo",
        "rhyme": "erdo",
    },
    "verse_3": {
        "text": "Rio triste lleva corazon el tiempo",
        "last_word": "tiempo",
        "rhyme": "empo",
    },
    "target_rhyme": "erdo",
    "current_rhyme": "empo",
    "errors": [
        "El verso 3 no rima consonantemente con el verso 2: se esperaba la rima 'erdo', pero se obtuvo 'empo'."
    ],
    "feedback": "Para cumplir la pareja B de ABBA, conserva la rima del verso 2 ('erdo') y reescribe el verso 3 para que termine con esa misma rima consonante."
}
```

Notas importantes:

- `target_rhyme` debe ser la rima del verso 2.
- `current_rhyme` debe ser la rima del verso 3.
- `is_valid` debe ser `True` solo si ambas rimas existen y coinciden.
- Si faltan versos, `has_enough_verses` debe ser `False`.
- Si no se puede extraer alguna rima, debe aparecer en `errors`.

## Casos que debe contemplar

### Caso correcto

Entrada:

```python
[
    "Late la tarde sobre mi memoria",
    "Cruza la sombra del jardin dormido",
    "Calla en la fuente su rumor perdido",
    "Vuelve la luz antigua de mi historia",
]
```

Resultado esperado:

- verso 2 y verso 3 riman;
- `is_valid == True`;
- `target_rhyme == current_rhyme`;
- `errors == []`.

### Caso incorrecto

Entrada:

```python
[
    "Vuelan anos, dejan huella en el alma",
    "Imagen clara en la memoria recuerdo",
    "Rio triste lleva corazon el tiempo",
    "Pasa el tiempo, dejandonos sin alma",
]
```

Resultado esperado:

- verso 2 y verso 3 no riman;
- `is_valid == False`;
- `target_rhyme` corresponde al verso 2;
- `current_rhyme` corresponde al verso 3;
- el feedback debe indicar que el verso 3 debe reescribirse buscando la rima del verso 2.

### Caso con menos de 3 versos

Entrada:

```python
[
    "Primer verso",
    "Segundo verso",
]
```

Resultado esperado:

- `has_enough_verses == False`;
- `is_valid == False`;
- debe incluirse un error indicando que se necesitan al menos 3 versos para diagnosticar la rima interior B.

## Integracion en `langgraph_beam_stanza.py`

Una vez creada la funcion en `sonnet_metrics.py`, hay que importarla en `langgraph_beam_stanza.py`:

```python
from sonnet_metrics import (
    TARGET_SYLLABLES_PER_VERSE,
    count_verse_syllables,
    diagnose_stanza_inner_rhyme,
    diagnose_stanza_outer_rhyme,
    evaluate_stanza_abba,
)
```

## Donde usar el diagnostico en el flujo actual

En este Paso 2, el diagnostico debe usarse solo para trazabilidad y observacion.

Lugar recomendado:

```python
score_node(...)
```

Justo despues de:

```python
evaluation = evaluate_stanza_abba(stanza)
outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
```

se deberia calcular:

```python
inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
```

Este diagnostico deberia guardarse en el candidato puntuado:

```python
scored_candidate = {
    ...
    "inner_rhyme_diagnosis": inner_rhyme_diagnosis,
}
```

## Cambios en la traza

Actualizar `build_trace_entry(...)` para incluir el diagnostico:

```python
"inner_rhyme_diagnosis": inner_rhyme_diagnosis,
```

Tambien debe anadirse un resumen breve:

```python
"inner_rhyme_summary": summarize_inner_rhyme_diagnosis(inner_rhyme_diagnosis),
```

## Nueva funcion resumen recomendada

En `langgraph_beam_stanza.py`, crear:

```python
def summarize_inner_rhyme_diagnosis(diagnosis: dict[str, Any]) -> str:
    ...
```

Debe devolver frases compactas como:

```text
rima interior B = correcta | v2='ido' | v3='ido'
```

o:

```text
rima interior B = incorrecta | objetivo='erdo' | v3='empo'
```

o:

```text
rima interior B = no evaluable | faltan versos
```

## Cambios en salida por terminal

En `score_node(...)` o `prune_node(...)`, imprimir un resumen compacto:

```python
print(f"Rima interior B: {summarize_inner_rhyme_diagnosis(inner_rhyme_diagnosis)}")
```

Esto servira para ver rapidamente si la pareja 2-3 ya esta controlada.

## Cambios en exportacion

En `save_final_result(...)`, guardar el diagnostico del mejor beam en el JSON final:

```python
"inner_rhyme_diagnosis": inner_rhyme_diagnosis,
```

En el `.txt` final, anadir una linea resumen:

```text
Resumen rima interior B: rima interior B = incorrecta | objetivo='erdo' | v3='empo'
```

## Criterio de exito del Paso 2

El Paso 2 estara completo cuando:

1. Exista una funcion en `sonnet_metrics.py` que diagnostique si el verso 2 y el verso 3 riman.
2. El diagnostico indique claramente:
   - palabra final del verso 2;
   - palabra final del verso 3;
   - rima del verso 2;
   - rima del verso 3;
   - si la pareja B esta correcta o no.
3. `langgraph_beam_stanza.py` calcule ese diagnostico para cada candidato puntuado.
4. La traza guarde el diagnostico.
5. La salida por terminal muestre un resumen breve de la rima interior B.
6. El JSON final incluya el diagnostico del mejor beam.
7. La salida `.txt` final incluya un resumen de la rima interior B.

## Que NO hacer en este paso

No implementar todavia:

- reparacion de rima con LLM;
- prompts para reescribir el verso 3;
- prompts para reescribir el verso 4;
- reparacion automatica de la pareja A;
- reparacion automatica de la pareja B;
- cambios en la reparacion metrica local;
- evaluacion completa AXYA o XBBY como sustituto de ABBA;
- cambios en los pesos de scoring;
- cambios grandes de arquitectura.

Este paso es solo de diagnostico y trazabilidad.

## Relacion con el siguiente paso

Tras este Paso 2, el sistema tendra dos diagnosticos separados:

```text
outer_rhyme_diagnosis -> pareja A, versos 1 y 4
inner_rhyme_diagnosis -> pareja B, versos 2 y 3
```

Con ambos diagnosticos sera posible decidir, en un paso posterior, que reparar primero:

- si falla la pareja A, reescribir solo el verso 4 para rimar con el verso 1;
- si falla la pareja B, reescribir solo el verso 3 para rimar con el verso 2;
- despues de cualquier reparacion de rima, volver a comprobar la metrica.

Ese refinamiento todavia no se implementa aqui. Este paso solo deja preparado el sistema para saber exactamente que pareja de rima falla.
