# Paso 1: diagnostico explicito de la rima AXYA

## Objetivo del paso

El objetivo de este primer paso es preparar el sistema para trabajar la rima de forma progresiva, empezando solo por la pareja exterior de la estrofa:

```text
Verso 1 -> A
Verso 2 -> X
Verso 3 -> Y
Verso 4 -> A
```

En este paso NO se va a reparar todavia la rima con el LLM. Tampoco se va a intentar obtener una estrofa ABBA completa.

Lo que hay que implementar es un diagnostico claro de la rima AXYA:

- extraer la palabra final del verso 1;
- extraer la palabra final del verso 4;
- extraer la rima consonante del verso 1;
- extraer la rima consonante del verso 4;
- comprobar si ambas rimas coinciden;
- generar un mensaje de diagnostico util para el siguiente paso.

Este diagnostico sera la base para que, en el Paso 2, el prompt pueda decir algo concreto como:

```text
El verso 1 termina en "atras", cuya rima consonante es "as".
Reescribe el verso 4 para que termine tambien con rima consonante "as".
```

## Por que este paso va antes de reparar ABBA

Atacar directamente ABBA obliga al modelo a resolver dos parejas de rima a la vez:

```text
1 con 4
2 con 3
```

Eso es demasiado amplio para depurar de forma limpia.

Por recomendacion del tutor, se separa el problema:

1. Primero se busca AXYA: que verso 1 y verso 4 rimen.
2. Despues se volvera a revisar la metrica.
3. Solo cuando la pareja A este controlada, se pasara a la pareja B: verso 2 con verso 3.

## Archivos implicados

En este paso deberian tocarse principalmente estos archivos:

- `src/sonnet_metrics.py`
- `src/langgraph_beam_stanza.py`

No deberia tocarse todavia:

- la reparacion metrica local, salvo para leer sus resultados;
- la generacion principal de candidatos;
- la logica de reparacion de rima con LLM, porque aun no se implementa en este paso.

## Funciones existentes que se van a reutilizar

En `src/sonnet_metrics.py` ya existen funciones utiles:

- `get_last_word(verse: str) -> str | None`
  - Obtiene la ultima palabra util de un verso.

- `extract_verse_rhyme(verse: str) -> str | None`
  - Extrae la rima consonante de la ultima palabra del verso.

- `extract_consonant_rhyme(word: str) -> str`
  - Extrae la rima consonante de una palabra concreta.

- `extract_rhymes(verses: list[str]) -> list[str | None]`
  - Extrae la rima de varios versos.

- `evaluate_stanza_abba(stanza: str | list[str]) -> dict[str, object]`
  - Evalua la estrofa completa como ABBA. Se mantiene como evaluador final, pero no basta para diagnosticar la pareja A de forma explicita.

## Nueva funcionalidad a implementar

### 1. Nueva funcion en `sonnet_metrics.py`

Crear una funcion especifica para diagnosticar la pareja A:

```python
def diagnose_stanza_outer_rhyme(stanza: str | list[str]) -> dict[str, object]:
    ...
```

Nombre recomendado: `diagnose_stanza_outer_rhyme`.

Motivo del nombre:

- "stanza" porque trabaja sobre una estrofa;
- "outer" porque analiza los versos exteriores, 1 y 4;
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
2. Comprobar si hay al menos 4 versos.
3. Tomar el verso 1 y el verso 4.
4. Extraer la ultima palabra de ambos versos.
5. Extraer la rima consonante de ambos versos.
6. Comparar las dos rimas.
7. Devolver un diccionario con diagnostico estructurado.

### 4. Estructura recomendada del diccionario

La funcion deberia devolver algo de este estilo:

```python
{
    "target_pattern": "AXYA",
    "required_pair": [1, 4],
    "has_enough_verses": True,
    "is_valid": False,
    "verse_1": {
        "text": "El tiempo se va, y yo me quedo atras",
        "last_word": "atras",
        "rhyme": "as",
    },
    "verse_4": {
        "text": "Que nunca se apaga, como una llama ardiente",
        "last_word": "ardiente",
        "rhyme": "ente",
    },
    "target_rhyme": "as",
    "current_rhyme": "ente",
    "errors": [
        "El verso 4 no rima consonantemente con el verso 1: se esperaba la rima 'as', pero se obtuvo 'ente'."
    ],
    "feedback": "Para cumplir AXYA, conserva la rima del verso 1 ('as') y reescribe el verso 4 para que termine con esa misma rima consonante."
}
```

Notas importantes:

- `target_rhyme` debe ser la rima del verso 1.
- `current_rhyme` debe ser la rima del verso 4.
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

- verso 1 y verso 4 riman;
- `is_valid == True`;
- `target_rhyme == current_rhyme`;
- `errors == []`.

### Caso incorrecto

Entrada:

```python
[
    "El tiempo se va, y yo me quedo atras",
    "En recuerdos que me impiden dormir",
    "Un llanto persiste en mi corazon",
    "Que nunca se apaga, como una llama ardiente",
]
```

Resultado esperado:

- verso 1 y verso 4 no riman;
- `is_valid == False`;
- `target_rhyme` corresponde al verso 1;
- `current_rhyme` corresponde al verso 4;
- el feedback debe indicar que el verso 4 debe reescribirse buscando la rima del verso 1.

### Caso con menos de 4 versos

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
- debe incluirse un error indicando que se necesitan al menos 4 versos para diagnosticar AXYA.

## Integracion en `langgraph_beam_stanza.py`

Una vez creada la funcion en `sonnet_metrics.py`, hay que importarla en `langgraph_beam_stanza.py`:

```python
from sonnet_metrics import (
    TARGET_SYLLABLES_PER_VERSE,
    count_verse_syllables,
    diagnose_stanza_outer_rhyme,
    evaluate_stanza_abba,
)
```

## Donde usar el diagnostico en el flujo actual

En este Paso 1, el diagnostico debe usarse solo para trazabilidad y observacion.

Lugar recomendado:

```python
score_node(...)
```

Justo despues de:

```python
evaluation = evaluate_stanza_abba(stanza)
```

se deberia calcular:

```python
outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
```

Este diagnostico deberia guardarse en el candidato puntuado:

```python
scored_candidate = {
    ...
    "outer_rhyme_diagnosis": outer_rhyme_diagnosis,
}
```

## Cambios en la traza

Actualizar `build_trace_entry(...)` para incluir el diagnostico:

```python
"outer_rhyme_diagnosis": outer_rhyme_diagnosis,
```

Tambien puede añadirse un resumen breve:

```python
"outer_rhyme_summary": summarize_outer_rhyme_diagnosis(outer_rhyme_diagnosis),
```

## Nueva funcion resumen recomendada

En `langgraph_beam_stanza.py`, crear:

```python
def summarize_outer_rhyme_diagnosis(diagnosis: dict[str, Any]) -> str:
    ...
```

Debe devolver frases compactas como:

```text
rima AXYA = correcta | v1='as' | v4='as'
```

o:

```text
rima AXYA = incorrecta | objetivo='as' | v4='ente'
```

o:

```text
rima AXYA = no evaluable | faltan versos
```

## Cambios en salida por terminal

En `score_node(...)` o `prune_node(...)`, imprimir un resumen compacto:

```python
print(f"Rima exterior AXYA: {summarize_outer_rhyme_diagnosis(outer_rhyme_diagnosis)}")
```

Esto servira para ver rapidamente si la pareja 1-4 ya esta controlada.

## Cambios en exportacion

En `save_final_result(...)`, guardar el diagnostico del mejor beam en el JSON final:

```python
"outer_rhyme_diagnosis": outer_rhyme_diagnosis,
```

En el `.txt` final, añadir una linea resumen:

```text
Resumen rima exterior AXYA: rima AXYA = incorrecta | objetivo='as' | v4='ente'
```

## Criterio de exito del Paso 1

El Paso 1 estara completo cuando:

1. Exista una funcion en `sonnet_metrics.py` que diagnostique si el verso 1 y el verso 4 riman.
2. El diagnostico indique claramente:
   - palabra final del verso 1;
   - palabra final del verso 4;
   - rima del verso 1;
   - rima del verso 4;
   - si la pareja A esta correcta o no.
3. `langgraph_beam_stanza.py` calcule ese diagnostico para cada candidato puntuado.
4. La traza guarde el diagnostico.
5. La salida por terminal muestre un resumen breve de AXYA.
6. El JSON final incluya el diagnostico del mejor beam.

## Que NO hacer en este paso

No implementar todavia:

- reparacion de rima con LLM;
- prompts para reescribir el verso 4;
- reparacion de la pareja B;
- evaluacion completa AXYA como sustituto de ABBA;
- cambios en los pesos de scoring;
- cambios en `ENABLE_LOCAL_METER_REPAIR`;
- cambios grandes de arquitectura.

Este paso es solo de diagnostico y trazabilidad.

## Relacion con el Paso 2

El Paso 2 usara el diagnostico creado aqui para enriquecer el prompt de reparacion de rima.

Ejemplo de uso futuro:

```text
La rima objetivo del verso 1 es "as".
El verso 4 actual tiene rima "ente".
Reescribe solo el verso 4 para que mantenga 11 silabas y termine con rima consonante "as".
```

Por eso es importante que este Paso 1 quede bien estructurado: el reparador de rima no deberia tener que volver a calcular ni inferir estos datos desde cero.
