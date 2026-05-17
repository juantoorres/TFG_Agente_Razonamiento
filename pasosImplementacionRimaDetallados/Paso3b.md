# Paso 3b: corregir la extraccion de rima consonante antes de seguir evaluando la reparacion A

## Objetivo del paso

El objetivo de este paso intermedio es corregir un problema detectado durante la prueba del Paso 3.

El refinamiento de rima exterior A no esta funcionando bien porque el sistema esta usando como `target_rhyme` una cadena demasiado larga. En el ejemplo observado:

```text
Verso 1: El tiempo, un rio que trae recuerdos amargos,
Ultima palabra: amargos
Rima detectada actualmente: amargos
```

Pero la rima consonante esperada no deberia ser la palabra completa `amargos`.

La rima consonante debe extraerse desde la vocal tonica hasta el final. Para `amargos`, la silaba tonica es `mar`, por lo que la rima deberia ser:

```text
argos
```

Este fallo provoca que el reparador de rima exterior A pida o valide cosas demasiado restrictivas. Por ejemplo, intenta conseguir una palabra que rime con `amargos` completo, cuando realmente deberia buscar palabras con rima `argos`, como `largos`.

Este paso NO debe cambiar el refinamiento de rima A todavia. Antes hay que corregir la base: la extraccion de rima consonante.

## Problema observado

Salida final observada:

```text
Resumen reparacion rima A: reparacion rima A = activa | intento = True | cambio = False
Resumen rima exterior AXYA: rima AXYA = incorrecta | objetivo = 'amargos' | v4 = 'anciano'
```

Fragmento relevante de la traza:

```json
{
  "target_rhyme": "amargos",
  "current_rhyme": "anciano",
  "verse_1": {
    "text": "El tiempo, un rio que trae recuerdos amargos,",
    "last_word": "amargos",
    "rhyme": "amargos"
  }
}
```

El problema es que:

```text
extract_consonant_rhyme("amargos") -> "amargos"
```

cuando deberia ser:

```text
extract_consonant_rhyme("amargos") -> "argos"
```

## Por que esto afecta al Paso 3

El Paso 3 usa `diagnose_stanza_outer_rhyme(...)` para obtener:

```python
target_rhyme = outer_rhyme_diagnosis["target_rhyme"]
```

Ese `target_rhyme` se pasa al prompt de reparacion de rima A:

```text
La rima objetivo es: amargos
```

Tambien se usa para aceptar o rechazar variantes:

```python
variant_diagnosis = diagnose_stanza_outer_rhyme(candidate_stanza)
outer_rhyme_valid = bool(variant_diagnosis.get("is_valid", False))
```

Si el diagnostico exige `amargos` en vez de `argos`, entonces variantes razonables pueden ser rechazadas.

Ejemplo:

```text
Verso 1 termina en: amargos
Verso 4 termina en: largos
```

Resultado deseado:

```text
amargos -> argos
largos  -> argos
AXYA correcta
```

Resultado actual probable:

```text
amargos -> amargos
largos  -> argos
AXYA incorrecta
```

## Archivos implicados

Este paso debe tocar principalmente:

- `src/sonnet_metrics.py`

Solo si es necesario para mostrar o resumir mejor la informacion, podria tocarse:

- `src/langgraph_beam_stanza.py`

Pero la correccion principal debe estar en el modulo de metrica y rima, no en el prompt.

## Funciones donde ocurre el problema

### 1. `extract_consonant_rhyme(word: str) -> str`

Esta funcion devuelve la rima consonante desde la vocal tonica hasta el final:

```python
def extract_consonant_rhyme(word: str) -> str:
    clean_word = clean_verse(word).replace(" ", "")
    stressed_vowel_position = get_last_stressed_vowel_position(clean_word)
    if stressed_vowel_position is None:
        return ""
    return strip_accents(clean_word[stressed_vowel_position:])
```

El error no esta necesariamente en esta funcion, sino en la posicion tonica que recibe.

### 2. `get_last_stressed_vowel_position(word: str) -> int | None`

Esta funcion decide que vocal es tonica.

Para palabras sin tilde, hace:

```python
syllables = split_word_syllables(clean_word)
stress_type = get_word_stress_type(clean_word)
...
vowel_offset = _find_stressed_vowel_offset_in_syllable(
    syllables[stressed_syllable_index]
)
```

El problema probable esta en que las silabas devueltas por `pyphen` no siempre se alinean con la estructura poetica esperada.

Caso observado:

```text
amargos
```

Si `pyphen` separa como:

```python
["amar", "gos"]
```

y la palabra es llana, la silaba tonica estimada sera `"amar"`. Despues `_find_stressed_vowel_offset_in_syllable("amar")` escoge la primera vocal fuerte de la silaba, es decir la `a` inicial.

Resultado:

```text
amargos -> amargos
```

Pero poeticamente/linguisticamente la division relevante seria:

```text
a-mar-gos
```

La vocal tonica esta en `mar`, por lo que la rima debe empezar en la segunda `a`:

```text
argos
```

### 3. `_find_stressed_vowel_offset_in_syllable(syllable: str) -> int | None`

Esta funcion actualmente:

1. busca la primera vocal fuerte;
2. si no encuentra, busca la ultima vocal.

Para una silaba aproximada como `"amar"`, eso devuelve la primera `a`, pero para rima convendria elegir la vocal nuclear mas cercana al final de esa silaba cuando `pyphen` agrupa demasiado.

Este es un punto candidato para el ajuste.

## Cambio recomendado

El cambio debe corregir la posicion de la vocal tonica en palabras sin tilde cuando `pyphen` agrupa una silaba demasiado larga.

La opcion mas simple y acotada es modificar:

```python
_find_stressed_vowel_offset_in_syllable(...)
```

para que, al localizar la vocal tonica aproximada dentro de una silaba sin tilde, no elija la primera vocal fuerte, sino la ultima vocal fuerte de la silaba. Si no hay vocal fuerte, debe mantener el comportamiento de buscar la ultima vocal.

Cambio conceptual:

```text
Antes:
    en "amar" devuelve la primera "a"

Despues:
    en "amar" devuelve la ultima vocal fuerte, la segunda "a"
```

Implementacion esperada:

```python
def _find_stressed_vowel_offset_in_syllable(syllable: str) -> int | None:
    for index in range(len(syllable) - 1, -1, -1):
        if syllable[index] in STRONG_VOWELS:
            return index

    for index in range(len(syllable) - 1, -1, -1):
        if syllable[index] in VOWELS:
            return index

    return None
```

## Salidas esperadas tras el cambio

### Palabras llanas sin tilde

```python
extract_consonant_rhyme("amargos") == "argos"
extract_consonant_rhyme("largos") == "argos"
extract_consonant_rhyme("amargo") == "argo"
extract_consonant_rhyme("largo") == "argo"
```

### Palabras con tilde escrita

Estas no deberian cambiar, porque `get_last_stressed_vowel_position` devuelve directamente la posicion de la vocal con tilde:

```python
extract_consonant_rhyme("corazón") == "on"
extract_consonant_rhyme("canción") == "on"
```

### Palabras llanas comunes ya usadas

Conviene comprobar que siguen siendo razonables:

```python
extract_consonant_rhyme("memoria") == "oria"
extract_consonant_rhyme("historia") == "oria"
extract_consonant_rhyme("vida") == "ida"
extract_consonant_rhyme("herida") == "ida"
```

### Diagnostico de rima exterior AXYA

Con esta estrofa:

```python
[
    "El tiempo trae recuerdos amargos",
    "Cruza la sombra del jardin dormido",
    "Calla en la fuente su rumor perdido",
    "La tarde deja silencios largos",
]
```

Resultado esperado:

```python
diagnose_stanza_outer_rhyme(stanza)["is_valid"] == True
diagnose_stanza_outer_rhyme(stanza)["target_rhyme"] == "argos"
diagnose_stanza_outer_rhyme(stanza)["current_rhyme"] == "argos"
```

### Diagnostico de rima interior B

Con esta estrofa:

```python
[
    "El tiempo trae recuerdos amargos",
    "Cruza la sombra del jardin dormido",
    "Calla en la fuente su rumor perdido",
    "La tarde deja silencios largos",
]
```

Resultado esperado:

```python
diagnose_stanza_inner_rhyme(stanza)["is_valid"] == True
diagnose_stanza_inner_rhyme(stanza)["target_rhyme"] == "ido"
diagnose_stanza_inner_rhyme(stanza)["current_rhyme"] == "ido"
```

## Relacion con el prompt de reparacion A

No se debe cambiar todavia `_build_outer_rhyme_repair_messages(...)`.

La razon es que el prompt ya recibe:

```python
target_rhyme = outer_rhyme_diagnosis.get("target_rhyme")
```

Si se corrige la extraccion de rima en `sonnet_metrics.py`, el prompt pasara automaticamente de:

```text
La rima objetivo es: amargos
```

a:

```text
La rima objetivo es: argos
```

Por tanto, el primer cambio debe estar en la extraccion de rima, no en el prompt.

## Pruebas minimas recomendadas

Ejecutar pruebas directas sin Ollama:

```powershell
$env:PYTHONPATH='.\src'
python -c "from sonnet_metrics import extract_consonant_rhyme; print(extract_consonant_rhyme('amargos'))"
```

Debe imprimir:

```text
argos
```

Probar tambien:

```powershell
$env:PYTHONPATH='.\src'
python -c "from sonnet_metrics import extract_consonant_rhyme; words=['amargos','largos','amargo','largo','corazón','canción','memoria','historia','vida','herida']; print({w: extract_consonant_rhyme(w) for w in words})"
```

Salida esperada aproximada:

```python
{
    "amargos": "argos",
    "largos": "argos",
    "amargo": "argo",
    "largo": "argo",
    "corazón": "on",
    "canción": "on",
    "memoria": "oria",
    "historia": "oria",
    "vida": "ida",
    "herida": "ida",
}
```

Probar diagnostico exterior:

```powershell
$env:PYTHONPATH='.\src'
python -c "from sonnet_metrics import diagnose_stanza_outer_rhyme; stanza=['El tiempo trae recuerdos amargos','Cruza la sombra del jardin dormido','Calla en la fuente su rumor perdido','La tarde deja silencios largos']; print(diagnose_stanza_outer_rhyme(stanza))"
```

Debe mostrar:

```text
is_valid: True
target_rhyme: argos
current_rhyme: argos
```

## Criterio de exito del Paso 3b

El Paso 3b estara completo cuando:

1. `extract_consonant_rhyme("amargos")` devuelva `"argos"`.
2. `extract_consonant_rhyme("largos")` devuelva `"argos"`.
3. `diagnose_stanza_outer_rhyme(...)` considere correcta una pareja `amargos` / `largos`.
4. `diagnose_stanza_inner_rhyme(...)` siga funcionando para parejas como `dormido` / `perdido`.
5. Las palabras con tilde escrita sigan funcionando como antes.
6. No se haya modificado todavia la logica de reparacion de rima A.
7. No se haya modificado la reparacion metrica.

## Que NO hacer en este paso

No implementar todavia:

- nuevos prompts para rima A;
- reparacion de la rima interior B;
- cambios en el criterio de aceptacion de variantes;
- cambios en Beam Search;
- cambios en pesos de scoring;
- cambios en reparacion metrica;
- heuristicas complejas de rima poetica avanzada.

Este paso es solo una correccion puntual de la extraccion de rima consonante para que los diagnosticos y el reparador A trabajen con una rima objetivo realista.
