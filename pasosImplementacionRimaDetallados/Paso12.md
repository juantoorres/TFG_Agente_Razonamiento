# Paso 12: hacer opcional el diccionario `RHYME_HINT_EXAMPLES`

## Objetivo del paso

El objetivo de este paso es permitir activar o desactivar el uso del diccionario interno:

```python
RHYME_HINT_EXAMPLES
```

mediante una constante global al inicio del programa.

La motivacion es poder ejecutar experimentos comparativos:

```text
- con diccionario de palabras candidatas;
- sin diccionario de palabras candidatas.
```

Este paso no debe eliminar el diccionario ni sustituirlo todavia por generacion dinamica con LLM. Solo debe hacer que su uso sea opcional.

## Motivacion

Actualmente, `RHYME_HINT_EXAMPLES` contiene palabras finales candidatas escritas en el codigo.

Ejemplo:

```python
"ar": ["llorar", "pasar", "mirar", "soñar", "callar", "recordar"]
"ado": ["helado", "olvidado", "callado", "pasado", "soñado", "amado"]
```

Estas palabras ayudan a reducir el espacio de busqueda en la reparacion de rima. Sin embargo, al estar hardcodeadas, pueden plantear dudas metodologicas:

- reducen la generalidad del sistema;
- pueden sesgar el vocabulario;
- pueden restar naturalidad o variedad poetica;
- no sirven igual de bien para cualquier rima nueva.

Por eso interesa poder desactivarlas facilmente y observar que ocurre.

## Alcance del paso

Este paso debe tocar principalmente:

- `src/langgraph_beam_stanza.py`

No debe tocar:

- `src/sonnet_metrics.py`;
- el contador de silabas;
- la extraccion de rima;
- `evaluate_stanza_abba(...)`;
- el Beam Search;
- el scoring;
- la reparacion metrica inicial;
- la reparacion metrica posterior a A;
- la reparacion metrica posterior a B;
- los prompts principales de generacion salvo que sea estrictamente necesario;
- `RHYME_HINT_EXAMPLES` como contenido.

No se debe implementar todavia:

- generacion dinamica de palabras candidatas con LLM;
- consulta a diccionarios externos;
- nuevas dependencias;
- eliminacion definitiva de `RHYME_HINT_EXAMPLES`;
- cambios en los pesos de evaluacion;
- cambios en `k`, `max_steps` o `alpha`.

## Nueva constante global

Anadir al inicio de `src/langgraph_beam_stanza.py`, junto al resto de constantes globales:

```python
ENABLE_RHYME_HINT_EXAMPLES = False
```

Para las pruebas actuales del usuario, debe quedar inicialmente en:

```python
False
```

### Significado

`ENABLE_RHYME_HINT_EXAMPLES = True`

- comportamiento actual;
- se usa el diccionario `RHYME_HINT_EXAMPLES`;
- si hay una rima objetivo conocida, se devuelven palabras candidatas asociadas;
- A y B pueden forzar palabras finales de esa lista.

`ENABLE_RHYME_HINT_EXAMPLES = False`

- no se usa el diccionario hardcodeado;
- el sistema no recibe palabras candidatas predefinidas;
- la reparacion de rima depende mas del LLM y del diagnostico de rima;
- se evita el sesgo de vocabulario del diccionario;
- es posible que la rima falle mas a menudo.

## Funcion principal afectada

La funcion clave es:

```python
get_candidate_final_words_for_rhyme(
    anchor_word: str | None,
    target_rhyme: str | None,
) -> list[str]
```

Esta funcion es el punto comun que usan las reparaciones de rima A y B para obtener palabras candidatas.

Por tanto, el cambio debe concentrarse ahi.

## Comportamiento actual

Actualmente la funcion hace algo equivalente a:

```python
candidates = list(RHYME_HINT_EXAMPLES.get(clean_target_rhyme, []))
if clean_anchor_word and clean_anchor_word not in candidates:
    candidates.append(clean_anchor_word)
```

Es decir:

1. Busca palabras candidatas en `RHYME_HINT_EXAMPLES`.
2. Si existe palabra ancla, la anade como referencia.
3. Elimina duplicados.

## Comportamiento nuevo

La funcion debe comportarse asi:

```python
if ENABLE_RHYME_HINT_EXAMPLES:
    candidates = list(RHYME_HINT_EXAMPLES.get(clean_target_rhyme, []))
else:
    candidates = []

if clean_anchor_word and clean_anchor_word not in candidates:
    candidates.append(clean_anchor_word)
```

### Por que mantener la palabra ancla

Aunque el diccionario este desactivado, conviene seguir devolviendo la palabra ancla.

Ejemplo:

```text
verso 1 termina en "helado"
rima objetivo = "ado"
```

Con el diccionario desactivado, no habra candidatas como:

```text
olvidado, callado, pasado
```

pero la funcion puede seguir devolviendo:

```text
helado
```

Esto permite que los prompts sigan teniendo una referencia minima. Despues, las funciones que calculan:

```python
enforced_candidate_final_words = [
    word for word in candidate_final_words if word != anchor_word
]
```

obtendran una lista vacia.

Eso significa:

- no habra palabras candidatas forzables;
- no se obligara al modelo a terminar en palabras concretas;
- se usara el prompt general de reparacion de rima.

Este comportamiento es deseable para el experimento.

## Cambios esperados en rima A

En `repair_stanza_outer_rhyme_with_ollama(...)`, cuando `ENABLE_RHYME_HINT_EXAMPLES = False`:

```python
candidate_final_words
```

debera contener como mucho la palabra final del verso 1.

Y:

```python
enforced_candidate_final_words
```

debera quedar vacia, porque se elimina la palabra ancla.

Consecuencia:

- no se forzara al verso 4 a terminar en una palabra del diccionario;
- el prompt debera pedir una palabra real que rime con el verso 1;
- el filtro `uses_candidate_final_word` no bloqueara variantes por no estar en una lista hardcodeada.

## Cambios esperados en rima B

En `repair_stanza_inner_rhyme_with_ollama(...)`, cuando `ENABLE_RHYME_HINT_EXAMPLES = False`:

```python
candidate_final_words
```

debera contener como mucho la palabra final del verso 2.

Y:

```python
enforced_candidate_final_words
```

debera quedar vacia.

Consecuencia:

- no se usara la generacion condicionada por palabra candidata concreta;
- no se haran llamadas separadas del tipo:

```text
termina exactamente en "llorar"
termina exactamente en "pasar"
```

- se usara la generacion general de reparacion B;
- el LLM tendra mas libertad para elegir la palabra final.

## Cambios en la generacion condicionada de B

La condicion actual para usar generacion condicionada es:

```python
if (
    INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE
    and enforced_candidate_final_words
):
    ...
```

No debe cambiarse.

Con `ENABLE_RHYME_HINT_EXAMPLES = False`, `enforced_candidate_final_words` quedara vacia, por lo que automaticamente se usara el metodo general.

Esto es correcto.

## Cambios en prompts

No es necesario cambiar directamente los prompts de reparacion A o B.

Al quedar vacia la lista `enforced_candidate_final_words`, las secciones del prompt que enumeran candidatas no deberian aparecer.

Debe comprobarse que los prompts siguen incluyendo una orientacion general:

```text
Busca una palabra real que comparta la rima consonante...
```

Si esto ya ocurre, no tocar los prompts.

## Cambios en reportes

No hace falta crear reportes nuevos, pero si conviene anadir al `run_parameters`:

```python
"enable_rhyme_hint_examples": ENABLE_RHYME_HINT_EXAMPLES,
```

Tambien conviene que aparezca en el `.txt` final:

```text
Diccionario de palabras candidatas de rima activo: False
```

Esto es importante para comparar ejecuciones con y sin diccionario.

## Cambios en `save_final_result(...)`

Anadir una linea en `final_stanza_*.txt`, cerca de los parametros de rima:

```text
Diccionario de palabras candidatas de rima activo: False
```

Debe tomar el valor desde:

```python
run_parameters.get("enable_rhyme_hint_examples")
```

## Cambios en `main(...)`

En el diccionario `run_parameters`, anadir:

```python
"enable_rhyme_hint_examples": ENABLE_RHYME_HINT_EXAMPLES,
```

## Que NO hacer en este paso

No implementar:

- generacion dinamica de candidatas con LLM;
- cache dinamica de rimas;
- llamadas adicionales a Ollama para buscar palabras;
- validacion nueva de palabras candidatas;
- cambios en `RHYME_HINT_EXAMPLES`;
- eliminacion del diccionario;
- cambios en `sonnet_metrics.py`;
- cambios en Beam Search;
- cambios en scoring;
- cambios en reparaciones metricas;
- cambios en elitismo.

## Pruebas minimas recomendadas

### 1. Probar con diccionario desactivado

Con:

```python
ENABLE_RHYME_HINT_EXAMPLES = False
```

Ejecutar:

```powershell
python src/langgraph_beam_stanza.py
```

Comprobar en el `.txt` final:

```text
Diccionario de palabras candidatas de rima activo: False
```

### 2. Revisar JSON de rima A

En `final_stanza_metrics_*.json`, revisar:

```text
outer_rhyme_repair_report.candidate_final_words
outer_rhyme_repair_report.enforced_candidate_final_words
```

Con el diccionario desactivado, se espera:

```text
candidate_final_words = [palabra_ancla] o []
enforced_candidate_final_words = []
```

### 3. Revisar JSON de rima B

En `final_stanza_metrics_*.json`, revisar:

```text
inner_rhyme_repair_report.candidate_final_words
inner_rhyme_repair_report.enforced_candidate_final_words
inner_rhyme_repair_report.conditioned_by_candidate
```

Con el diccionario desactivado, se espera:

```text
candidate_final_words = [palabra_ancla] o []
enforced_candidate_final_words = []
conditioned_by_candidate = False
```

### 4. Probar con diccionario activado

Cambiar:

```python
ENABLE_RHYME_HINT_EXAMPLES = True
```

Ejecutar de nuevo.

Comprobar que vuelven a aparecer candidatas cuando la rima existe en el diccionario.

### 5. Probar compilacion

Ejecutar:

```powershell
python -m compileall src
```

Debe compilar sin errores.

## Criterio de exito del Paso 12

El Paso 12 estara completo cuando:

1. Exista la constante:
   - `ENABLE_RHYME_HINT_EXAMPLES`.
2. Su valor inicial sea:
   - `False`.
3. `get_candidate_final_words_for_rhyme(...)` use `RHYME_HINT_EXAMPLES` solo si la constante esta activa.
4. Con la constante desactivada, A no fuerce palabras finales hardcodeadas.
5. Con la constante desactivada, B no use generacion condicionada por palabras hardcodeadas.
6. El valor de la constante aparezca en `run_parameters`.
7. El valor de la constante aparezca en `final_stanza_*.txt`.
8. No se haya eliminado `RHYME_HINT_EXAMPLES`.
9. No se haya implementado todavia la generacion dinamica de palabras candidatas.
10. No se haya cambiado Beam Search, scoring, metrica, rima ni elitismo.

## Resultado esperado

Con:

```python
ENABLE_RHYME_HINT_EXAMPLES = False
```

el sistema seguira intentando reparar rima A y B, pero sin apoyarse en listas hardcodeadas de palabras finales.

Esto permitira comprobar experimentalmente:

```text
- si la rima cae demasiado sin el diccionario;
- si las salidas son mas naturales;
- si merece la pena sustituir el diccionario por generacion dinamica con LLM.
```

Si los resultados empeoran mucho, el siguiente paso razonable seria implementar una generacion dinamica de candidatas de rima:

```text
LLM propone palabras candidatas -> el sistema valida rima -> se usan en la reparacion.
```
