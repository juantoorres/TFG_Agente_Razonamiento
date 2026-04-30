"""Funciones basicas de limpieza para el analisis de sonetos."""

"""
    NOTA:

    En esta primera version, la sinalefa no trata casos avanzados
    como sinalefa triple, hiatos poeticos, sineresis o diéresis.
"""

try:
    import pyphen
except ImportError:
    # Para separar silabas con pyphen, instalar con:
    # pip install pyphen
    pyphen = None


COMMON_PUNCTUATION = ".,;:!¡?¿\"'«»()[]{}"
PUNCTUATION_TRANSLATION = str.maketrans("", "", COMMON_PUNCTUATION)
ACCENTED_VOWELS = "áéíóú"
VOWELS = "aeiouáéíóúü"
STRONG_VOWELS = "aeoáéó"
EXPECTED_SONNET_VERSES = 14
TARGET_SYLLABLES_PER_VERSE = 11
VERSE_COUNT_WEIGHT = 0.2
SYLLABLE_WEIGHT = 0.4
RHYME_WEIGHT = 0.4
EXPECTED_SONNET_RHYME_PATTERN = [
    "A",
    "B",
    "B",
    "A",
    "A",
    "B",
    "B",
    "A",
    "C",
    "D",
    "C",
    "C",
    "D",
    "C",
]
ACCENT_TRANSLATION = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
    }
)

if pyphen is None:
    SPANISH_HYPHENATOR = None
else:
    SPANISH_HYPHENATOR = pyphen.Pyphen(lang="es_ES")


def normalize_text(text: str) -> str:
    """Normaliza un texto conservando tildes, diéresis, ene y puntuacion.

    Convierte el texto a minusculas, elimina espacios iniciales y finales,
    y reemplaza cualquier secuencia de espacios internos por un unico espacio.
    """
    return " ".join(text.lower().strip().split())


def normalize_verses_input(sonnet: str | list[str]) -> list[str]:
    """Normaliza la entrada de un soneto a una lista de versos.

    Si recibe un texto completo, lo separa por saltos de linea, aplica
    ``strip`` a cada verso y elimina lineas vacias. Si recibe una lista, aplica
    el mismo tratamiento a cada elemento. No modifica el contenido interno de
    los versos.
    """
    if isinstance(sonnet, str):
        raw_verses = sonnet.splitlines()
    elif isinstance(sonnet, list):
        raw_verses = sonnet
    else:
        raise TypeError("El soneto debe ser un texto o una lista de versos.")

    return [verse.strip() for verse in raw_verses if verse.strip()]


def evaluate_verse_count(sonnet: str | list[str]) -> dict[str, object]:
    """Evalua si un soneto tiene exactamente 14 versos.

    Acepta un texto completo separado por saltos de linea o una lista de versos.
    Devuelve una puntuacion binaria: ``1.0`` si hay exactamente 14 versos y
    ``0.0`` en caso contrario.
    """
    verses = normalize_verses_input(sonnet)
    actual_verses = len(verses)
    is_valid = actual_verses == EXPECTED_SONNET_VERSES
    errors = []

    if actual_verses < EXPECTED_SONNET_VERSES:
        missing_verses = EXPECTED_SONNET_VERSES - actual_verses
        errors.append(
            f"El soneto tiene {actual_verses} versos; faltan {missing_verses}."
        )
    elif actual_verses > EXPECTED_SONNET_VERSES:
        extra_verses = actual_verses - EXPECTED_SONNET_VERSES
        errors.append(
            f"El soneto tiene {actual_verses} versos; sobran {extra_verses}."
        )

    return {
        "expected_verses": EXPECTED_SONNET_VERSES,
        "actual_verses": actual_verses,
        "score": 1.0 if is_valid else 0.0,
        "is_valid": is_valid,
        "errors": errors,
    }


def remove_punctuation(text: str) -> str:
    """Elimina signos de puntuacion comunes sin alterar letras ni espacios.

    Conserva letras con tilde, diéresis, ene y los espacios entre palabras.
    Solo retira los signos definidos en ``COMMON_PUNCTUATION``.
    """
    return text.translate(PUNCTUATION_TRANSLATION)


def clean_verse(verse: str) -> str:
    """Limpia un verso aplicando normalizacion y retirada de puntuacion.

    Primero normaliza mayusculas y espacios, despues elimina puntuacion comun
    y finalmente vuelve a normalizar los espacios resultantes.
    """
    normalized_verse = normalize_text(verse)
    verse_without_punctuation = remove_punctuation(normalized_verse)
    return normalize_text(verse_without_punctuation)


def split_words(verse: str) -> list[str]:
    """Devuelve las palabras limpias de un verso.

    Usa ``clean_verse`` internamente. Si el verso esta vacio o solo contiene
    espacios y puntuacion eliminable, devuelve una lista vacia.
    """
    clean_text = clean_verse(verse)
    if not clean_text:
        return []
    return clean_text.split(" ")


def get_last_word(verse: str) -> str | None:
    """Devuelve la ultima palabra util de un verso.

    Limpia el verso con ``split_words`` y devuelve su ultima palabra. Si el
    verso esta vacio o no contiene palabras tras la limpieza, devuelve ``None``.
    """
    words = split_words(verse)
    if not words:
        return None
    return words[-1]


def get_last_words(verses: list[str]) -> list[str | None]:
    """Devuelve la ultima palabra util de cada verso recibido.

    Usa ``get_last_word`` internamente para mantener una unica logica de
    limpieza y extraccion.
    """
    return [get_last_word(verse) for verse in verses]


def has_written_accent(word: str) -> bool:
    """Indica si una palabra contiene una vocal con tilde escrita."""
    normalized_word = normalize_text(word)
    return any(vowel in normalized_word for vowel in ACCENTED_VOWELS)


def get_stressed_vowel_index(word: str) -> int | None:
    """Devuelve el indice de la vocal tonica si hay tilde escrita.

    La busqueda no distingue entre mayusculas y minusculas. Si la palabra no
    contiene ninguna vocal con tilde, devuelve ``None``.
    """
    normalized_word = normalize_text(word)
    for index, character in enumerate(normalized_word):
        if character in ACCENTED_VOWELS:
            return index
    return None


def split_word_syllables(word: str) -> list[str]:
    """Separa una palabra en silabas aproximadas usando pyphen.

    Recibe una palabra, la limpia de puntuacion y espacios, y devuelve una
    lista de silabas aproximadas con el diccionario espanol ``es_ES``. Conserva
    tildes y ene. Si la palabra esta vacia, devuelve una lista vacia.

    pyphen usa patrones de separacion aproximados; mas adelante el conteo del
    verso completo se ajustara con sinalefas y la regla de la palabra final.
    """
    clean_word = clean_verse(word).replace(" ", "")
    if not clean_word:
        return []

    if SPANISH_HYPHENATOR is None:
        raise ImportError(
            "pyphen no esta instalado. Instalalo con: pip install pyphen"
        )

    syllables = SPANISH_HYPHENATOR.inserted(clean_word).split("-")
    return [syllable for syllable in syllables if syllable]


def count_word_syllables(word: str) -> int:
    """Cuenta las silabas aproximadas de una palabra usando ``pyphen``.

    Si la palabra esta vacia, devuelve ``0``. No aplica sinalefa ni ajustes de
    verso completo; esas reglas se incorporaran en funciones posteriores.
    """
    return len(split_word_syllables(word))


def _get_stressed_syllable_index(word: str, stressed_vowel_index: int) -> int:
    """Localiza la silaba que contiene la vocal tonica indicada."""
    syllables = split_word_syllables(word)
    current_index = 0

    for syllable_index, syllable in enumerate(syllables):
        next_index = current_index + len(syllable)
        if current_index <= stressed_vowel_index < next_index:
            return syllable_index
        current_index = next_index

    return len(syllables) - 1


def get_word_stress_type(word: str) -> str:
    """Clasifica una palabra como aguda, llana o esdrujula.

    Si la palabra tiene tilde escrita, se usa ``pyphen`` para separar sus
    silabas y ubicar en cual cae la vocal tonica. Si no tiene tilde, se aplican
    las reglas generales del espanol: palabras terminadas en vocal, ``n`` o
    ``s`` son llanas; el resto son agudas.
    """
    clean_word = clean_verse(word).replace(" ", "")
    if not clean_word:
        raise ValueError("No se puede clasificar una palabra vacia.")

    stressed_vowel_index = get_stressed_vowel_index(clean_word)
    if stressed_vowel_index is None:
        if clean_word[-1] in "aeiouáéíóúüns":
            return "llana"
        return "aguda"

    syllables = split_word_syllables(clean_word)
    stressed_syllable_index = _get_stressed_syllable_index(
        clean_word,
        stressed_vowel_index,
    )
    position_from_end = len(syllables) - stressed_syllable_index

    if position_from_end == 1:
        return "aguda"
    if position_from_end == 2:
        return "llana"
    return "esdrujula"


def get_final_word_syllable_adjustment(word: str) -> int:
    """Devuelve el ajuste metrico asociado a la acentuacion final.

    Las palabras agudas suman una silaba, las llanas no modifican el conteo y
    las esdrujulas restan una silaba.
    """
    stress_type = get_word_stress_type(word)
    if stress_type == "aguda":
        return 1
    if stress_type == "llana":
        return 0
    return -1


def count_raw_verse_syllables(verse: str) -> int:
    """Cuenta las silabas de un verso sin ajustes metricos.

    Limpia el verso con ``split_words`` y suma las silabas aproximadas de cada
    palabra usando ``count_word_syllables``. No aplica sinalefas ni ajuste por
    palabra final. Si el verso esta vacio, devuelve ``0``.
    """
    words = split_words(verse)
    if not words:
        return 0
    return sum(count_word_syllables(word) for word in words)


def count_verse_syllables_without_sinalefa(verse: str) -> int:
    """Cuenta las silabas metricas de un verso sin aplicar sinalefas.

    Parte del conteo crudo por palabras y aplica solo el ajuste metrico de la
    palabra final: +1 si es aguda, 0 si es llana y -1 si es esdrujula. Si el
    verso esta vacio, devuelve ``0``.
    """
    raw_syllable_count = count_raw_verse_syllables(verse)
    last_word = get_last_word(verse)
    if last_word is None:
        return 0
    return raw_syllable_count + get_final_word_syllable_adjustment(last_word)


def analyze_verse_syllables_basic(verse: str) -> dict[str, object]:
    """Devuelve una traza basica del conteo silabico de un verso.

    La informacion incluye el verso original, su version limpia, las palabras,
    las silabas aproximadas de cada palabra, el conteo crudo, la ultima palabra,
    su tipo acentual, el ajuste final aplicado y el total sin sinalefa.
    """
    clean_text = clean_verse(verse)
    words = split_words(verse)
    word_syllables = [
        {
            "word": word,
            "syllables": split_word_syllables(word),
            "syllable_count": count_word_syllables(word),
        }
        for word in words
    ]
    raw_syllable_count = sum(
        word_info["syllable_count"] for word_info in word_syllables
    )
    last_word = words[-1] if words else None

    if last_word is None:
        stress_type = None
        final_adjustment = 0
        total_without_sinalefa = 0
    else:
        stress_type = get_word_stress_type(last_word)
        final_adjustment = get_final_word_syllable_adjustment(last_word)
        total_without_sinalefa = raw_syllable_count + final_adjustment

    return {
        "verse": verse,
        "clean_verse": clean_text,
        "words": words,
        "word_syllables": word_syllables,
        "raw_syllable_count": raw_syllable_count,
        "last_word": last_word,
        "stress_type": stress_type,
        "final_adjustment": final_adjustment,
        "total_without_sinalefa": total_without_sinalefa,
    }


def is_vowel_sound_start(word: str) -> bool:
    """Indica si una palabra empieza por sonido vocalico.

    Considera vocales simples, vocales con tilde, ``ü`` y palabras que empiezan
    por ``h`` seguida de vocal, como ``hora`` o ``hierba``.
    """
    clean_word = clean_verse(word).replace(" ", "")
    if not clean_word:
        return False

    if clean_word[0] in VOWELS:
        return True
    return len(clean_word) > 1 and clean_word[0] == "h" and clean_word[1] in VOWELS


def is_vowel_sound_end(word: str) -> bool:
    """Indica si una palabra termina en sonido vocalico.

    Considera vocales simples, vocales con tilde y ``ü`` como finales vocalicos.
    """
    clean_word = clean_verse(word).replace(" ", "")
    if not clean_word:
        return False
    return clean_word[-1] in VOWELS


def get_sinalefa_pairs(words: list[str]) -> list[tuple[str, str]]:
    """Devuelve las parejas consecutivas de palabras con sinalefa.

    Hay sinalefa cuando una palabra termina en sonido vocalico y la siguiente
    empieza por sonido vocalico. Esta primera version no trata casos avanzados
    como sinalefa triple, hiatos poeticos, sineresis o diéresis.
    """
    if len(words) < 2:
        return []

    pairs = []
    for current_word, next_word in zip(words, words[1:]):
        if is_vowel_sound_end(current_word) and is_vowel_sound_start(next_word):
            pairs.append((current_word, next_word))
    return pairs


def count_sinalefas(words: list[str]) -> int:
    """Cuenta las sinalefas entre palabras consecutivas ya limpias."""
    return len(get_sinalefa_pairs(words))


def count_verse_syllables(verse: str) -> int:
    """Cuenta las silabas metricas de un verso aplicando sinalefas basicas.

    Calcula primero el total sin sinalefa, detecta las sinalefas entre palabras
    consecutivas y resta una silaba por cada una. Si el verso esta vacio,
    devuelve ``0``.
    """
    words = split_words(verse)
    if not words:
        return 0

    total_without_sinalefa = count_verse_syllables_without_sinalefa(verse)
    return total_without_sinalefa - count_sinalefas(words)


def analyze_verse_syllables(verse: str) -> dict[str, object]:
    """Devuelve una traza del conteo silabico con sinalefas basicas.

    Extiende ``analyze_verse_syllables_basic`` con el numero de sinalefas, las
    parejas donde se detectan y el total metrico final.
    """
    analysis = analyze_verse_syllables_basic(verse)
    words = analysis["words"]

    if not isinstance(words, list):
        words = []

    sinalefa_pairs = get_sinalefa_pairs(words)
    total_without_sinalefa = analysis["total_without_sinalefa"]
    if not isinstance(total_without_sinalefa, int):
        total_without_sinalefa = 0

    analysis["sinalefa_count"] = len(sinalefa_pairs)
    analysis["sinalefa_pairs"] = sinalefa_pairs
    analysis["total_syllables"] = total_without_sinalefa - len(sinalefa_pairs)
    return analysis


def strip_accents(text: str) -> str:
    """Elimina tildes y diéresis de vocales conservando la ene.

    Devuelve siempre el texto en minusculas. La ``ñ`` se conserva porque no se
    aplica normalizacion Unicode global, solo una traduccion explicita de
    vocales acentuadas.
    """
    return text.lower().translate(ACCENT_TRANSLATION)


def _get_syllable_start_positions(syllables: list[str]) -> list[int]:
    """Devuelve el indice inicial de cada silaba dentro de la palabra."""
    positions = []
    current_position = 0
    for syllable in syllables:
        positions.append(current_position)
        current_position += len(syllable)
    return positions


def _find_stressed_vowel_offset_in_syllable(syllable: str) -> int | None:
    """Localiza la vocal tonica aproximada dentro de una silaba sin tilde."""
    for index, character in enumerate(syllable):
        if character in STRONG_VOWELS:
            return index

    for index in range(len(syllable) - 1, -1, -1):
        if syllable[index] in VOWELS:
            return index

    return None


def get_last_stressed_vowel_position(word: str) -> int | None:
    """Devuelve el indice de la vocal tonica dentro de una palabra.

    Si la palabra tiene tilde escrita, devuelve la posicion de esa vocal. Si no
    tiene tilde, aplica las reglas generales del espanol y usa ``pyphen`` para
    separar silabas y localizar la vocal tonica aproximada. Si no puede
    determinarla, devuelve ``None``.
    """
    clean_word = clean_verse(word).replace(" ", "")
    if not clean_word:
        return None

    stressed_vowel_index = get_stressed_vowel_index(clean_word)
    if stressed_vowel_index is not None:
        return stressed_vowel_index

    syllables = split_word_syllables(clean_word)
    if not syllables:
        return None

    stress_type = get_word_stress_type(clean_word)
    if stress_type == "aguda":
        stressed_syllable_index = len(syllables) - 1
    elif stress_type == "llana":
        stressed_syllable_index = max(len(syllables) - 2, 0)
    else:
        stressed_syllable_index = max(len(syllables) - 3, 0)

    vowel_offset = _find_stressed_vowel_offset_in_syllable(
        syllables[stressed_syllable_index]
    )
    if vowel_offset is None:
        return None

    syllable_positions = _get_syllable_start_positions(syllables)
    return syllable_positions[stressed_syllable_index] + vowel_offset


def extract_consonant_rhyme(word: str) -> str:
    """Extrae la rima consonante de una palabra.

    La rima consonante va desde la vocal tonica hasta el final de la palabra.
    Para facilitar la comparacion, la rima se devuelve en minusculas y sin
    tildes, conservando la ``ñ``.
    """
    clean_word = clean_verse(word).replace(" ", "")
    stressed_vowel_position = get_last_stressed_vowel_position(clean_word)
    if stressed_vowel_position is None:
        return ""
    return strip_accents(clean_word[stressed_vowel_position:])


def extract_verse_rhyme(verse: str) -> str | None:
    """Extrae la rima consonante de la ultima palabra de un verso.

    Si el verso no contiene palabras utiles, devuelve ``None``.
    """
    last_word = get_last_word(verse)
    if last_word is None:
        return None
    return extract_consonant_rhyme(last_word)


def extract_rhymes(verses: list[str]) -> list[str | None]:
    """Devuelve la rima consonante de cada verso recibido."""
    return [extract_verse_rhyme(verse) for verse in verses]


def _scheme_letter(index: int) -> str:
    """Convierte un indice en una letra de esquema de rima."""
    letters = ""
    current_index = index

    while True:
        letters = chr(ord("A") + current_index % 26) + letters
        current_index = current_index // 26 - 1
        if current_index < 0:
            return letters


def build_rhyme_scheme(rhymes: list[str | None]) -> list[str | None]:
    """Convierte una lista de rimas reales en letras de esquema.

    La primera rima distinta se marca como ``A``, la segunda como ``B`` y asi
    sucesivamente. Las rimas ``None`` se conservan como ``None``.
    """
    rhyme_to_letter: dict[str, str] = {}
    scheme: list[str | None] = []

    for rhyme in rhymes:
        if rhyme is None:
            scheme.append(None)
            continue

        if rhyme not in rhyme_to_letter:
            rhyme_to_letter[rhyme] = _scheme_letter(len(rhyme_to_letter))
        scheme.append(rhyme_to_letter[rhyme])

    return scheme


def describe_rhyme_errors(
    actual_scheme: list[str | None],
    expected_pattern: list[str],
) -> list[str]:
    """Devuelve mensajes legibles para los errores de esquema de rima."""
    errors = []

    for index, expected_rhyme in enumerate(expected_pattern):
        verse_number = index + 1
        if index >= len(actual_scheme) or actual_scheme[index] is None:
            errors.append(f"Verso {verse_number}: no se pudo extraer rima.")
            continue

        actual_rhyme = actual_scheme[index]
        if actual_rhyme != expected_rhyme:
            errors.append(
                f"Verso {verse_number}: rima {actual_rhyme}, "
                f"pero se esperaba {expected_rhyme}."
            )

    return errors


def evaluate_rhyme_scheme(verses: list[str]) -> dict[str, object]:
    """Evalua el esquema de rima consonante de un soneto clasico.

    Extrae las rimas reales, construye el esquema observado y lo compara con el
    patron ``ABBA ABBA CDC CDC``. Si hay mas de 14 versos, compara solo los
    primeros 14. Si hay menos, las posiciones faltantes cuentan como
    incorrectas.
    """
    errors = []
    total_positions = EXPECTED_SONNET_VERSES

    if len(verses) != total_positions:
        errors.append(
            f"Se recibieron {len(verses)} versos; se esperaban "
            f"{total_positions}."
        )

    verses_to_compare = verses[:total_positions]
    actual_rhymes = extract_rhymes(verses_to_compare)
    actual_scheme = build_rhyme_scheme(actual_rhymes)

    correct_positions = 0
    for index, expected_rhyme in enumerate(EXPECTED_SONNET_RHYME_PATTERN):
        if index < len(actual_scheme) and actual_scheme[index] == expected_rhyme:
            correct_positions += 1

    errors.extend(
        describe_rhyme_errors(actual_scheme, EXPECTED_SONNET_RHYME_PATTERN)
    )

    return {
        "expected_pattern": EXPECTED_SONNET_RHYME_PATTERN,
        "actual_rhymes": actual_rhymes,
        "actual_scheme": actual_scheme,
        "correct_positions": correct_positions,
        "total_positions": total_positions,
        "score": correct_positions / total_positions,
        "errors": errors,
    }


def describe_syllable_errors(verses: list[str], counts: list[int]) -> list[str]:
    """Devuelve mensajes legibles sobre errores de computo silabico.

    Revisa los primeros 14 versos. Los versos que no tienen 11 silabas generan
    un error descriptivo. Si faltan versos, se informa de cada posicion
    ausente; si sobran, se anade un error global indicando cuantos sobran.
    """
    errors = []
    total_expected_verses = EXPECTED_SONNET_VERSES

    for index in range(min(len(verses), total_expected_verses)):
        count = counts[index]
        if count != TARGET_SYLLABLES_PER_VERSE:
            errors.append(
                f"Verso {index + 1}: tiene {count} silabas, "
                f"pero deberia tener {TARGET_SYLLABLES_PER_VERSE}."
            )

    if len(verses) < total_expected_verses:
        for index in range(len(verses), total_expected_verses):
            errors.append(
                f"Verso {index + 1}: falta verso; deberia tener "
                f"{TARGET_SYLLABLES_PER_VERSE} silabas."
            )

    if len(verses) > total_expected_verses:
        extra_verses = len(verses) - total_expected_verses
        errors.append(
            f"Sobran {extra_verses} versos; se recibieron {len(verses)} "
            f"y se esperaban {total_expected_verses}."
        )

    return errors


def evaluate_syllable_count(verses: list[str]) -> dict[str, object]:
    """Evalua si los versos de un soneto tienen 11 silabas metricas.

    Usa ``count_verse_syllables`` para contar cada verso. El score se calcula
    como ``versos_correctos / 14``. Si hay mas de 14 versos, solo los primeros
    14 cuentan para la puntuacion; si hay menos, los versos faltantes cuentan
    como incorrectos.
    """
    total_expected_verses = EXPECTED_SONNET_VERSES
    verses_to_evaluate = verses[:total_expected_verses]
    verse_syllable_counts = [
        count_verse_syllables(verse) for verse in verses_to_evaluate
    ]
    correct_verses = sum(
        1
        for count in verse_syllable_counts
        if count == TARGET_SYLLABLES_PER_VERSE
    )

    return {
        "target_syllables": TARGET_SYLLABLES_PER_VERSE,
        "verse_syllable_counts": verse_syllable_counts,
        "correct_verses": correct_verses,
        "total_expected_verses": total_expected_verses,
        "score": correct_verses / total_expected_verses,
        "errors": describe_syllable_errors(verses, verse_syllable_counts),
    }


def _clamp_score(score: float) -> float:
    """Limita una puntuacion al intervalo entre 0.0 y 1.0."""
    return max(0.0, min(1.0, score))


def build_sonnet_feedback(evaluation: dict[str, object]) -> str:
    """Construye feedback textual para mejorar un soneto generado.

    El texto resume la puntuacion global e incluye errores de extension,
    computo silabico y rima. Esta pensado para pasarlo despues al LLM generador
    dentro del flujo de LangGraph.
    """
    score_20 = evaluation["score_20"]
    verse_count = evaluation["verse_count"]
    syllables = evaluation["syllables"]
    rhyme = evaluation["rhyme"]

    feedback_lines = [
        f"Puntuacion formal total: {score_20}/20.",
    ]

    all_errors = evaluation["errors"]
    if not all_errors:
        feedback_lines.append(
            "El soneto cumple las restricciones formales evaluadas: "
            "14 versos, computo silabico objetivo y esquema de rima esperado."
        )
        return "\n".join(feedback_lines)

    feedback_lines.append(
        "Revisa los siguientes aspectos formales antes de generar una nueva "
        "version:"
    )

    feedback_lines.append("Numero de versos:")
    verse_count_errors = verse_count["errors"]
    if verse_count_errors:
        feedback_lines.extend(f"- {error}" for error in verse_count_errors)
    else:
        feedback_lines.append("- El numero de versos es correcto.")

    feedback_lines.append("Silabas metricas:")
    syllable_errors = syllables["errors"]
    if syllable_errors:
        feedback_lines.extend(f"- {error}" for error in syllable_errors)
    else:
        feedback_lines.append("- Todos los versos evaluados tienen 11 silabas.")

    feedback_lines.append("Rima consonante:")
    rhyme_errors = rhyme["errors"]
    if rhyme_errors:
        feedback_lines.extend(f"- {error}" for error in rhyme_errors)
    else:
        feedback_lines.append("- El esquema de rima coincide con ABBA ABBA CDC CDC.")

    return "\n".join(feedback_lines)


def evaluate_sonnet(sonnet: str | list[str]) -> dict[str, object]:
    """Evalua objetivamente un soneto segun extension, silabas y rima.

    Acepta un texto completo o una lista de versos, normaliza la entrada y
    combina las puntuaciones parciales con pesos: 0.2 para numero de versos,
    0.4 para silabas y 0.4 para rima.
    """
    verses = normalize_verses_input(sonnet)
    verse_count_evaluation = evaluate_verse_count(verses)
    syllable_evaluation = evaluate_syllable_count(verses)
    rhyme_evaluation = evaluate_rhyme_scheme(verses)

    raw_score = (
        float(verse_count_evaluation["score"]) * VERSE_COUNT_WEIGHT
        + float(syllable_evaluation["score"]) * SYLLABLE_WEIGHT
        + float(rhyme_evaluation["score"]) * RHYME_WEIGHT
    )
    score = round(_clamp_score(raw_score), 4)

    errors = (
        list(verse_count_evaluation["errors"])
        + list(syllable_evaluation["errors"])
        + list(rhyme_evaluation["errors"])
    )

    evaluation: dict[str, object] = {
        "score": score,
        "score_20": round(score * 20, 2),
        "verses": verses,
        "verse_count": verse_count_evaluation,
        "syllables": syllable_evaluation,
        "rhyme": rhyme_evaluation,
        "errors": errors,
        "feedback": "",
    }
    evaluation["feedback"] = build_sonnet_feedback(evaluation)
    return evaluation


if __name__ == "__main__":
    examples = [
        "En el espejo de mi alma, se ven remotos recuerdos",
        "¡Oh dulce memoria del tiempo perdido!",
        "La tarde cae sobre el mar.",
        "",
        "...",
        "  La vida   es sueño; y los sueños, sueños son.  ",
    ]

    for example in examples:
        print(f"Original: {example}")
        print(f"Normalizado: {normalize_text(example)}")
        print(f"Sin puntuacion: {remove_punctuation(normalize_text(example))}")
        print(f"Verso limpio: {clean_verse(example)}")
        print(f"Palabras: {split_words(example)}")
        print(f"Ultima palabra: {get_last_word(example)}")
        print()

    print(f"Ultimas palabras: {get_last_words(examples)}")
    print()

    stress_examples = [
        "canción",
        "memoria",
        "pájaro",
        "reloj",
        "tiempo",
        "corazón",
        "lágrima",
    ]

    for word in stress_examples:
        print(f"Palabra: {word}")
        print(f"Tiene tilde: {has_written_accent(word)}")
        print(f"Indice vocal tonica: {get_stressed_vowel_index(word)}")
        print(f"Tipo: {get_word_stress_type(word)}")
        print(f"Ajuste: {get_final_word_syllable_adjustment(word)}")
        print()

    syllable_examples = [
        "tiempo",
        "memoria",
        "corazón",
        "pájaro",
        "río",
        "alma",
        "sueño",
        "olvidado",
    ]

    for word in syllable_examples:
        print(f"Palabra: {word}")
        print(f"Silabas: {split_word_syllables(word)}")
        print(f"Numero de silabas: {count_word_syllables(word)}")
        print()

    verse_syllable_examples = [
        "En el espejo de mi alma",
        "se ven remotos recuerdos",
        "Oh dulce memoria del tiempo perdido",
        "La vida es sueño",
        "Mi corazón volvió a cantar",
        "",
    ]

    for verse in verse_syllable_examples:
        analysis = analyze_verse_syllables_basic(verse)
        print(f"Verso: {verse}")
        print(f"Verso limpio: {analysis['clean_verse']}")
        print(f"Palabras: {analysis['words']}")
        print(f"Silabas por palabra: {analysis['word_syllables']}")
        print(f"Conteo crudo: {analysis['raw_syllable_count']}")
        print(f"Ultima palabra: {analysis['last_word']}")
        print(f"Tipo final: {analysis['stress_type']}")
        print(f"Ajuste final: {analysis['final_adjustment']}")
        print(f"Total sin sinalefa: {analysis['total_without_sinalefa']}")
        print()

    sinalefa_examples = [
        "En el espejo de mi alma",
        "Mi alma espera en silencio",
        "La vida es sueño",
        "Vuelve a abrirse el alba",
        "Canta el ave en el aire",
        "El tiempo huye ahora",
    ]

    for verse in sinalefa_examples:
        analysis = analyze_verse_syllables(verse)
        print(f"Verso: {verse}")
        print(f"Palabras: {analysis['words']}")
        print(f"Total sin sinalefa: {analysis['total_without_sinalefa']}")
        print(f"Sinalefas: {analysis['sinalefa_count']}")
        print(f"Parejas con sinalefa: {analysis['sinalefa_pairs']}")
        print(f"Total con sinalefa: {analysis['total_syllables']}")
        print()

    rhyme_word_examples = [
        "corazón",
        "canción",
        "memoria",
        "gloria",
        "vida",
        "herida",
        "tiempo",
        "recuerdo",
    ]

    for word in rhyme_word_examples:
        print(f"Palabra: {word}")
        print(f"Sin tildes: {strip_accents(word)}")
        print(f"Posicion vocal tonica: {get_last_stressed_vowel_position(word)}")
        print(f"Rima consonante: {extract_consonant_rhyme(word)}")
        print()

    rhyme_verse_examples = [
        "Late en silencio mi viejo corazón",
        "Se eleva en la noche una canción",
        "Guarda la tarde su antigua memoria",
        "Brilla en la sombra la fugaz gloria",
        "...",
    ]

    for verse in rhyme_verse_examples:
        print(f"Verso: {verse}")
        print(f"Ultima palabra: {get_last_word(verse)}")
        print(f"Rima consonante: {extract_verse_rhyme(verse)}")
        print()

    print(f"Rimas de versos: {extract_rhymes(rhyme_verse_examples)}")
    print()

    artificial_sonnet = [
        "Late profundo mi corazón",
        "Guarda la tarde antigua memoria",
        "Resplandece callada la gloria",
        "Vuelve en la sombra la canción",
        "Arde secreta mi razón",
        "Camina lenta la historia",
        "Renace limpia la victoria",
        "Tiembla en la noche mi ilusión",
        "Busca la luz de nueva vida",
        "Se pierde mirando el mar",
        "Regresa curando la herida",
        "Abre su puerta la despedida",
        "Aprende mirando hacia el mar",
        "Sueña despierta la despedida",
    ]
    rhyme_evaluation = evaluate_rhyme_scheme(artificial_sonnet)

    print("Evaluacion de rima del soneto artificial:")
    print(f"Patron esperado: {rhyme_evaluation['expected_pattern']}")
    print(f"Rimas reales: {rhyme_evaluation['actual_rhymes']}")
    print(f"Esquema real: {rhyme_evaluation['actual_scheme']}")
    print(f"Posiciones correctas: {rhyme_evaluation['correct_positions']}")
    print(f"Puntuacion: {rhyme_evaluation['score']}")
    print(f"Errores: {rhyme_evaluation['errors']}")
    print()

    syllable_evaluation = evaluate_syllable_count(artificial_sonnet)

    print("Evaluacion silabica del soneto artificial:")
    for verse_number, (verse, syllable_count) in enumerate(
        zip(artificial_sonnet, syllable_evaluation["verse_syllable_counts"]),
        start=1,
    ):
        print(f"Verso {verse_number}: {syllable_count} silabas -> {verse}")

    print(f"Versos correctos: {syllable_evaluation['correct_verses']}")
    print(f"Puntuacion: {syllable_evaluation['score']}")
    print(f"Errores: {syllable_evaluation['errors']}")
    print()

    artificial_sonnet_text = "\n".join(artificial_sonnet)
    short_sonnet = artificial_sonnet[:10]
    long_sonnet = artificial_sonnet + ["Verso adicional fuera del soneto"]

    print("Evaluacion de extension del soneto como texto:")
    print(evaluate_verse_count(artificial_sonnet_text))
    print()

    print("Evaluacion de extension de lista con menos de 14 versos:")
    print(evaluate_verse_count(short_sonnet))
    print()

    print("Evaluacion de extension de lista con mas de 14 versos:")
    print(evaluate_verse_count(long_sonnet))
    print()
    print("------------------------------------------------------------------------------------")
    print()

    sonnet_evaluation = evaluate_sonnet(artificial_sonnet_text)

    print("Evaluacion global objetiva del soneto artificial:")
    print(f"Score: {sonnet_evaluation['score']}")
    print(f"Score sobre 20: {sonnet_evaluation['score_20']}")
    print(f"Errores: {sonnet_evaluation['errors']}")
    print("Feedback:")
    print(sonnet_evaluation["feedback"])
