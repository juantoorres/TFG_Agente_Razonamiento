from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Dict, List, TypedDict

import requests
from langgraph.graph import END, StateGraph

from sonnet_metrics import (
    TARGET_SYLLABLES_PER_VERSE,
    count_verse_syllables,
    diagnose_stanza_inner_rhyme,
    diagnose_stanza_outer_rhyme,
    evaluate_stanza_abba,
    get_last_word,
)


# =========================
# 1. Constantes globales
# =========================

BASE_URL = "http://localhost:11434"

# Modelo local usado para generar las estrofas candidatas.
GENERATION_MODEL = "mistral:7b-instruct"

# Hiperparametros principales del experimento. Estan concentrados aqui para
# poder documentarlos y cambiarlos de forma sencilla en pruebas posteriores.
TEMPERATURE = 0.3
RETRY_TEMPERATURE = 0.1 # Que en el segundo intento, el modelo no sea tan 'creativo' y se ciña más al formato estricto.
NUM_PREDICT = 600
ALPHA = 1.0
EPSILON = 1e-6

# Heuristica opcional: usa un diccionario hardcodeado de palabras finales
# candidatas para rimas frecuentes. En False permite probar el sistema sin
# apoyo lexico fijo antes de plantear una generacion dinamica de candidatas.
ENABLE_RHYME_HINT_EXAMPLES = False

# Fase experimental opcional: intenta reparar solo la metrica de versos que no
# tienen 11 silabas. No toca la rima ni bloquea versos correctos.
ENABLE_LOCAL_METER_REPAIR = True
METER_REPAIR_VARIANTS_PER_VERSE = 5
METER_REPAIR_TEMPERATURE = 0.4
METER_REPAIR_NUM_PREDICT = 200

# Fase experimental opcional: si la rima exterior AXYA falla, intenta
# reescribir solo el verso 4 para que rime con el verso 1.
ENABLE_OUTER_RHYME_REPAIR = True
OUTER_RHYME_REPAIR_VARIANTS = 5
OUTER_RHYME_REPAIR_TEMPERATURE = 0.3
OUTER_RHYME_REPAIR_NUM_PREDICT = 200

# Segunda oportunidad metrica tras una variante que ya corrige la rima A.
ENABLE_POST_A_RHYME_METER_REPAIR = True
POST_A_RHYME_METER_REPAIR_VARIANTS = 5
POST_A_RHYME_METER_REPAIR_TEMPERATURE = 0.4
POST_A_RHYME_METER_REPAIR_NUM_PREDICT = 200

# Fase experimental opcional: si la rima interior B falla, intenta
# reescribir solo el verso 3 para que rime con el verso 2.
ENABLE_INNER_RHYME_REPAIR = True
INNER_RHYME_REPAIR_VARIANTS = 5
INNER_RHYME_REPAIR_TEMPERATURE = 0.3
INNER_RHYME_REPAIR_NUM_PREDICT = 200
INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE = False
INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE = 2
INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS = 4

# Segunda oportunidad metrica tras una variante que ya corrige la rima B.
ENABLE_POST_B_RHYME_METER_REPAIR = True
POST_B_RHYME_METER_REPAIR_VARIANTS = 5
POST_B_RHYME_METER_REPAIR_TEMPERATURE = 0.4
POST_B_RHYME_METER_REPAIR_NUM_PREDICT = 200

ENABLE_BEAM_ELITISM = True
ELITE_BEAMS_TO_KEEP = 1
ENABLE_PROMPT_CONSTRAINT_PROTECTION = True


# =========================
# 2. Estado del grafo
# =========================

class BeamSearchState(TypedDict):
    question: str
    beams: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    step: int
    max_steps: int
    k: int


# Cada beam representa una estrofa completa, no un verso aislado.
# Campos esperados en cada beam/candidato:
# - stanza: list[str] | None
#   Lista de 4 versos candidata. En el estado inicial vale None.
# - score: float
#   Puntuacion agregada del beam.
# - step_score: float
#   Puntuacion formal de la ultima estrofa evaluada.
# - score_history: list[float]
#   Historial de puntuaciones por iteracion.
# - feedback: str
#   Feedback formal que se pasa al LLM en la siguiente expansion.
# - metrics: dict[str, Any]
#   Metricas completas devueltas por evaluate_stanza_abba.
# - generation_reasoning: str
#   Breve explicacion textual asociada a la generacion.
# - meter_repair_report: dict[str, Any]
#   Informe opcional de la reparacion local de metrica aplicada antes del score.
# - outer_rhyme_diagnosis: dict[str, Any]
#   Diagnostico explicito de la rima exterior AXYA, versos 1 y 4.
# - inner_rhyme_diagnosis: dict[str, Any]
#   Diagnostico explicito de la rima interior B, versos 2 y 3.
# - outer_rhyme_repair_report: dict[str, Any]
#   Informe opcional de la reparacion de rima exterior A.


# =========================
# 3. Utilidad general Ollama
# =========================

def chat_ollama(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = TEMPERATURE,
    num_predict: int = NUM_PREDICT,
) -> str:
    """Llama a Ollama por HTTP y fuerza una respuesta JSON."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=240)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


# =========================
# 4. Limpieza y parseo JSON
# =========================

def clean_generated_verse(verse: str) -> str:
    """Limpia numeracion, comillas exteriores y espacios duplicados."""
    cleaned_verse = str(verse).strip()

    # Casos habituales: "1. verso", "1) verso", "1- verso", "- verso".
    cleaned_verse = re.sub(
        r"^\s*(?:[-*]\s+|\d+\s*(?:[.)]|-)\s*)",
        "",
        cleaned_verse,
    )

    quote_pairs = {
        '"': '"',
        "'": "'",
        "\u201c": "\u201d",
        "\u2018": "\u2019",
        "\u00ab": "\u00bb",
    }
    if len(cleaned_verse) >= 2:
        first_character = cleaned_verse[0]
        last_character = cleaned_verse[-1]
        if quote_pairs.get(first_character) == last_character:
            cleaned_verse = cleaned_verse[1:-1].strip()

    return " ".join(cleaned_verse.split())


def clean_generated_stanza(verses: list[str]) -> list[str]:
    """Limpia una lista de versos generados y descarta versos vacios."""
    cleaned_verses = []
    for verse in verses:
        cleaned_verse = clean_generated_verse(verse)
        if cleaned_verse:
            cleaned_verses.append(cleaned_verse)
    return cleaned_verses


def _parse_stanza_payload(payload: Dict[str, Any]) -> list[str]:
    """Extrae la lista de versos desde un objeto JSON ya parseado."""
    verses = payload.get("verses")
    if not isinstance(verses, list):
        raise ValueError("El campo 'verses' no existe o no es una lista.")

    return clean_generated_stanza([str(verse) for verse in verses])


def parse_stanza_candidates_response(
    raw_response: str,
    num_candidates: int,
) -> list[dict[str, Any]]:
    """Parsea la respuesta JSON con candidatos de estrofa.

    No exige exactamente 4 versos en el parseo. Esa restriccion se evalua
    despues con evaluate_stanza_abba para que el Beam Search pueda penalizar
    salidas imperfectas en lugar de descartarlas siempre.
    """
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta del modelo no es un objeto JSON.")

    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("El campo 'candidates' no existe o no es una lista.")

    parsed_candidates: list[dict[str, Any]] = []
    for item in candidates[:num_candidates]:
        if not isinstance(item, dict):
            continue

        try:
            stanza = _parse_stanza_payload(item)
        except ValueError:
            continue
        if not stanza:
            continue

        generation_reasoning = str(
            item.get("generation_reasoning", "Sin explicacion.")
        ).strip()

        parsed_candidates.append(
            {
                "stanza": stanza,
                "generation_reasoning": (
                    generation_reasoning if generation_reasoning else "Sin explicacion."
                ),
            }
        )

    if not parsed_candidates:
        raise ValueError("No se ha podido extraer ningun candidato valido.")

    return parsed_candidates


# =========================
# 5. Construccion del prompt
# =========================

def format_numbered_verses(verses: list[str] | None) -> str:
    """Formatea una estrofa previa con numeracion para incluirla en el prompt."""
    if not verses:
        return "[sin versos]"
    return "\n".join(
        f"{verse_number}. {verse}"
        for verse_number, verse in enumerate(verses, start=1)
    )


def summarize_feedback_for_prompt(
    feedback: str | None,
    max_chars: int = 1800,
) -> str:
    """Recorta el feedback para mantener el prompt compacto."""
    clean_feedback = str(feedback or "").strip()
    if len(clean_feedback) <= max_chars:
        return clean_feedback
    return f"{clean_feedback[:max_chars].rstrip()}\n[feedback recortado]"


def build_constraint_protection_prompt(
    previous_beam: dict[str, Any] | None,
) -> str:
    """Construye instrucciones para conservar restricciones ya satisfechas."""
    if not ENABLE_PROMPT_CONSTRAINT_PROTECTION:
        return ""

    if not isinstance(previous_beam, dict):
        return ""

    previous_stanza = previous_beam.get("stanza")
    if not isinstance(previous_stanza, list) or not previous_stanza:
        return ""

    lines = []

    outer_rhyme_diagnosis = previous_beam.get("outer_rhyme_diagnosis", {})
    if (
        isinstance(outer_rhyme_diagnosis, dict)
        and outer_rhyme_diagnosis.get("is_valid", False)
    ):
        lines.extend(
            [
                "La rima exterior A ya es correcta.",
                "Conserva el verso 1 y el verso 4 siempre que sea posible.",
                "Conserva especialmente las palabras finales del verso 1 y del verso 4.",
            ]
        )

    inner_rhyme_diagnosis = previous_beam.get("inner_rhyme_diagnosis", {})
    if (
        isinstance(inner_rhyme_diagnosis, dict)
        and inner_rhyme_diagnosis.get("is_valid", False)
    ):
        lines.extend(
            [
                "La rima interior B ya es correcta.",
                "Conserva el verso 2 y el verso 3 siempre que sea posible.",
                "Conserva especialmente las palabras finales del verso 2 y del verso 3.",
            ]
        )

    metrics = previous_beam.get("metrics", {})
    syllables = metrics.get("syllables", {}) if isinstance(metrics, dict) else {}
    if isinstance(syllables, dict):
        correct_verses = syllables.get("correct_verses")
        total_expected_verses = syllables.get("total_expected_verses")
        if (
            isinstance(correct_verses, int)
            and isinstance(total_expected_verses, int)
            and total_expected_verses > 0
        ):
            if correct_verses == total_expected_verses:
                lines.extend(
                    [
                        "Todos los versos ya tienen 11 silabas metricas.",
                        "No alargues ni acortes versos que ya cumplen la metrica.",
                    ]
                )
            elif correct_verses > 0:
                lines.append(
                    "Conserva los versos que ya tienen 11 silabas metricas siempre que sea posible."
                )

    return "\n".join(lines)


def _build_stanza_generation_messages(
    question: str,
    previous_stanza: list[str] | None,
    feedback: str | None,
    num_candidates: int,
    strict: bool = False,
    previous_beam: dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """Construye los mensajes para pedir estrofas candidatas a Ollama."""
    strict_rules = ""
    if strict:
        strict_rules = (
            "\nModo estricto de reintento:\n"
            "- Devuelve solamente JSON valido.\n"
            "- No incluyas markdown, titulo, comentarios ni texto fuera del JSON.\n"
            "- Devuelve todos los corchetes y llaves de cierre.\n"
            "- Cada candidato debe contener las claves exactas "
            '"verses" y "generation_reasoning".\n'
        )

    rhyme_rules = (
        "REGLA DE RIMA ABBA:\n"
        "- Verso 1 termina con rima A.\n"
        "- Verso 2 termina con rima B.\n"
        "- Verso 3 termina con rima B.\n"
        "- Verso 4 termina con rima A.\n"
        "- El verso 1 debe rimar consonantemente con el verso 4.\n"
        "- El verso 2 debe rimar consonantemente con el verso 3.\n"
        "- Evita repetir exactamente la misma palabra final en versos que riman.\n"
        "- El verso 4 debe rimar con el verso 1, pero preferiblemente con una palabra final distinta.\n"
    )

    construction_strategy = (
        "ESTRATEGIA DE CONSTRUCCION:\n"
        "- Antes de escribir, decide internamente dos rimas consonantes: A y B.\n"
        "- Usa la rima A solo en los versos 1 y 4.\n"
        "- Usa la rima B solo en los versos 2 y 3.\n"
        "- No expliques esta decision; solo aplicala en los versos.\n"
    )

    verse_length_rules = (
        "LONGITUD DE VERSO:\n"
        "- Escribe versos breves y compactos.\n"
        "- Cada verso debe tender a 11 silabas metricas.\n"
        "- Evita frases largas, explicativas o narrativas.\n"
        "- Prioriza estructura, metrica y rima antes que belleza estetica.\n"
    )

    json_format_rules = (
        "FORMATO JSON OBLIGATORIO:\n"
        "Devuelve exactamente un objeto JSON con esta forma:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "verses": [\n'
        '        "verso 1",\n'
        '        "verso 2",\n'
        '        "verso 3",\n'
        '        "verso 4"\n'
        "      ],\n"
        '      "generation_reasoning": "explicacion breve"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "No incluyas texto antes ni despues del JSON.\n"
    )

    system_prompt = (
        "Eres un poeta y asistente de generacion controlada dentro de un "
        "sistema de Beam Search.\n"
        "Debes proponer estrofas clasicas en espanol.\n"
        "No anadas explicaciones fuera del JSON.\n"
        "Debes devolver JSON valido.\n\n"
        "RESTRICCIONES FORMALES PRIORITARIAS:\n"
        "- Exactamente 4 versos.\n"
        "- Cada verso debe intentar tener 11 silabas metricas.\n"
        "- Rima consonante ABBA.\n"
        "- Sin titulo.\n"
        "- Sin numeracion dentro de los versos.\n"
        "- Cada verso debe ser una cadena independiente.\n"
        "- La unidad generada siempre es la estrofa completa.\n"
        "- No generes un soneto completo.\n"
        "- generation_reasoning debe ser breve.\n\n"
        f"{construction_strategy}\n"
        f"{rhyme_rules}\n"
        f"{verse_length_rules}\n"
        f"{strict_rules}\n"
        f"{json_format_rules}"
    )

    if previous_stanza:
        summarized_feedback = summarize_feedback_for_prompt(feedback)
        constraint_protection = build_constraint_protection_prompt(previous_beam)
        constraint_protection_section = ""
        if constraint_protection:
            constraint_protection_section = (
                "RESTRICCIONES YA SATISFECHAS QUE DEBES CONSERVAR:\n"
                f"{constraint_protection}\n\n"
            )

        user_prompt = (
            f"Tarea original:\n{question}\n\n"
            "Estrofa candidata anterior, verso a verso:\n"
            f"{format_numbered_verses(previous_stanza)}\n\n"
            "Feedback formal recibido:\n"
            f"{summarized_feedback}\n\n"
            f"{constraint_protection_section}"
            f"Genera exactamente {num_candidates} versiones completas corregidas "
            "o mejoradas de la estrofa anterior.\n\n"
            "Instrucciones de correccion:\n"
            "- Devuelve siempre una estrofa completa de 4 versos.\n"
            "- Intenta que cada verso tenga 11 silabas metricas.\n"
            "- Respeta la rima consonante ABBA.\n"
            "- Corrige especificamente los errores indicados en el feedback.\n"
            "- Si un verso tiene demasiadas silabas, acortalo.\n"
            "- Si un verso tiene pocas silabas, alargalo ligeramente.\n"
            "- Si falla la rima, cambia la palabra final del verso afectado.\n"
            "- No repitas literalmente la estrofa anterior si contiene errores.\n"
            "- No anadas titulo ni texto fuera del JSON.\n\n"
            f"{construction_strategy}\n"
            f"{rhyme_rules}\n"
            f"{verse_length_rules}\n"
            f"{json_format_rules}"
            "Responde solo con JSON valido."
        )
    else:
        user_prompt = (
            f"Tarea original:\n{question}\n\n"
            f"Genera exactamente {num_candidates} primeras versiones candidatas "
            "de una estrofa desde cero.\n"
            "Cada candidato debe tener exactamente 4 versos, tender a versos "
            "endecasilabos y usar rima consonante ABBA.\n\n"
            f"{construction_strategy}\n"
            f"{rhyme_rules}\n"
            f"{verse_length_rules}\n"
            f"{json_format_rules}"
            "Responde solo con JSON valido."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =========================
# 6. Generacion con Ollama
# =========================

def generate_stanza_candidates_with_ollama(
    question: str,
    previous_stanza: list[str] | None,
    feedback: str | None,
    num_candidates: int = 3,
    previous_beam: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Genera estrofas candidatas usando Ollama.

    Hace un primer intento con la temperatura principal y un reintento mas
    conservador si falla el parseo JSON. Si ambos fallan, devuelve candidatos
    vacios para que el grafo pueda terminar de forma controlada.
    """
    attempts = [
        {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "strict": False,
        },
        {
            "temperature": RETRY_TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "strict": True,
        },
    ]
    errors = []

    for attempt_number, attempt in enumerate(attempts, start=1):
        raw_response = ""
        messages = _build_stanza_generation_messages(
            question=question,
            previous_stanza=previous_stanza,
            feedback=feedback,
            num_candidates=num_candidates,
            strict=attempt["strict"],
            previous_beam=previous_beam,
        )

        try:
            raw_response = chat_ollama(
                model=GENERATION_MODEL,
                messages=messages,
                temperature=attempt["temperature"],
                num_predict=attempt["num_predict"],
            )
            return parse_stanza_candidates_response(raw_response, num_candidates)
        except Exception as exc:
            error_message = f"Intento {attempt_number}: {exc}"
            errors.append(error_message)
            print(f"[WARN] Fallo generando estrofas con Ollama. {error_message}")
            if raw_response:
                print("[WARN] Respuesta recibida del generador:")
                print(raw_response)

    fallback_reason = (
        "Fallback: no se pudieron generar candidatos de estrofa con Ollama. "
        f"Errores: {' | '.join(errors)}"
    )
    return [
        {
            "stanza": [],
            "generation_reasoning": fallback_reason,
        }
        for _ in range(num_candidates)
    ]


# =========================
# 7. Reparacion local de metrica
# =========================

def syllable_distance_to_target(syllable_count: int) -> int:
    """Mide la distancia absoluta entre un conteo y el objetivo metrico."""
    return abs(syllable_count - TARGET_SYLLABLES_PER_VERSE)


def parse_meter_repair_variants_response(
    raw_response: str,
    max_variants: int,
) -> list[str]:
    """Parsea variantes de un unico verso devueltas por el LLM.

    El formato esperado es {"variants": ["...", "..."]}. Cada variante se
    limpia con la misma funcion usada para versos generados.
    """
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta del reparador no es un objeto JSON.")

    variants = parsed.get("variants")
    if not isinstance(variants, list):
        raise ValueError("El campo 'variants' no existe o no es una lista.")

    cleaned_variants = []
    seen_variants = set()
    for variant in variants[:max_variants]:
        cleaned_variant = clean_generated_verse(str(variant))
        if not cleaned_variant or cleaned_variant in seen_variants:
            continue
        seen_variants.add(cleaned_variant)
        cleaned_variants.append(cleaned_variant)

    if not cleaned_variants:
        raise ValueError("No se ha podido extraer ninguna variante valida.")

    return cleaned_variants


def _build_meter_repair_messages(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    num_variants: int,
) -> List[Dict[str, str]]:
    """Construye el prompt para reparar la metrica de un solo verso."""
    verse_number = verse_index + 1
    original_verse = stanza[verse_index]
    target = TARGET_SYLLABLES_PER_VERSE
    direction = (
        "acortarlo"
        if original_syllables > target
        else "alargarlo ligeramente"
    )

    system_prompt = (
        "Eres un asistente de reparacion metrica para poesia espanola.\n"
        "Tu tarea es proponer variantes de un unico verso.\n"
        "Debes centrarte solo en aproximar el verso a 11 silabas metricas.\n"
        "No evalues belleza poetica ni expliques el resultado.\n"
        "No intentes reparar la rima de la estrofa en esta fase.\n"
        "Si puedes, conserva la palabra final para no alterar la rima existente.\n"
        "Devuelve solamente JSON valido.\n\n"
        "FORMATO JSON OBLIGATORIO:\n"
        "{\n"
        '  "variants": [\n'
        '    "variante 1",\n'
        '    "variante 2"\n'
        "  ]\n"
        "}\n"
        "No incluyas texto antes ni despues del JSON."
    )

    user_prompt = (
        f"Tarea original:\n{question}\n\n"
        "Estrofa completa como contexto:\n"
        f"{format_numbered_verses(stanza)}\n\n"
        f"Verso que hay que reparar: {verse_number}\n"
        f"Verso original: {original_verse}\n"
        f"Conteo metrico aproximado actual: {original_syllables}\n"
        f"Objetivo: {target} silabas metricas.\n\n"
        f"Genera exactamente {num_variants} variantes del mismo verso para "
        f"{direction} y acercarlo a {target} silabas metricas.\n"
        "Condiciones:\n"
        "- Devuelve solo variantes de ese verso, no la estrofa completa.\n"
        "- No numeres las variantes.\n"
        "- No anadas titulo.\n"
        "- Manten el sentido general del verso.\n"
        "- Si es posible, conserva la palabra final del verso original.\n"
        "- Responde solo con JSON valido."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_meter_repair_variants_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    num_variants: int = METER_REPAIR_VARIANTS_PER_VERSE,
) -> list[str]:
    """Pide al LLM variantes de un unico verso para reparar su metrica."""
    raw_response = ""
    messages = _build_meter_repair_messages(
        question=question,
        stanza=stanza,
        verse_index=verse_index,
        original_syllables=original_syllables,
        num_variants=num_variants,
    )

    raw_response = chat_ollama(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=METER_REPAIR_TEMPERATURE,
        num_predict=METER_REPAIR_NUM_PREDICT,
    )
    return parse_meter_repair_variants_response(raw_response, num_variants)


def _build_meter_repair_messages_preserving_final_word(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    required_final_word: str,
    num_variants: int,
) -> List[Dict[str, str]]:
    """Construye el prompt metrico conservando una palabra final exacta."""
    verse_number = verse_index + 1
    original_verse = stanza[verse_index]
    target = TARGET_SYLLABLES_PER_VERSE
    direction = (
        "acortarlo"
        if original_syllables > target
        else "alargarlo ligeramente"
    )

    system_prompt = (
        "Eres un asistente de reparacion metrica para poesia espanola.\n"
        "Tu tarea es proponer variantes de un unico verso.\n"
        "Debes centrarte solo en aproximar el verso a 11 silabas metricas.\n"
        "Debes conservar exactamente la palabra final obligatoria.\n"
        "No evalues belleza poetica ni expliques el resultado.\n"
        "Devuelve solamente JSON valido.\n\n"
        "FORMATO JSON OBLIGATORIO:\n"
        "{\n"
        '  "variants": [\n'
        '    "variante 1",\n'
        '    "variante 2"\n'
        "  ]\n"
        "}\n"
        "No incluyas texto antes ni despues del JSON."
    )

    user_prompt = (
        f"Tarea original:\n{question}\n\n"
        "Estrofa completa como contexto:\n"
        f"{format_numbered_verses(stanza)}\n\n"
        f"Verso que hay que reparar: {verse_number}\n"
        f"Verso original: {original_verse}\n"
        f"Conteo metrico aproximado actual: {original_syllables}\n"
        f"Objetivo: {target} silabas metricas.\n"
        f"La palabra final obligatoria es: {required_final_word}\n\n"
        f"Genera exactamente {num_variants} variantes del mismo verso para "
        f"{direction} y acercarlo a {target} silabas metricas.\n"
        "Condiciones:\n"
        "- Reescribe solo el verso indicado.\n"
        "- Devuelve solo variantes de ese verso, no la estrofa completa.\n"
        "- No modifiques ningun otro verso.\n"
        "- No numeres las variantes.\n"
        "- No anadas titulo.\n"
        "- Manten el sentido general del verso.\n"
        f"- Todas las variantes deben terminar exactamente en {required_final_word}.\n"
        "- No uses una forma derivada, flexionada o parecida de "
        f"{required_final_word}.\n"
        f"- No escribas ninguna palabra despues de {required_final_word}.\n"
        "- No cambies la palabra final aunque eso limite la calidad poetica.\n"
        "- Responde solo con JSON valido."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_meter_repair_variants_preserving_final_word_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    original_syllables: int,
    required_final_word: str,
    num_variants: int = POST_B_RHYME_METER_REPAIR_VARIANTS,
    temperature: float = POST_B_RHYME_METER_REPAIR_TEMPERATURE,
    num_predict: int = POST_B_RHYME_METER_REPAIR_NUM_PREDICT,
) -> list[str]:
    """Pide variantes metricas conservando una palabra final exacta."""
    raw_response = ""
    messages = _build_meter_repair_messages_preserving_final_word(
        question=question,
        stanza=stanza,
        verse_index=verse_index,
        original_syllables=original_syllables,
        required_final_word=required_final_word,
        num_variants=num_variants,
    )

    raw_response = chat_ollama(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=temperature,
        num_predict=num_predict,
    )
    return parse_meter_repair_variants_response(raw_response, num_variants)


def build_meter_repair_report(enabled: bool, stanza: list[str]) -> dict[str, Any]:
    """Crea la estructura base del informe de reparacion metrica."""
    return {
        "enabled": enabled,
        "changed": False,
        "original_stanza": stanza.copy(),
        "repaired_stanza": stanza.copy(),
        "verse_repairs": [],
    }


def build_post_rhyme_meter_repair_report(
    enabled: bool,
    stanza: list[str],
    verse_index: int,
    required_final_word: str,
    repair_label: str,
) -> dict[str, Any]:
    """Crea la estructura base de una post-reparacion metrica de rima."""
    original_verse = (
        stanza[verse_index]
        if 0 <= verse_index < len(stanza)
        else None
    )

    return {
        "enabled": enabled,
        "repair_label": repair_label,
        "attempted": False,
        "changed": False,
        "reason": None,
        "original_stanza": stanza.copy(),
        "repaired_stanza": stanza.copy(),
        "verse_number": verse_index + 1,
        "required_final_word": required_final_word,
        "original_verse": original_verse,
        "original_syllables": None,
        "original_distance": None,
        "selected_variant": None,
        "selected_syllables": None,
        "selected_distance": None,
        "variants": [],
        "error": None,
    }


def repair_verse_meter_preserving_final_word_with_ollama(
    question: str,
    stanza: list[str],
    verse_index: int,
    required_final_word: str,
    enabled: bool,
    num_variants: int,
    temperature: float,
    num_predict: int,
    repair_label: str,
) -> tuple[list[str], dict[str, Any]]:
    """Repara la metrica de un verso sin cambiar su palabra final."""
    report = build_post_rhyme_meter_repair_report(
        enabled=enabled,
        stanza=stanza,
        verse_index=verse_index,
        required_final_word=required_final_word,
        repair_label=repair_label,
    )
    repaired_stanza = stanza.copy()

    if not enabled:
        report["reason"] = "Reparacion metrica posterior a rima desactivada."
        return repaired_stanza, report

    if not 0 <= verse_index < len(stanza):
        report["reason"] = "La estrofa no contiene el verso indicado."
        return repaired_stanza, report

    original_verse = stanza[verse_index]
    original_syllables = count_verse_syllables(original_verse)
    original_distance = syllable_distance_to_target(original_syllables)
    report["original_syllables"] = original_syllables
    report["original_distance"] = original_distance

    if original_distance == 0:
        report["reason"] = "El verso ya tiene 11 silabas metricas."
        report["repaired_stanza"] = repaired_stanza.copy()
        return repaired_stanza, report

    report["attempted"] = True

    try:
        variants = generate_meter_repair_variants_preserving_final_word_with_ollama(
            question=question,
            stanza=stanza,
            verse_index=verse_index,
            original_syllables=original_syllables,
            required_final_word=required_final_word,
            num_variants=num_variants,
            temperature=temperature,
            num_predict=num_predict,
        )
    except Exception as exc:
        report["error"] = str(exc)
        report["reason"] = (
            "Fallo generando variantes metricas posteriores a la rima."
        )
        return repaired_stanza, report

    best_variant = None
    best_syllables = original_syllables
    best_distance = original_distance

    for variant in variants:
        variant_syllables = count_verse_syllables(variant)
        variant_distance = syllable_distance_to_target(variant_syllables)
        variant_final_word = get_last_word(variant)
        preserves_final_word = variant_final_word == required_final_word
        improved = preserves_final_word and variant_distance < original_distance

        variant_info = {
            "verse": variant,
            "syllables": variant_syllables,
            "distance": variant_distance,
            "final_word": variant_final_word,
            "preserves_final_word": preserves_final_word,
            "improved": improved,
            "accepted": False,
            "rejection_reason": None,
        }

        if not preserves_final_word:
            variant_info["rejection_reason"] = (
                "No conserva exactamente la palabra final obligatoria."
            )
        elif not improved:
            variant_info["rejection_reason"] = (
                "No mejora la distancia metrica del verso."
            )
        elif variant_distance < best_distance:
            best_variant = variant
            best_syllables = variant_syllables
            best_distance = variant_distance

        report["variants"].append(variant_info)

    if best_variant is None:
        report["reason"] = (
            "Ninguna variante conserva la palabra final y mejora la metrica."
        )
        report["repaired_stanza"] = repaired_stanza.copy()
        return repaired_stanza, report

    repaired_stanza[verse_index] = best_variant
    report["changed"] = True
    report["reason"] = (
        "Se reparo la metrica conservando la palabra final obligatoria."
    )
    report["selected_variant"] = best_variant
    report["selected_syllables"] = best_syllables
    report["selected_distance"] = best_distance
    report["repaired_stanza"] = repaired_stanza.copy()

    for variant_info in report["variants"]:
        if variant_info["verse"] == best_variant:
            variant_info["accepted"] = True
            variant_info["rejection_reason"] = None
            break

    return repaired_stanza, report


def repair_stanza_meter_with_ollama(
    question: str,
    stanza: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Intenta reparar localmente versos que no sean endecasilabos.

    La sustitucion se acepta solo si la mejor variante reduce la distancia al
    objetivo de 11 silabas. Si no hay mejora, se conserva el verso original.
    """
    report = build_meter_repair_report(ENABLE_LOCAL_METER_REPAIR, stanza)
    if not ENABLE_LOCAL_METER_REPAIR or not stanza:
        return stanza.copy(), report

    repaired_stanza = stanza.copy()

    for verse_index, original_verse in enumerate(stanza):
        original_syllables = count_verse_syllables(original_verse)
        original_distance = syllable_distance_to_target(original_syllables)

        if original_distance == 0:
            report["verse_repairs"].append(
                {
                    "verse_number": verse_index + 1,
                    "original_verse": original_verse,
                    "original_syllables": original_syllables,
                    "original_distance": original_distance,
                    "attempted": False,
                    "reason": "El verso ya tiene 11 silabas metricas.",
                    "improved": False,
                    "selected_variant": None,
                    "selected_syllables": None,
                    "variants": [],
                }
            )
            continue

        verse_report = {
            "verse_number": verse_index + 1,
            "original_verse": original_verse,
            "original_syllables": original_syllables,
            "original_distance": original_distance,
            "attempted": True,
            "improved": False,
            "selected_variant": None,
            "selected_syllables": None,
            "selected_distance": None,
            "variants": [],
            "error": None,
        }

        try:
            variants = generate_meter_repair_variants_with_ollama(
                question=question,
                stanza=repaired_stanza,
                verse_index=verse_index,
                original_syllables=original_syllables,
            )
        except Exception as exc:
            verse_report["error"] = str(exc)
            report["verse_repairs"].append(verse_report)
            print(
                "[WARN] Fallo reparando metrica del verso "
                f"{verse_index + 1}: {exc}"
            )
            continue

        best_variant = None
        best_syllables = original_syllables
        best_distance = original_distance

        for variant in variants:
            variant_syllables = count_verse_syllables(variant)
            variant_distance = syllable_distance_to_target(variant_syllables)
            variant_info = {
                "verse": variant,
                "syllables": variant_syllables,
                "distance": variant_distance,
            }
            verse_report["variants"].append(variant_info)

            if variant_distance < best_distance:
                best_variant = variant
                best_syllables = variant_syllables
                best_distance = variant_distance

        if best_variant is not None:
            repaired_stanza[verse_index] = best_variant
            report["changed"] = True
            verse_report["improved"] = True
            verse_report["selected_variant"] = best_variant
            verse_report["selected_syllables"] = best_syllables
            verse_report["selected_distance"] = best_distance

        report["verse_repairs"].append(verse_report)

    report["repaired_stanza"] = repaired_stanza.copy()
    return repaired_stanza, report


# =========================
# 8. Reparacion de rima exterior A
# =========================

def parse_rhyme_repair_variants_response(
    raw_response: str,
    max_variants: int,
) -> list[str]:
    """Parsea variantes JSON de reparacion de rima para un unico verso."""
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta del reparador de rima no es un objeto JSON.")

    variants = parsed.get("variants")
    if not isinstance(variants, list):
        raise ValueError("El campo 'variants' no existe o no es una lista.")

    cleaned_variants = []
    seen_variants = set()
    for variant in variants[:max_variants]:
        cleaned_variant = clean_generated_verse(str(variant))
        if not cleaned_variant or cleaned_variant in seen_variants:
            continue
        seen_variants.add(cleaned_variant)
        cleaned_variants.append(cleaned_variant)

    if not cleaned_variants:
        raise ValueError("No se ha podido extraer ninguna variante valida.")

    return cleaned_variants


def parse_outer_rhyme_repair_variants_response(
    raw_response: str,
    max_variants: int,
) -> list[str]:
    """Parsea variantes del verso 4 para reparar la rima exterior A."""
    return parse_rhyme_repair_variants_response(raw_response, max_variants)


RHYME_HINT_EXAMPLES = {
    "eva": ["lleva", "nueva", "eleva", "conlleva", "nieva"],
    "argos": ["amargos", "largos"],
    "argo": ["amargo", "largo"],
    "oria": ["memoria", "historia", "gloria"],
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
}


def get_candidate_final_words_for_rhyme(
    anchor_word: str | None,
    target_rhyme: str | None,
) -> list[str]:
    """Devuelve palabras finales candidatas para una rima objetivo."""
    clean_anchor_word = str(anchor_word or "").strip()
    clean_target_rhyme = str(target_rhyme or "").strip()

    if ENABLE_RHYME_HINT_EXAMPLES:
        candidates = list(RHYME_HINT_EXAMPLES.get(clean_target_rhyme, []))
    else:
        candidates = []

    if clean_anchor_word and clean_anchor_word not in candidates:
        candidates.append(clean_anchor_word)

    deduped_candidates = []
    seen_candidates = set()
    for candidate in candidates:
        clean_candidate = str(candidate or "").strip()
        if not clean_candidate or clean_candidate in seen_candidates:
            continue
        seen_candidates.add(clean_candidate)
        deduped_candidates.append(clean_candidate)

    return deduped_candidates


def build_rhyme_target_hint(
    anchor_word: str | None,
    target_rhyme: str | None,
    anchor_verse_label: str = "verso 1",
    repaired_verse_label: str = "verso 4",
) -> str:
    """Construye una ayuda textual sobre palabras reales con la rima objetivo."""
    clean_anchor_word = str(anchor_word or "").strip()
    clean_target_rhyme = str(target_rhyme or "").strip()
    clean_anchor_verse_label = str(anchor_verse_label or "verso ancla").strip()
    clean_repaired_verse_label = str(
        repaired_verse_label or "verso reparable"
    ).strip()
    candidate_words = get_candidate_final_words_for_rhyme(
        anchor_word=clean_anchor_word,
        target_rhyme=clean_target_rhyme,
    )
    non_anchor_candidate_words = [
        word for word in candidate_words if word != clean_anchor_word
    ]

    lines = [
        f'La rima objetivo "{clean_target_rhyme}" es una terminacion de rima, '
        "no una palabra que debas copiar literalmente.",
        f"El {clean_repaired_verse_label} debe terminar en una palabra real "
        "que tenga esa rima consonante.",
    ]

    if clean_anchor_word:
        lines.append(
            f'El {clean_anchor_verse_label} termina en "{clean_anchor_word}"; '
            f'el {clean_repaired_verse_label} no tiene que terminar '
            f'necesariamente en "{clean_anchor_word}", pero si en una palabra '
            "que rime consonantemente con ella."
        )
        lines.append(
            f'Evita que el {clean_repaired_verse_label} termine tambien en '
            f'"{clean_anchor_word}" si existe otra palabra real con la misma '
            "rima consonante."
        )

    if non_anchor_candidate_words:
        lines.append(
            f"Palabras finales candidatas para el {clean_repaired_verse_label}: "
            f"{', '.join(non_anchor_candidate_words)}."
        )
        lines.append(
            f"El nuevo {clean_repaired_verse_label} debe terminar exactamente "
            "en una de esas palabras."
        )
        lines.append(
            f'No escribas solo "{clean_target_rhyme}"; usa una palabra real '
            "completa."
        )
    elif candidate_words:
        lines.append(
            "La lista actual solo contiene la palabra final del "
            f"{clean_anchor_verse_label} como referencia; busca otra palabra "
            "real que comparta la rima consonante, si es posible."
        )
        lines.append(
            f'No escribas solo "{clean_target_rhyme}"; usa una palabra real '
            "completa."
        )
    else:
        lines.append(
            "Busca una palabra real que comparta la rima consonante "
            f'"{clean_target_rhyme}"'
            + (f' con "{clean_anchor_word}"' if clean_anchor_word else "")
            + "; no copies solo la terminacion."
        )

    return "\n".join(lines)


def _build_outer_rhyme_repair_messages(
    question: str,
    stanza: list[str],
    outer_rhyme_diagnosis: dict[str, Any],
    num_variants: int,
) -> List[Dict[str, str]]:
    """Construye el prompt para pedir variantes del verso 4."""
    verse_1 = outer_rhyme_diagnosis.get("verse_1", {})
    if not isinstance(verse_1, dict):
        verse_1 = {}

    verse_4 = outer_rhyme_diagnosis.get("verse_4", {})
    if not isinstance(verse_4, dict):
        verse_4 = {}

    verse_1_text = verse_1.get("text", "")
    verse_1_last_word = verse_1.get("last_word", "")
    target_rhyme = outer_rhyme_diagnosis.get("target_rhyme", "")
    current_rhyme = outer_rhyme_diagnosis.get("current_rhyme", "")
    current_verse_4 = verse_4.get("text", stanza[3] if len(stanza) >= 4 else "")
    rhyme_target_hint = build_rhyme_target_hint(
        anchor_word=verse_1_last_word,
        target_rhyme=target_rhyme,
    )
    candidate_final_words = get_candidate_final_words_for_rhyme(
        anchor_word=verse_1_last_word,
        target_rhyme=target_rhyme,
    )
    non_anchor_candidate_final_words = [
        word for word in candidate_final_words if word != verse_1_last_word
    ]
    enforced_candidate_final_words = non_anchor_candidate_final_words
    candidate_final_words_section = ""
    candidate_final_words_restrictions = ""
    if enforced_candidate_final_words:
        candidate_final_words_section = (
            "Palabras finales candidatas para el nuevo verso 4:\n"
            + "\n".join(f"- {word}" for word in enforced_candidate_final_words)
            + "\n\n"
            "Cada variante debe terminar exactamente en una de esas palabras.\n"
            "No uses otras palabras finales en este paso.\n"
            "No escribas texto despues de la palabra final.\n\n"
        )
        candidate_final_words_restrictions = (
            "- Cada variante debe terminar exactamente en una de las palabras finales candidatas.\n"
            "- No uses una palabra final distinta de la lista.\n"
            "- No escribas nada despues de la palabra final candidata.\n"
            "- La ultima palabra de cada variante debe ser una palabra de la lista.\n"
        )
    elif verse_1_last_word:
        candidate_final_words_section = (
            "No hay palabras finales candidatas distintas de la palabra final "
            f'del verso 1 ("{verse_1_last_word}") en la lista actual.\n'
            "Busca una palabra real distinta que comparta la misma rima "
            "consonante, si es posible.\n\n"
        )

    system_prompt = (
        "Eres un asistente de reparacion de rima consonante para poesia "
        "espanola.\n"
        "Tu tarea es proponer variantes de un unico verso: el verso 4.\n"
        "No reescribas la estrofa completa.\n"
        "No evalues belleza poetica ni expliques el resultado.\n"
        "Devuelve solamente JSON valido.\n\n"
        "FORMATO JSON OBLIGATORIO:\n"
        "{\n"
        '  "variants": [\n'
        '    "nuevo verso 4",\n'
        '    "nuevo verso 4 alternativo"\n'
        "  ]\n"
        "}\n"
        "No incluyas texto antes ni despues del JSON."
    )

    user_prompt = (
        f"Tarea original:\n{question}\n\n"
        "Estrofa completa como contexto:\n"
        f"{format_numbered_verses(stanza)}\n\n"
        "Verso 1 como ancla:\n"
        f"{verse_1_text}\n"
        f"Palabra final del verso 1: {verse_1_last_word}\n"
        f"Rima consonante objetivo del verso 1: {target_rhyme}\n\n"
        "Orientacion para la palabra final del verso 4:\n"
        f"{rhyme_target_hint}\n\n"
        f"{candidate_final_words_section}"
        "Verso 4 actual:\n"
        f"{current_verse_4}\n"
        f"Rima actual del verso 4: {current_rhyme}\n\n"
        f"Genera exactamente {num_variants} variantes para reescribir solo "
        "el verso 4.\n"
        "Restricciones:\n"
        "- Reescribe solo el verso 4.\n"
        "- No modifiques el verso 1.\n"
        "- No modifiques los versos 2 y 3.\n"
        "- El nuevo verso 4 debe rimar consonantemente con el verso 1.\n"
        f"- La rima objetivo es: {target_rhyme}.\n"
        "- Evita que el verso 4 termine con la misma palabra final que el verso 1 si existe alternativa.\n"
        "- La rima objetivo es una terminacion de rima, no una palabra obligatoria.\n"
        "- No copies literalmente la rima objetivo como palabra final si no es una palabra natural.\n"
        "- Termina el verso 4 con una palabra real que tenga esa rima consonante.\n"
        "- La palabra final del verso 4 debe ser una palabra completa y natural en espanol.\n"
        "- No fuerces terminaciones artificiales.\n"
        "- No termines el verso con fragmentos como \"eva\", \"argos\", \"ido\" o similares si no funcionan como palabra real.\n"
        f"{candidate_final_words_restrictions}"
        "- Intenta que el nuevo verso 4 tenga 11 silabas metricas.\n"
        "- Devuelve solo variantes del verso 4, no la estrofa completa.\n"
        "- No anadas titulo.\n"
        "- No numeres las variantes.\n"
        "- Responde solo con JSON valido."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_outer_rhyme_repair_variants_with_ollama(
    question: str,
    stanza: list[str],
    outer_rhyme_diagnosis: dict[str, Any],
    num_variants: int = OUTER_RHYME_REPAIR_VARIANTS,
) -> list[str]:
    """Pide al LLM variantes del verso 4 para reparar la rima exterior A."""
    messages = _build_outer_rhyme_repair_messages(
        question=question,
        stanza=stanza,
        outer_rhyme_diagnosis=outer_rhyme_diagnosis,
        num_variants=num_variants,
    )

    raw_response = chat_ollama(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=OUTER_RHYME_REPAIR_TEMPERATURE,
        num_predict=OUTER_RHYME_REPAIR_NUM_PREDICT,
    )
    return parse_rhyme_repair_variants_response(raw_response, num_variants)


def build_outer_rhyme_repair_report(
    enabled: bool,
    stanza: list[str],
) -> dict[str, Any]:
    """Crea la estructura base del informe de reparacion de rima A."""
    return {
        "enabled": enabled,
        "attempted": False,
        "changed": False,
        "reason": None,
        "original_stanza": stanza.copy(),
        "repaired_stanza": stanza.copy(),
        "target_rhyme": None,
        "current_rhyme": None,
        "verse_1": None,
        "candidate_final_words": [],
        "enforced_candidate_final_words": [],
        "post_a_meter_repair_enabled": ENABLE_POST_A_RHYME_METER_REPAIR,
        "post_a_meter_repair_attempts": 0,
        "post_a_meter_repair_successes": 0,
        "original_verse_4": None,
        "original_verse_4_syllables": None,
        "selected_variant": None,
        "selected_variant_syllables": None,
        "variants": [],
        "error": None,
    }


def repair_stanza_outer_rhyme_with_ollama(
    question: str,
    stanza: list[str],
    outer_rhyme_diagnosis: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Intenta reparar la rima exterior A reescribiendo solo el verso 4."""
    report = build_outer_rhyme_repair_report(ENABLE_OUTER_RHYME_REPAIR, stanza)
    repaired_stanza = stanza.copy()

    report["target_rhyme"] = outer_rhyme_diagnosis.get("target_rhyme")
    report["current_rhyme"] = outer_rhyme_diagnosis.get("current_rhyme")
    report["verse_1"] = outer_rhyme_diagnosis.get("verse_1")

    if not ENABLE_OUTER_RHYME_REPAIR:
        report["reason"] = "Reparacion de rima A desactivada."
        return repaired_stanza, report

    if len(stanza) != 4:
        report["reason"] = "La estrofa no tiene 4 versos."
        return repaired_stanza, report

    original_verse_4 = stanza[3]
    original_verse_4_syllables = count_verse_syllables(original_verse_4)
    original_distance = syllable_distance_to_target(original_verse_4_syllables)
    report["original_verse_4"] = original_verse_4
    report["original_verse_4_syllables"] = original_verse_4_syllables

    if outer_rhyme_diagnosis.get("is_valid", False):
        report["reason"] = "La rima exterior AXYA ya es correcta."
        return repaired_stanza, report

    target_rhyme = outer_rhyme_diagnosis.get("target_rhyme")
    if not target_rhyme:
        report["reason"] = "No hay rima objetivo para reparar la rima A."
        return repaired_stanza, report

    verse_1 = outer_rhyme_diagnosis.get("verse_1", {})
    anchor_word = verse_1.get("last_word") if isinstance(verse_1, dict) else None
    candidate_final_words = get_candidate_final_words_for_rhyme(
        anchor_word=anchor_word,
        target_rhyme=target_rhyme,
    )
    enforced_candidate_final_words = [
        word for word in candidate_final_words if word != anchor_word
    ]
    report["candidate_final_words"] = candidate_final_words
    report["enforced_candidate_final_words"] = enforced_candidate_final_words

    report["attempted"] = True

    try:
        variants = generate_outer_rhyme_repair_variants_with_ollama(
            question=question,
            stanza=stanza,
            outer_rhyme_diagnosis=outer_rhyme_diagnosis,
        )
    except Exception as exc:
        report["error"] = str(exc)
        report["reason"] = "Fallo generando variantes de rima exterior A."
        print(f"[WARN] Fallo reparando rima exterior A: {exc}")
        return repaired_stanza, report

    acceptable_variants = []

    for variant_index, variant in enumerate(variants):
        candidate_stanza = stanza.copy()
        candidate_stanza[3] = variant

        variant_diagnosis = diagnose_stanza_outer_rhyme(candidate_stanza)
        variant_syllables = count_verse_syllables(variant)
        variant_distance = syllable_distance_to_target(variant_syllables)
        outer_rhyme_valid = bool(variant_diagnosis.get("is_valid", False))
        metric_not_worse = variant_distance <= original_distance
        variant_final_word = get_last_word(variant)
        final_word_matches_anchor = bool(
            anchor_word and variant_final_word == anchor_word
        )
        uses_candidate_final_word = (
            not enforced_candidate_final_words
            or variant_final_word in enforced_candidate_final_words
        )
        is_acceptable = (
            outer_rhyme_valid
            and metric_not_worse
            and uses_candidate_final_word
        )

        variant_info = {
            "verse": variant,
            "syllables": variant_syllables,
            "final_word": variant_final_word,
            "final_word_matches_anchor": final_word_matches_anchor,
            "uses_candidate_final_word": uses_candidate_final_word,
            "outer_rhyme_valid": outer_rhyme_valid,
            "outer_rhyme_summary": summarize_outer_rhyme_diagnosis(
                variant_diagnosis
            ),
            "post_a_meter_repair_attempted": False,
            "post_a_meter_repair_report": None,
            "post_a_meter_repaired_verse": None,
            "post_a_meter_repaired_syllables": None,
            "post_a_meter_repaired_distance": None,
            "post_a_meter_repair_accepted": False,
            "accepted": False,
            "rejection_reason": None,
        }

        if is_acceptable:
            acceptable_variants.append(
                {
                    "index": variant_index,
                    "verse": variant,
                    "syllables": variant_syllables,
                    "distance": variant_distance,
                    "final_word_matches_anchor": final_word_matches_anchor,
                    "variant_info": variant_info,
                }
            )
        elif (
            outer_rhyme_valid
            and uses_candidate_final_word
            and not metric_not_worse
            and ENABLE_POST_A_RHYME_METER_REPAIR
        ):
            report["post_a_meter_repair_attempts"] += 1
            variant_info["post_a_meter_repair_attempted"] = True

            candidate_stanza_for_meter = stanza.copy()
            candidate_stanza_for_meter[3] = variant
            post_meter_stanza, post_meter_report = (
                repair_verse_meter_preserving_final_word_with_ollama(
                    question=question,
                    stanza=candidate_stanza_for_meter,
                    verse_index=3,
                    required_final_word=variant_final_word,
                    enabled=ENABLE_POST_A_RHYME_METER_REPAIR,
                    num_variants=POST_A_RHYME_METER_REPAIR_VARIANTS,
                    temperature=POST_A_RHYME_METER_REPAIR_TEMPERATURE,
                    num_predict=POST_A_RHYME_METER_REPAIR_NUM_PREDICT,
                    repair_label="post_a",
                )
            )
            variant_info["post_a_meter_repair_report"] = post_meter_report

            post_meter_verse = post_meter_stanza[3]
            post_meter_syllables = count_verse_syllables(post_meter_verse)
            post_meter_distance = syllable_distance_to_target(
                post_meter_syllables
            )
            post_meter_final_word = get_last_word(post_meter_verse)
            post_meter_diagnosis = diagnose_stanza_outer_rhyme(post_meter_stanza)
            post_meter_outer_rhyme_valid = bool(
                post_meter_diagnosis.get("is_valid", False)
            )
            post_meter_preserves_final_word = (
                post_meter_final_word == variant_final_word
            )
            post_meter_metric_not_worse = post_meter_distance <= original_distance
            post_meter_is_acceptable = (
                post_meter_outer_rhyme_valid
                and post_meter_preserves_final_word
                and post_meter_metric_not_worse
                and uses_candidate_final_word
            )

            variant_info["post_a_meter_repaired_verse"] = post_meter_verse
            variant_info["post_a_meter_repaired_syllables"] = (
                post_meter_syllables
            )
            variant_info["post_a_meter_repaired_distance"] = (
                post_meter_distance
            )

            if post_meter_is_acceptable:
                variant_info["post_a_meter_repair_accepted"] = True
                report["post_a_meter_repair_successes"] += 1
                acceptable_variants.append(
                    {
                        "index": variant_index,
                        "verse": post_meter_verse,
                        "syllables": post_meter_syllables,
                        "distance": post_meter_distance,
                        "final_word_matches_anchor": final_word_matches_anchor,
                        "variant_info": variant_info,
                    }
                )
            else:
                variant_info["rejection_reason"] = (
                    "Corrige A, pero la reparacion metrica posterior no "
                    "consigue una variante aceptable."
                )
        elif not uses_candidate_final_word:
            variant_info["rejection_reason"] = (
                "No termina en una palabra final candidata distinta del verso 1."
            )
        elif not outer_rhyme_valid:
            variant_info["rejection_reason"] = "No corrige la rima exterior AXYA."
        else:
            variant_info["rejection_reason"] = (
                "Empeora la distancia metrica del verso 4."
            )

        report["variants"].append(variant_info)

    if not acceptable_variants:
        report["reason"] = "Ninguna variante corrige AXYA sin empeorar metrica."
        report["repaired_stanza"] = repaired_stanza.copy()
        return repaired_stanza, report

    selected = min(
        acceptable_variants,
        key=lambda item: (
            item["final_word_matches_anchor"],
            item["distance"],
            item["index"],
        ),
    )
    selected["variant_info"]["accepted"] = True
    selected["variant_info"]["rejection_reason"] = None

    repaired_stanza[3] = selected["verse"]
    report["changed"] = True
    report["reason"] = "Se acepto una variante que corrige AXYA."
    report["selected_variant"] = selected["verse"]
    report["selected_variant_syllables"] = selected["syllables"]
    report["repaired_stanza"] = repaired_stanza.copy()
    return repaired_stanza, report


# =========================
# 9. Reparacion de rima interior B
# =========================

def _build_inner_rhyme_repair_messages(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    num_variants: int,
) -> List[Dict[str, str]]:
    """Construye el prompt para pedir variantes del verso 3."""
    verse_2 = inner_rhyme_diagnosis.get("verse_2", {})
    if not isinstance(verse_2, dict):
        verse_2 = {}

    verse_3 = inner_rhyme_diagnosis.get("verse_3", {})
    if not isinstance(verse_3, dict):
        verse_3 = {}

    verse_2_text = verse_2.get("text", "")
    verse_2_last_word = verse_2.get("last_word", "")
    target_rhyme = inner_rhyme_diagnosis.get("target_rhyme", "")
    current_rhyme = inner_rhyme_diagnosis.get("current_rhyme", "")
    current_verse_3 = verse_3.get("text", stanza[2] if len(stanza) >= 3 else "")
    rhyme_target_hint = build_rhyme_target_hint(
        anchor_word=verse_2_last_word,
        target_rhyme=target_rhyme,
        anchor_verse_label="verso 2",
        repaired_verse_label="verso 3",
    )
    candidate_final_words = get_candidate_final_words_for_rhyme(
        anchor_word=verse_2_last_word,
        target_rhyme=target_rhyme,
    )
    non_anchor_candidate_final_words = [
        word for word in candidate_final_words if word != verse_2_last_word
    ]
    enforced_candidate_final_words = non_anchor_candidate_final_words
    candidate_final_words_section = ""
    candidate_final_words_restrictions = ""
    if enforced_candidate_final_words:
        candidate_final_words_section = (
            "Palabras finales candidatas para el nuevo verso 3:\n"
            + "\n".join(f"- {word}" for word in enforced_candidate_final_words)
            + "\n\n"
            "Cada variante debe terminar exactamente en una de esas palabras.\n"
            "No uses otras palabras finales en este paso.\n"
            "No escribas texto despues de la palabra final.\n\n"
        )
        candidate_final_words_restrictions = (
            "- Cada variante debe terminar exactamente en una de las palabras finales candidatas.\n"
            "- No uses una palabra final distinta de la lista.\n"
            "- No escribas nada despues de la palabra final candidata.\n"
            "- La ultima palabra de cada variante debe ser una palabra de la lista.\n"
        )
    elif verse_2_last_word:
        candidate_final_words_section = (
            "No hay palabras finales candidatas distintas de la palabra final "
            f'del verso 2 ("{verse_2_last_word}") en la lista actual.\n'
            "Busca una palabra real distinta que comparta la misma rima "
            "consonante, si es posible.\n\n"
        )

    system_prompt = (
        "Eres un asistente de reparacion de rima consonante para poesia "
        "espanola.\n"
        "Tu tarea es proponer variantes de un unico verso: el verso 3.\n"
        "No reescribas la estrofa completa.\n"
        "No evalues belleza poetica ni expliques el resultado.\n"
        "Devuelve solamente JSON valido.\n\n"
        "FORMATO JSON OBLIGATORIO:\n"
        "{\n"
        '  "variants": [\n'
        '    "nuevo verso 3",\n'
        '    "nuevo verso 3 alternativo"\n'
        "  ]\n"
        "}\n"
        "No incluyas texto antes ni despues del JSON."
    )

    user_prompt = (
        f"Tarea original:\n{question}\n\n"
        "Estrofa completa como contexto:\n"
        f"{format_numbered_verses(stanza)}\n\n"
        "Verso 2 como ancla:\n"
        f"{verse_2_text}\n"
        f"Palabra final del verso 2: {verse_2_last_word}\n"
        f"Rima consonante objetivo del verso 2: {target_rhyme}\n\n"
        "Orientacion para la palabra final del verso 3:\n"
        f"{rhyme_target_hint}\n\n"
        f"{candidate_final_words_section}"
        "Verso 3 actual:\n"
        f"{current_verse_3}\n"
        f"Rima actual del verso 3: {current_rhyme}\n\n"
        f"Genera exactamente {num_variants} variantes para reescribir solo "
        "el verso 3.\n"
        "Restricciones:\n"
        "- Reescribe solo el verso 3.\n"
        "- No modifiques el verso 1.\n"
        "- No modifiques el verso 2.\n"
        "- No modifiques el verso 4.\n"
        "- El nuevo verso 3 debe rimar consonantemente con el verso 2.\n"
        f"- La rima objetivo es: {target_rhyme}.\n"
        "- Evita que el verso 3 termine con la misma palabra final que el verso 2 si existe alternativa.\n"
        "- La rima objetivo es una terminacion de rima, no una palabra obligatoria.\n"
        "- No copies literalmente la rima objetivo como palabra final si no es una palabra natural.\n"
        "- Termina el verso 3 con una palabra real que tenga esa rima consonante.\n"
        "- La palabra final del verso 3 debe ser una palabra completa y natural en espanol.\n"
        "- No fuerces terminaciones artificiales.\n"
        "- No termines el verso con fragmentos como \"eva\", \"argos\", \"ido\" o similares si no funcionan como palabra real.\n"
        f"{candidate_final_words_restrictions}"
        "- Intenta que el nuevo verso 3 tenga 11 silabas metricas.\n"
        "- Devuelve solo variantes del verso 3, no la estrofa completa.\n"
        "- No anadas titulo.\n"
        "- No numeres las variantes.\n"
        "- Responde solo con JSON valido."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_inner_rhyme_repair_variants_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    num_variants: int = INNER_RHYME_REPAIR_VARIANTS,
) -> list[str]:
    """Pide al LLM variantes del verso 3 para reparar la rima interior B."""
    messages = _build_inner_rhyme_repair_messages(
        question=question,
        stanza=stanza,
        inner_rhyme_diagnosis=inner_rhyme_diagnosis,
        num_variants=num_variants,
    )

    raw_response = chat_ollama(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=INNER_RHYME_REPAIR_TEMPERATURE,
        num_predict=INNER_RHYME_REPAIR_NUM_PREDICT,
    )
    return parse_rhyme_repair_variants_response(raw_response, num_variants)


def _build_inner_rhyme_repair_messages_for_final_word(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    required_final_word: str,
    num_variants: int,
) -> List[Dict[str, str]]:
    """Construye el prompt B obligando una palabra final concreta."""
    verse_2 = inner_rhyme_diagnosis.get("verse_2", {})
    if not isinstance(verse_2, dict):
        verse_2 = {}

    verse_3 = inner_rhyme_diagnosis.get("verse_3", {})
    if not isinstance(verse_3, dict):
        verse_3 = {}

    clean_required_final_word = str(required_final_word or "").strip()
    derived_form_hint = ""
    if clean_required_final_word.endswith(("ar", "er", "ir")):
        verb_stem = clean_required_final_word[:-2]
        derived_form = f"{verb_stem}a"
        gerund_form = f"{verb_stem}ando"
        derived_form_hint = (
            f'- No uses "{derived_form}", "{gerund_form}" '
            f'ni otra forma derivada de "{clean_required_final_word}".\n'
        )

    verse_2_text = verse_2.get("text", "")
    verse_2_last_word = verse_2.get("last_word", "")
    target_rhyme = inner_rhyme_diagnosis.get("target_rhyme", "")
    current_rhyme = inner_rhyme_diagnosis.get("current_rhyme", "")
    current_verse_3 = verse_3.get("text", stanza[2] if len(stanza) >= 3 else "")

    system_prompt = (
        "Eres un asistente de reparacion de rima consonante para poesia "
        "espanola.\n"
        "Tu tarea es proponer variantes de un unico verso: el verso 3.\n"
        "No reescribas la estrofa completa.\n"
        "No evalues belleza poetica ni expliques el resultado.\n"
        "Devuelve solamente JSON valido.\n\n"
        "FORMATO JSON OBLIGATORIO:\n"
        "{\n"
        '  "variants": [\n'
        '    "nuevo verso 3",\n'
        '    "nuevo verso 3 alternativo"\n'
        "  ]\n"
        "}\n"
        "No incluyas texto antes ni despues del JSON."
    )

    user_prompt = (
        f"Tarea original:\n{question}\n\n"
        "Estrofa completa como contexto:\n"
        f"{format_numbered_verses(stanza)}\n\n"
        "Verso 2 como ancla:\n"
        f"{verse_2_text}\n"
        f"Palabra final del verso 2: {verse_2_last_word}\n"
        f"Rima consonante objetivo del verso 2: {target_rhyme}\n\n"
        "Verso 3 actual:\n"
        f"{current_verse_3}\n"
        f"Rima actual del verso 3: {current_rhyme}\n\n"
        "Palabra final obligatoria para todas las variantes: "
        f"{clean_required_final_word}\n\n"
        f'Todas las variantes deben terminar exactamente en "{clean_required_final_word}".\n'
        f'La ultima palabra debe ser exactamente "{clean_required_final_word}".\n'
        f'No escribas nada despues de "{clean_required_final_word}".\n'
        f"No uses una forma flexionada o derivada de "
        f'"{clean_required_final_word}".\n\n'
        f"Genera exactamente {num_variants} variantes para reescribir solo "
        "el verso 3.\n"
        "Restricciones:\n"
        "- Reescribe solo el verso 3.\n"
        "- No modifiques el verso 1.\n"
        "- No modifiques el verso 2.\n"
        "- No modifiques el verso 4.\n"
        "- El nuevo verso 3 debe rimar consonantemente con el verso 2.\n"
        f"- La palabra final obligatoria es: {clean_required_final_word}.\n"
        f'- Todas las variantes deben terminar exactamente en "{clean_required_final_word}".\n'
        "- No uses una forma flexionada o derivada de la palabra final obligatoria.\n"
        f"{derived_form_hint}"
        f'- No escribas nada despues de "{clean_required_final_word}".\n'
        "- Intenta que el nuevo verso 3 tenga 11 silabas metricas.\n"
        "- Devuelve solo variantes del verso 3, no la estrofa completa.\n"
        "- No anadas titulo.\n"
        "- No numeres las variantes.\n"
        "- Responde solo con JSON valido."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_inner_rhyme_repair_variants_for_final_word_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    required_final_word: str,
    num_variants: int = INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE,
) -> list[str]:
    """Pide variantes del verso 3 terminando en una palabra concreta."""
    messages = _build_inner_rhyme_repair_messages_for_final_word(
        question=question,
        stanza=stanza,
        inner_rhyme_diagnosis=inner_rhyme_diagnosis,
        required_final_word=required_final_word,
        num_variants=num_variants,
    )

    raw_response = chat_ollama(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=INNER_RHYME_REPAIR_TEMPERATURE,
        num_predict=INNER_RHYME_REPAIR_NUM_PREDICT,
    )
    return parse_rhyme_repair_variants_response(raw_response, num_variants)


def generate_conditioned_inner_rhyme_repair_variants_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
    enforced_candidate_final_words: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Genera variantes B condicionadas por palabras finales concretas."""
    candidate_words = [
        str(word).strip()
        for word in enforced_candidate_final_words[:INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS]
        if str(word).strip()
    ]
    conditioned_generation_reports = []
    variants = []
    seen_variants = set()

    for word in candidate_words:
        generation_report = {
            "required_final_word": word,
            "attempted": True,
            "variants": [],
            "error": None,
        }

        try:
            word_variants = generate_inner_rhyme_repair_variants_for_final_word_with_ollama(
                question=question,
                stanza=stanza,
                inner_rhyme_diagnosis=inner_rhyme_diagnosis,
                required_final_word=word,
            )
        except Exception as exc:
            generation_report["error"] = str(exc)
            conditioned_generation_reports.append(generation_report)
            continue

        generation_report["variants"] = word_variants
        conditioned_generation_reports.append(generation_report)

        for variant in word_variants:
            if variant in seen_variants:
                continue
            seen_variants.add(variant)
            variants.append(variant)

    return variants, conditioned_generation_reports


def build_inner_rhyme_repair_report(
    enabled: bool,
    stanza: list[str],
) -> dict[str, Any]:
    """Crea la estructura base del informe de reparacion de rima B."""
    return {
        "enabled": enabled,
        "attempted": False,
        "changed": False,
        "reason": None,
        "original_stanza": stanza.copy(),
        "repaired_stanza": stanza.copy(),
        "target_rhyme": None,
        "current_rhyme": None,
        "verse_2": None,
        "candidate_final_words": [],
        "enforced_candidate_final_words": [],
        "has_enforced_candidate_final_words": False,
        "candidate_warning": None,
        "conditioned_by_candidate": False,
        "conditioned_candidate_words": [],
        "conditioned_generation_reports": [],
        "post_b_meter_repair_enabled": ENABLE_POST_B_RHYME_METER_REPAIR,
        "post_b_meter_repair_attempts": 0,
        "post_b_meter_repair_successes": 0,
        "original_verse_3": None,
        "original_verse_3_syllables": None,
        "selected_variant": None,
        "selected_variant_syllables": None,
        "variants": [],
        "error": None,
    }


def repair_stanza_inner_rhyme_with_ollama(
    question: str,
    stanza: list[str],
    inner_rhyme_diagnosis: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Intenta reparar la rima interior B reescribiendo solo el verso 3."""
    report = build_inner_rhyme_repair_report(ENABLE_INNER_RHYME_REPAIR, stanza)
    repaired_stanza = stanza.copy()

    report["target_rhyme"] = inner_rhyme_diagnosis.get("target_rhyme")
    report["current_rhyme"] = inner_rhyme_diagnosis.get("current_rhyme")
    report["verse_2"] = inner_rhyme_diagnosis.get("verse_2")

    if not ENABLE_INNER_RHYME_REPAIR:
        report["reason"] = "Reparacion de rima B desactivada."
        return repaired_stanza, report

    if len(stanza) != 4:
        report["reason"] = "La estrofa no tiene 4 versos."
        return repaired_stanza, report

    original_verse_3 = stanza[2]
    original_verse_3_syllables = count_verse_syllables(original_verse_3)
    original_distance = syllable_distance_to_target(original_verse_3_syllables)
    report["original_verse_3"] = original_verse_3
    report["original_verse_3_syllables"] = original_verse_3_syllables

    if inner_rhyme_diagnosis.get("is_valid", False):
        report["reason"] = "La rima interior B ya es correcta."
        return repaired_stanza, report

    target_rhyme = inner_rhyme_diagnosis.get("target_rhyme")
    if not target_rhyme:
        report["reason"] = "No hay rima objetivo para reparar la rima B."
        return repaired_stanza, report

    verse_2 = inner_rhyme_diagnosis.get("verse_2", {})
    anchor_word = verse_2.get("last_word") if isinstance(verse_2, dict) else None
    candidate_final_words = get_candidate_final_words_for_rhyme(
        anchor_word=anchor_word,
        target_rhyme=target_rhyme,
    )
    enforced_candidate_final_words = [
        word for word in candidate_final_words if word != anchor_word
    ]
    report["candidate_final_words"] = candidate_final_words
    report["enforced_candidate_final_words"] = enforced_candidate_final_words
    report["has_enforced_candidate_final_words"] = bool(
        enforced_candidate_final_words
    )
    if anchor_word and not enforced_candidate_final_words:
        report["candidate_warning"] = (
            "No hay palabras finales candidatas distintas de la palabra ancla "
            "para la rima B."
        )
    else:
        report["candidate_warning"] = None

    report["attempted"] = True

    if (
        INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE
        and enforced_candidate_final_words
    ):
        conditioned_candidate_words = enforced_candidate_final_words[
            :INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS
        ]
        variants, conditioned_generation_reports = (
            generate_conditioned_inner_rhyme_repair_variants_with_ollama(
                question=question,
                stanza=stanza,
                inner_rhyme_diagnosis=inner_rhyme_diagnosis,
                enforced_candidate_final_words=enforced_candidate_final_words,
            )
        )
        report["conditioned_by_candidate"] = True
        report["conditioned_candidate_words"] = conditioned_candidate_words
        report["conditioned_generation_reports"] = conditioned_generation_reports

        if not variants:
            report["reason"] = (
                "No se generaron variantes condicionadas validas para la "
                "rima interior B."
            )
            report["repaired_stanza"] = repaired_stanza.copy()
            return repaired_stanza, report
    else:
        try:
            variants = generate_inner_rhyme_repair_variants_with_ollama(
                question=question,
                stanza=stanza,
                inner_rhyme_diagnosis=inner_rhyme_diagnosis,
            )
        except Exception as exc:
            report["error"] = str(exc)
            report["reason"] = "Fallo generando variantes de rima interior B."
            print(f"[WARN] Fallo reparando rima interior B: {exc}")
            return repaired_stanza, report

    if not variants:
        report["reason"] = "Fallo generando variantes de rima interior B."
        report["repaired_stanza"] = repaired_stanza.copy()
        return repaired_stanza, report

    acceptable_variants = []

    for variant_index, variant in enumerate(variants):
        candidate_stanza = stanza.copy()
        candidate_stanza[2] = variant

        variant_diagnosis = diagnose_stanza_inner_rhyme(candidate_stanza)
        variant_syllables = count_verse_syllables(variant)
        variant_distance = syllable_distance_to_target(variant_syllables)
        inner_rhyme_valid = bool(variant_diagnosis.get("is_valid", False))
        metric_not_worse = variant_distance <= original_distance
        variant_final_word = get_last_word(variant)
        final_word_matches_anchor = bool(
            anchor_word and variant_final_word == anchor_word
        )
        uses_candidate_final_word = (
            not enforced_candidate_final_words
            or variant_final_word in enforced_candidate_final_words
        )
        is_acceptable = (
            inner_rhyme_valid
            and metric_not_worse
            and uses_candidate_final_word
        )

        variant_info = {
            "verse": variant,
            "syllables": variant_syllables,
            "final_word": variant_final_word,
            "final_word_matches_anchor": final_word_matches_anchor,
            "uses_candidate_final_word": uses_candidate_final_word,
            "inner_rhyme_valid": inner_rhyme_valid,
            "inner_rhyme_summary": summarize_inner_rhyme_diagnosis(
                variant_diagnosis
            ),
            "post_b_meter_repair_attempted": False,
            "post_b_meter_repair_report": None,
            "post_b_meter_repaired_verse": None,
            "post_b_meter_repaired_syllables": None,
            "post_b_meter_repaired_distance": None,
            "post_b_meter_repair_accepted": False,
            "accepted": False,
            "rejection_reason": None,
        }

        if is_acceptable:
            acceptable_variants.append(
                {
                    "index": variant_index,
                    "verse": variant,
                    "syllables": variant_syllables,
                    "distance": variant_distance,
                    "final_word_matches_anchor": final_word_matches_anchor,
                    "variant_info": variant_info,
                }
            )
        elif (
            inner_rhyme_valid
            and uses_candidate_final_word
            and not metric_not_worse
            and ENABLE_POST_B_RHYME_METER_REPAIR
        ):
            report["post_b_meter_repair_attempts"] += 1
            variant_info["post_b_meter_repair_attempted"] = True

            candidate_stanza_for_meter = stanza.copy()
            candidate_stanza_for_meter[2] = variant
            post_meter_stanza, post_meter_report = (
                repair_verse_meter_preserving_final_word_with_ollama(
                    question=question,
                    stanza=candidate_stanza_for_meter,
                    verse_index=2,
                    required_final_word=variant_final_word,
                    enabled=ENABLE_POST_B_RHYME_METER_REPAIR,
                    num_variants=POST_B_RHYME_METER_REPAIR_VARIANTS,
                    temperature=POST_B_RHYME_METER_REPAIR_TEMPERATURE,
                    num_predict=POST_B_RHYME_METER_REPAIR_NUM_PREDICT,
                    repair_label="post_b",
                )
            )
            variant_info["post_b_meter_repair_report"] = post_meter_report

            post_meter_verse = post_meter_stanza[2]
            post_meter_syllables = count_verse_syllables(post_meter_verse)
            post_meter_distance = syllable_distance_to_target(
                post_meter_syllables
            )
            post_meter_final_word = get_last_word(post_meter_verse)
            post_meter_diagnosis = diagnose_stanza_inner_rhyme(post_meter_stanza)
            post_meter_inner_rhyme_valid = bool(
                post_meter_diagnosis.get("is_valid", False)
            )
            post_meter_preserves_final_word = (
                post_meter_final_word == variant_final_word
            )
            post_meter_metric_not_worse = post_meter_distance <= original_distance
            post_meter_is_acceptable = (
                post_meter_inner_rhyme_valid
                and post_meter_preserves_final_word
                and post_meter_metric_not_worse
                and uses_candidate_final_word
            )

            variant_info["post_b_meter_repaired_verse"] = post_meter_verse
            variant_info["post_b_meter_repaired_syllables"] = (
                post_meter_syllables
            )
            variant_info["post_b_meter_repaired_distance"] = (
                post_meter_distance
            )

            if post_meter_is_acceptable:
                variant_info["post_b_meter_repair_accepted"] = True
                report["post_b_meter_repair_successes"] += 1
                acceptable_variants.append(
                    {
                        "index": variant_index,
                        "verse": post_meter_verse,
                        "syllables": post_meter_syllables,
                        "distance": post_meter_distance,
                        "final_word_matches_anchor": final_word_matches_anchor,
                        "variant_info": variant_info,
                    }
                )
            else:
                variant_info["rejection_reason"] = (
                    "Corrige B, pero la reparacion metrica posterior no "
                    "consigue una variante aceptable."
                )
        elif not uses_candidate_final_word:
            variant_info["rejection_reason"] = (
                "No termina en una palabra final candidata distinta del verso 2."
            )
        elif not inner_rhyme_valid:
            variant_info["rejection_reason"] = "No corrige la rima interior B."
        else:
            variant_info["rejection_reason"] = (
                "Empeora la distancia metrica del verso 3."
            )

        report["variants"].append(variant_info)

    if not acceptable_variants:
        report["reason"] = "Ninguna variante corrige B sin empeorar metrica."
        report["repaired_stanza"] = repaired_stanza.copy()
        return repaired_stanza, report

    selected = min(
        acceptable_variants,
        key=lambda item: (
            item["final_word_matches_anchor"],
            item["distance"],
            item["index"],
        ),
    )
    selected["variant_info"]["accepted"] = True
    selected["variant_info"]["rejection_reason"] = None

    repaired_stanza[2] = selected["verse"]
    report["changed"] = True
    report["reason"] = "Se acepto una variante que corrige la rima interior B."
    report["selected_variant"] = selected["verse"]
    report["selected_variant_syllables"] = selected["syllables"]
    report["repaired_stanza"] = repaired_stanza.copy()
    return repaired_stanza, report


# =========================
# 10. Resumenes y trazas
# =========================

def _get_nested_metric(metrics: dict[str, Any], section: str, key: str) -> Any:
    """Obtiene una metrica anidada de forma segura."""
    section_value = metrics.get(section, {})
    if not isinstance(section_value, dict):
        return None
    return section_value.get(key)


def summarize_metrics(metrics: dict[str, Any]) -> str:
    """Resume las metricas formales de evaluate_stanza_abba en una linea."""
    if not isinstance(metrics, dict):
        metrics = {}

    actual_verses = _get_nested_metric(metrics, "verse_count", "actual_verses")
    expected_verses = _get_nested_metric(metrics, "verse_count", "expected_verses")
    correct_syllables = _get_nested_metric(metrics, "syllables", "correct_verses")
    expected_syllables = _get_nested_metric(
        metrics,
        "syllables",
        "total_expected_verses",
    )
    correct_rhymes = _get_nested_metric(metrics, "rhyme", "correct_positions")
    total_rhymes = _get_nested_metric(metrics, "rhyme", "total_positions")
    score_20 = metrics.get("score_20")

    return (
        f"versos={actual_verses if actual_verses is not None else '?'}/"
        f"{expected_verses if expected_verses is not None else '?'} | "
        f"endecasilabos={correct_syllables if correct_syllables is not None else '?'}/"
        f"{expected_syllables if expected_syllables is not None else '?'} | "
        f"rima={correct_rhymes if correct_rhymes is not None else '?'}/"
        f"{total_rhymes if total_rhymes is not None else '?'} | "
        f"score={score_20 if score_20 is not None else '?'}/20"
    )


def summarize_meter_repair_report(report: dict[str, Any]) -> str:
    """Resume el informe de reparacion metrica en una linea."""
    if not isinstance(report, dict):
        return "reparación métrica = sin informe"

    if not report.get("enabled", False):
        return "reparación métrica = desactivada"

    verse_repairs = report.get("verse_repairs", [])
    if not isinstance(verse_repairs, list):
        verse_repairs = []

    attempted = sum(1 for item in verse_repairs if item.get("attempted"))
    improved = sum(1 for item in verse_repairs if item.get("improved"))
    changed = bool(report.get("changed", False))

    return (
        f"reparación métrica = activa | intentos = {attempted} | "
        f"mejoras = {improved} | cambio = {changed}"
    )


def summarize_outer_rhyme_repair_report(report: dict[str, Any]) -> str:
    """Resume el informe de reparacion de rima exterior A en una linea."""
    if not isinstance(report, dict):
        return "reparacion rima A = sin informe"

    if not report.get("enabled", False):
        return "reparacion rima A = desactivada"

    if not report.get("attempted", False):
        return "reparacion rima A = no necesaria"

    changed = bool(report.get("changed", False))
    post_a_attempts = report.get("post_a_meter_repair_attempts", 0)
    post_a_successes = report.get("post_a_meter_repair_successes", 0)
    return (
        f"reparacion rima A = activa | intento = True | cambio = {changed} | "
        f"post-metrica A = {post_a_attempts}/{post_a_successes}"
    )


def summarize_inner_rhyme_repair_report(report: dict[str, Any]) -> str:
    """Resume el informe de reparacion de rima interior B en una linea."""
    if not isinstance(report, dict):
        return "reparacion rima B = sin informe"

    if not report.get("enabled", False):
        return "reparacion rima B = desactivada"

    if not report.get("attempted", False):
        return "reparacion rima B = no necesaria"

    changed = bool(report.get("changed", False))
    post_b_attempts = report.get("post_b_meter_repair_attempts", 0)
    post_b_successes = report.get("post_b_meter_repair_successes", 0)
    return (
        f"reparacion rima B = activa | intento = True | cambio = {changed} | "
        f"post-metrica B = {post_b_attempts}/{post_b_successes}"
    )


def summarize_outer_rhyme_diagnosis(diagnosis: dict[str, Any]) -> str:
    """Resume el diagnostico de rima exterior AXYA en una linea."""
    if not isinstance(diagnosis, dict):
        return "rima AXYA = no evaluable | sin diagnostico"

    if not diagnosis.get("has_enough_verses", False):
        return "rima AXYA = no evaluable | faltan versos"

    target_rhyme = diagnosis.get("target_rhyme")
    current_rhyme = diagnosis.get("current_rhyme")

    if diagnosis.get("is_valid", False):
        return (
            "rima AXYA = correcta | "
            f"v1 = '{target_rhyme}' | v4 = '{current_rhyme}'"
        )

    return (
        "rima AXYA = incorrecta | "
        f"objetivo = '{target_rhyme}' | v4 = '{current_rhyme}'"
    )


def summarize_inner_rhyme_diagnosis(diagnosis: dict[str, Any]) -> str:
    """Resume el diagnostico de rima interior B en una linea."""
    if not isinstance(diagnosis, dict):
        return "rima interior B = no evaluable | sin diagnostico"

    if not diagnosis.get("has_enough_verses", False):
        return "rima interior B = no evaluable | faltan versos"

    target_rhyme = diagnosis.get("target_rhyme")
    current_rhyme = diagnosis.get("current_rhyme")

    if diagnosis.get("is_valid", False):
        return (
            "rima interior B = correcta | "
            f"v2 = '{target_rhyme}' | v3 = '{current_rhyme}'"
        )

    return (
        "rima interior B = incorrecta | "
        f"objetivo = '{target_rhyme}' | v3 = '{current_rhyme}'"
    )


def build_trace_entry(item: dict[str, Any]) -> dict[str, Any]:
    """Construye una entrada compacta para trazar beams y candidatos."""
    metrics = item.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    stanza = item.get("stanza")
    if stanza is not None and not isinstance(stanza, list):
        stanza = []

    score_history = item.get("score_history", [])
    if not isinstance(score_history, list):
        score_history = []

    meter_repair_report = item.get("meter_repair_report", {})
    if not isinstance(meter_repair_report, dict):
        meter_repair_report = {}

    outer_rhyme_repair_report = item.get("outer_rhyme_repair_report", {})
    if not isinstance(outer_rhyme_repair_report, dict):
        outer_rhyme_repair_report = {}

    inner_rhyme_repair_report = item.get("inner_rhyme_repair_report", {})
    if not isinstance(inner_rhyme_repair_report, dict):
        inner_rhyme_repair_report = {}

    outer_rhyme_diagnosis = item.get("outer_rhyme_diagnosis", {})
    if not isinstance(outer_rhyme_diagnosis, dict):
        outer_rhyme_diagnosis = {}

    inner_rhyme_diagnosis = item.get("inner_rhyme_diagnosis", {})
    if not isinstance(inner_rhyme_diagnosis, dict):
        inner_rhyme_diagnosis = {}

    return {
        "stanza": stanza,
        "score": item.get("score", 0.0),
        "step_score": item.get("step_score", 0.0),
        "score_history": score_history,
        "preserved_by_elitism": bool(item.get("preserved_by_elitism", False)),
        "metrics_summary": summarize_metrics(metrics),
        "meter_repair_summary": summarize_meter_repair_report(meter_repair_report),
        "meter_repair_report": meter_repair_report,
        "outer_rhyme_repair_summary": summarize_outer_rhyme_repair_report(
            outer_rhyme_repair_report
        ),
        "outer_rhyme_repair_report": outer_rhyme_repair_report,
        "inner_rhyme_repair_summary": summarize_inner_rhyme_repair_report(
            inner_rhyme_repair_report
        ),
        "inner_rhyme_repair_report": inner_rhyme_repair_report,
        "outer_rhyme_summary": summarize_outer_rhyme_diagnosis(
            outer_rhyme_diagnosis
        ),
        "outer_rhyme_diagnosis": outer_rhyme_diagnosis,
        "inner_rhyme_summary": summarize_inner_rhyme_diagnosis(
            inner_rhyme_diagnosis
        ),
        "inner_rhyme_diagnosis": inner_rhyme_diagnosis,
        "feedback": item.get("feedback", ""),
        "generation_reasoning": item.get("generation_reasoning", ""),
    }


def get_main_feedback_line(feedback: str | None) -> str:
    """Devuelve la primera linea no vacia del feedback."""
    for line in str(feedback or "").splitlines():
        stripped_line = line.strip()
        if stripped_line:
            return stripped_line
    return "Sin feedback formal todavia."


def is_preservable_beam(beam: dict[str, Any]) -> bool:
    """Indica si un beam previo puede conservarse por elitismo."""
    if not isinstance(beam, dict):
        return False

    stanza = beam.get("stanza")
    if not isinstance(stanza, list) or not stanza:
        return False

    score_history = beam.get("score_history")
    if not isinstance(score_history, list) or not score_history:
        return False

    return True


def mark_beam_as_elite_candidate(beam: dict[str, Any]) -> dict[str, Any]:
    """Marca una copia superficial del beam como conservada por elitismo."""
    elite_beam = beam.copy()

    stanza = elite_beam.get("stanza")
    if isinstance(stanza, list):
        elite_beam["stanza"] = stanza.copy()

    score_history = elite_beam.get("score_history")
    if isinstance(score_history, list):
        elite_beam["score_history"] = score_history.copy()

    elite_beam["preserved_by_elitism"] = True
    return elite_beam


def deduplicate_beams_by_stanza(
    beams: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Elimina beams duplicados conservando el primero de cada estrofa."""
    deduplicated_beams = []
    seen_stanzas = set()

    for index, beam in enumerate(beams):
        stanza = beam.get("stanza") if isinstance(beam, dict) else None
        if isinstance(stanza, list):
            key = tuple(str(verse).strip() for verse in stanza)
        else:
            key = ("__invalid_stanza__", index)

        if key in seen_stanzas:
            continue

        seen_stanzas.add(key)
        deduplicated_beams.append(beam)

    return deduplicated_beams


# =========================
# 11. Nodos de LangGraph
# =========================

def expand_node(state: BeamSearchState) -> dict:
    """Expande cada beam generando nuevas estrofas candidatas."""
    all_candidates: List[Dict[str, Any]] = []

    print(f"\n=== EXPAND STEP {state['step']} ===")

    for beam_index, beam in enumerate(state["beams"]):
        previous_stanza = beam.get("stanza")
        if not isinstance(previous_stanza, list) or not previous_stanza:
            previous_stanza = None

        current_feedback = beam.get(
            "feedback",
            "Todavia no se ha generado ninguna estrofa.",
        )

        print(f"\nBeam origen {beam_index}:")
        print(
            "  Modo: "
            + (
                "correccion de estrofa previa"
                if previous_stanza
                else "generacion inicial desde cero"
            )
        )
        print(f"  Score actual: {float(beam.get('score', 0.0)):.3f}")

        generated_stanzas = generate_stanza_candidates_with_ollama(
            question=state["question"],
            previous_stanza=previous_stanza,
            feedback=current_feedback,
            num_candidates=state["k"],
            previous_beam=beam,
        )

        for generated in generated_stanzas:
            candidate = {
                "stanza": generated["stanza"],
                "score": beam.get("score", 0.0),
                "step_score": beam.get("step_score", 0.0),
                "score_history": beam.get("score_history", []).copy(),
                "feedback": current_feedback,
                "metrics": beam.get("metrics", {}).copy(),
                "meter_repair_report": {},
                "outer_rhyme_repair_report": {},
                "inner_rhyme_repair_report": {},
                "outer_rhyme_diagnosis": {},
                "inner_rhyme_diagnosis": {},
                "preserved_by_elitism": False,
                "generation_reasoning": generated["generation_reasoning"],
            }
            all_candidates.append(candidate)

    print() # Dejo este espacio en blanco para ver mejor la salida
    print(f"Candidatos generados: {len(all_candidates)}")
    for index, candidate in enumerate(all_candidates):
        stanza = candidate.get("stanza", [])
        stanza_length = len(stanza) if isinstance(stanza, list) else 0
        print(f"  Candidato {index}: {stanza_length} versos generados")

    trace = state.get("trace", []).copy()
    trace.append(
        {
            "step": state["step"],
            "phase": "expand",
            "generated_candidates": [
                build_trace_entry(candidate) for candidate in all_candidates
            ],
        }
    )

    return {
        "candidates": all_candidates,
        "trace": trace,
    }


def score_node(state: BeamSearchState) -> dict:
    """Evalua formalmente cada estrofa candidata."""
    scored_candidates: List[Dict[str, Any]] = []

    print(f"=== SCORE STEP {state['step']} ===")

    for index, candidate in enumerate(state["candidates"]):
        stanza = candidate.get("stanza", [])
        if not isinstance(stanza, list):
            stanza = []

        stanza, meter_repair_report = repair_stanza_meter_with_ollama(
            question=state["question"],
            stanza=stanza,
        )
        outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
        stanza, outer_rhyme_repair_report = repair_stanza_outer_rhyme_with_ollama(
            question=state["question"],
            stanza=stanza,
            outer_rhyme_diagnosis=outer_rhyme_diagnosis,
        )
        inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
        stanza, inner_rhyme_repair_report = repair_stanza_inner_rhyme_with_ollama(
            question=state["question"],
            stanza=stanza,
            inner_rhyme_diagnosis=inner_rhyme_diagnosis,
        )
        evaluation = evaluate_stanza_abba(stanza)
        outer_rhyme_diagnosis = diagnose_stanza_outer_rhyme(stanza)
        inner_rhyme_diagnosis = diagnose_stanza_inner_rhyme(stanza)
        step_score = float(evaluation.get("score", 0.0))
        updated_history = candidate.get("score_history", []) + [step_score]
        total_score = aggregate_scores(updated_history, alpha=ALPHA)

        score_20 = evaluation.get("score_20", round(step_score * 20, 2))
        errors = evaluation.get("errors", [])
        error_count = len(errors) if isinstance(errors, list) else 0

        print(f"\nCandidato {index}:")
        print(f"Numero de versos: {len(stanza)}")
        print(f"Score formal: {step_score:.3f} ({score_20}/20)")
        print(f"Errores detectados: {error_count}")
        print(f"Resumen métricas: {summarize_metrics(evaluation)}")
        print(
            "Reparación métrica: "
            f"{summarize_meter_repair_report(meter_repair_report)}"
        )
        print(
            "Reparacion rima A: "
            f"{summarize_outer_rhyme_repair_report(outer_rhyme_repair_report)}"
        )
        print(
            "Reparacion rima B: "
            f"{summarize_inner_rhyme_repair_report(inner_rhyme_repair_report)}"
        )
        print(
            "Rima exterior AXYA: "
            f"{summarize_outer_rhyme_diagnosis(outer_rhyme_diagnosis)}"
        )
        print(
            "Rima interior B: "
            f"{summarize_inner_rhyme_diagnosis(inner_rhyme_diagnosis)}"
        )

        scored_candidate = {
            "stanza": stanza.copy(),
            "score": total_score,
            "step_score": step_score,
            "score_history": updated_history,
            "feedback": str(evaluation.get("feedback", "")),
            "metrics": evaluation,
            "meter_repair_report": meter_repair_report,
            "outer_rhyme_repair_report": outer_rhyme_repair_report,
            "inner_rhyme_repair_report": inner_rhyme_repair_report,
            "outer_rhyme_diagnosis": outer_rhyme_diagnosis,
            "inner_rhyme_diagnosis": inner_rhyme_diagnosis,
            "preserved_by_elitism": bool(
                candidate.get("preserved_by_elitism", False)
            ),
            "generation_reasoning": candidate.get("generation_reasoning", ""),
        }
        scored_candidates.append(scored_candidate)

    trace = state.get("trace", []).copy()
    trace.append(
        {
            "step": state["step"],
            "phase": "score",
            "candidates": [
                build_trace_entry(candidate) for candidate in scored_candidates
            ],
        }
    )

    return {
        "candidates": scored_candidates,
        "trace": trace,
    }


def prune_node(state: BeamSearchState) -> dict:
    """Selecciona los k mejores candidatos segun el score agregado."""
    print(f"=== PRUNE STEP {state['step']} ===")

    selection_pool = list(state["candidates"])

    if ENABLE_BEAM_ELITISM:
        previous_beams = state.get("beams", [])
        preservable_beams = [
            mark_beam_as_elite_candidate(beam)
            for beam in previous_beams
            if is_preservable_beam(beam)
        ]
        elite_beams = sorted(
            preservable_beams,
            key=lambda beam: beam.get("score", 0.0),
            reverse=True,
        )[:ELITE_BEAMS_TO_KEEP]
        selection_pool.extend(elite_beams)

    selection_pool = deduplicate_beams_by_stanza(selection_pool)
    sorted_candidates = sorted(
        selection_pool,
        key=lambda beam: beam["score"],
        reverse=True,
    )
    top_k = sorted_candidates[: state["k"]]

    for index, beam in enumerate(top_k):
        print(f"\nBeam {index}:")
        print(f"Score global: {beam['score']:.3f}")
        print(f"Score ultimo paso: {beam.get('step_score', 0.0):.3f}")
        print(f"Historial scores: {[round(s, 3) for s in beam.get('score_history', [])]}")
        print(
            "Conservado por elitismo: "
            f"{bool(beam.get('preserved_by_elitism', False))}"
        )
        print(f"Resumen métricas: {summarize_metrics(beam.get('metrics', {}))}")
        print(
            "Reparación métrica: "
            f"{summarize_meter_repair_report(beam.get('meter_repair_report', {}))}"
        )
        print(
            "Reparacion rima A: "
            f"{summarize_outer_rhyme_repair_report(beam.get('outer_rhyme_repair_report', {}))}"
        )
        print(
            "Reparacion rima B: "
            f"{summarize_inner_rhyme_repair_report(beam.get('inner_rhyme_repair_report', {}))}"
        )
        print(
            "Rima exterior AXYA: "
            f"{summarize_outer_rhyme_diagnosis(beam.get('outer_rhyme_diagnosis', {}))}"
        )
        print(
            "Rima interior B: "
            f"{summarize_inner_rhyme_diagnosis(beam.get('inner_rhyme_diagnosis', {}))}"
        )

        print("Estrofa completa:")
        stanza = beam.get("stanza")
        if isinstance(stanza, list) and stanza:
            for verse_number, verse in enumerate(stanza, start=1):
                print(f"  {verse_number}. {verse}")
        else:
            print("  [Sin estrofa generada todavia]")

        print("Feedback principal:")
        print("  ", get_main_feedback_line(beam.get("feedback", "")))

    trace = state.get("trace", []).copy()
    trace.append(
        {
            "step": state["step"],
            "phase": "prune",
            "selected_beams": [build_trace_entry(beam) for beam in top_k],
        }
    )

    return {
        "beams": top_k,
        "step": state["step"] + 1,
        "trace": trace,
    }


# =========================
# 12. Agregacion de puntuaciones
# =========================

def aggregate_scores(score_history: list[float], alpha: float = ALPHA) -> float:
    """Agrega scores con media geometrica normalizada."""
    if not score_history:
        return 0.0

    product = 1.0
    for score in score_history:
        safe_score = max(score, EPSILON)
        product *= safe_score

    T = len(score_history)
    return product ** (1.0 / (T ** alpha))


# =========================
# 13. Decision de continuacion
# =========================

def should_continue(state: BeamSearchState) -> str:
    """Para el grafo al alcanzar max_steps."""
    if state["step"] >= state["max_steps"]:
        trace = state.get("trace", [])
        best_score = 0.0
        beams = state.get("beams", [])
        if beams:
            best_beam = max(beams, key=lambda beam: beam.get("score", 0.0))
            best_score = float(best_beam.get("score", 0.0))

        trace.append(
            {
                "step": state["step"],
                "phase": "stop",
                "reason": "max_steps",
                "best_score": best_score,
            }
        )
        state["trace"] = trace
        print("Fin: alcanzado max_steps.")
        return "end"

    return "continue"


# =========================
# 14. Guardado de resultados
# =========================

def format_execution_time(seconds: float | None) -> str:
    """Formatea una duracion en segundos para mostrarla en el TXT final."""
    if seconds is None or isinstance(seconds, bool):
        return "no disponible"

    try:
        seconds_value = float(seconds)
    except (TypeError, ValueError):
        return "no disponible"

    if seconds_value < 0 or seconds_value != seconds_value:
        return "no disponible"

    minutes_value = seconds_value / 60.0
    return f"{seconds_value:.2f} segundos ({minutes_value:.2f} minutos)"


def save_final_result(
    best_beam: dict[str, Any],
    trace: list[dict[str, Any]],
    run_parameters: dict[str, Any],
    execution_time_seconds: float | None = None,
    output_dir: str = "outputs",
) -> None:
    """Guarda la estrofa final, sus metricas y la traza completa."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stanza_path = output_path / f"final_stanza_{timestamp}.txt"
    metrics_path = output_path / f"final_stanza_metrics_{timestamp}.json"
    trace_path = output_path / f"final_stanza_trace_{timestamp}.json"

    stanza = best_beam.get("stanza", [])
    if not isinstance(stanza, list):
        stanza = []

    metrics = best_beam.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    score_history = best_beam.get("score_history", [])
    if not isinstance(score_history, list):
        score_history = []

    meter_repair_report = best_beam.get("meter_repair_report", {})
    if not isinstance(meter_repair_report, dict):
        meter_repair_report = {}

    outer_rhyme_repair_report = best_beam.get("outer_rhyme_repair_report", {})
    if not isinstance(outer_rhyme_repair_report, dict):
        outer_rhyme_repair_report = {}

    inner_rhyme_repair_report = best_beam.get("inner_rhyme_repair_report", {})
    if not isinstance(inner_rhyme_repair_report, dict):
        inner_rhyme_repair_report = {}

    outer_rhyme_diagnosis = best_beam.get("outer_rhyme_diagnosis", {})
    if not isinstance(outer_rhyme_diagnosis, dict):
        outer_rhyme_diagnosis = {}

    inner_rhyme_diagnosis = best_beam.get("inner_rhyme_diagnosis", {})
    if not isinstance(inner_rhyme_diagnosis, dict):
        inner_rhyme_diagnosis = {}

    execution_time_summary = format_execution_time(execution_time_seconds)

    text_lines = [
        "ESTROFA FINAL",
        "",
        f"Modelo: {run_parameters.get('model')}",
        f"Temperatura: {run_parameters.get('temperature')}",
        f"Num predict: {run_parameters.get('num_predict')}",
        f"k: {run_parameters.get('k')}",
        f"max_steps: {run_parameters.get('max_steps')}",
        f"alpha: {run_parameters.get('alpha')}",
        f"Tiempo total de ejecucion: {execution_time_summary}",
        (
            "Diccionario de palabras candidatas de rima activo: "
            f"{run_parameters.get('enable_rhyme_hint_examples')}"
        ),
        f"Reparacion metrica activa: {run_parameters.get('enable_local_meter_repair')}",
        (
            "Variantes por verso para reparacion metrica: "
            f"{run_parameters.get('meter_repair_variants_per_verse')}"
        ),
        (
            "Temperatura reparacion metrica: "
            f"{run_parameters.get('meter_repair_temperature')}"
        ),
        (
            "Num predict reparacion metrica: "
            f"{run_parameters.get('meter_repair_num_predict')}"
        ),
        f"Reparacion rima A activa: {run_parameters.get('enable_outer_rhyme_repair')}",
        (
            "Variantes para reparacion rima A: "
            f"{run_parameters.get('outer_rhyme_repair_variants')}"
        ),
        (
            "Temperatura reparacion rima A: "
            f"{run_parameters.get('outer_rhyme_repair_temperature')}"
        ),
        (
            "Num predict reparacion rima A: "
            f"{run_parameters.get('outer_rhyme_repair_num_predict')}"
        ),
        (
            "Reparacion metrica posterior a rima A activa: "
            f"{run_parameters.get('enable_post_a_rhyme_meter_repair')}"
        ),
        (
            "Variantes para reparacion metrica posterior a rima A: "
            f"{run_parameters.get('post_a_rhyme_meter_repair_variants')}"
        ),
        (
            "Temperatura reparacion metrica posterior a rima A: "
            f"{run_parameters.get('post_a_rhyme_meter_repair_temperature')}"
        ),
        (
            "Num predict reparacion metrica posterior a rima A: "
            f"{run_parameters.get('post_a_rhyme_meter_repair_num_predict')}"
        ),
        f"Reparacion rima B activa: {run_parameters.get('enable_inner_rhyme_repair')}",
        (
            "Variantes para reparacion rima B: "
            f"{run_parameters.get('inner_rhyme_repair_variants')}"
        ),
        (
            "Temperatura reparacion rima B: "
            f"{run_parameters.get('inner_rhyme_repair_temperature')}"
        ),
        (
            "Num predict reparacion rima B: "
            f"{run_parameters.get('inner_rhyme_repair_num_predict')}"
        ),
        (
            "Reparacion rima B condicionada por candidata: "
            f"{run_parameters.get('inner_rhyme_repair_conditioned_by_candidate')}"
        ),
        (
            "Variantes por candidata para reparacion rima B: "
            f"{run_parameters.get('inner_rhyme_repair_variants_per_candidate')}"
        ),
        (
            "Max candidatas usadas para reparacion rima B: "
            f"{run_parameters.get('inner_rhyme_repair_max_candidate_words')}"
        ),
        (
            "Reparacion metrica posterior a rima B activa: "
            f"{run_parameters.get('enable_post_b_rhyme_meter_repair')}"
        ),
        (
            "Variantes para reparacion metrica posterior a rima B: "
            f"{run_parameters.get('post_b_rhyme_meter_repair_variants')}"
        ),
        (
            "Temperatura reparacion metrica posterior a rima B: "
            f"{run_parameters.get('post_b_rhyme_meter_repair_temperature')}"
        ),
        (
            "Num predict reparacion metrica posterior a rima B: "
            f"{run_parameters.get('post_b_rhyme_meter_repair_num_predict')}"
        ),
        "",
        f"Score global: {float(best_beam.get('score', 0.0)):.3f}",
        f"Historial de scores: {[round(float(score), 3) for score in score_history]}",
        f"Resumen de metricas: {summarize_metrics(metrics)}",
        f"Resumen de reparacion metrica: {summarize_meter_repair_report(meter_repair_report)}",
        (
            "Resumen reparacion rima A: "
            f"{summarize_outer_rhyme_repair_report(outer_rhyme_repair_report)}"
        ),
        (
            "Resumen reparacion rima B: "
            f"{summarize_inner_rhyme_repair_report(inner_rhyme_repair_report)}"
        ),
        (
            "Resumen rima exterior AXYA: "
            f"{summarize_outer_rhyme_diagnosis(outer_rhyme_diagnosis)}"
        ),
        (
            "Resumen rima interior B: "
            f"{summarize_inner_rhyme_diagnosis(inner_rhyme_diagnosis)}"
        ),
        "",
        "Feedback final:",
        str(best_beam.get("feedback", "Sin feedback formal todavia.")),
        "",
        "Estrofa completa:",
    ]

    if stanza:
        text_lines.extend(
            f"{verse_number}. {verse}"
            for verse_number, verse in enumerate(stanza, start=1)
        )
    else:
        text_lines.append("[Sin estrofa generada todavia]")

    stanza_path.write_text("\n".join(text_lines), encoding="utf-8")

    json_payload = {
        "run_parameters": run_parameters,
        "stanza": stanza,
        "score": best_beam.get("score", 0.0),
        "step_score": best_beam.get("step_score", 0.0),
        "score_history": score_history,
        "feedback": best_beam.get("feedback", ""),
        "metrics": metrics,
        "meter_repair_report": meter_repair_report,
        "outer_rhyme_repair_report": outer_rhyme_repair_report,
        "inner_rhyme_repair_report": inner_rhyme_repair_report,
        "outer_rhyme_diagnosis": outer_rhyme_diagnosis,
        "inner_rhyme_diagnosis": inner_rhyme_diagnosis,
        "generation_reasoning": best_beam.get("generation_reasoning", ""),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, ensure_ascii=False, indent=2)

    trace_payload = {
        "run_parameters": run_parameters,
        "trace": trace,
    }
    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace_payload, file, ensure_ascii=False, indent=2)

    print("\nArchivos generados:")
    print(f"  Estrofa final: {stanza_path}")
    print(f"  Metricas: {metrics_path}")
    print(f"  Traza: {trace_path}")


# =========================
# 15. Main
# =========================

def main() -> None:
    """Construye y ejecuta el grafo de Beam Search para estrofas ABBA."""
    builder = StateGraph(BeamSearchState)

    builder.add_node("expand", expand_node)
    builder.add_node("score", score_node)
    builder.add_node("prune", prune_node)

    builder.set_entry_point("expand")

    builder.add_edge("expand", "score")
    builder.add_edge("score", "prune")
    builder.add_conditional_edges(
        "prune",
        should_continue,
        {
            "continue": "expand",
            "end": END,
        },
    )

    graph = builder.compile()

    initial_state: BeamSearchState = {
        "question": (
            "Genera una estrofa de 4 versos endecasílabos con rima consonante ABBA sobre el paso del tiempo y la memoria."
        ),
        "beams": [
            {
                "stanza": None,
                "score": 0.0,
                "step_score": 0.0,
                "score_history": [],
                "feedback": "Todavia no se ha generado ninguna estrofa.",
                "metrics": {},
                "meter_repair_report": {},
                "outer_rhyme_repair_report": {},
                "inner_rhyme_repair_report": {},
                "outer_rhyme_diagnosis": {},
                "inner_rhyme_diagnosis": {},
                "generation_reasoning": "Estado inicial sin estrofa.",
            }
        ],
        "candidates": [],
        "trace": [],
        "step": 0,
        "max_steps": 4,
        "k": 2,
    }

    # Estos parámetros simplemente los guardo para tener un registro claro de con qué configuración se ejecutó el grafo.
    run_parameters = {
        "model": GENERATION_MODEL,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "k": initial_state["k"],
        "max_steps": initial_state["max_steps"],
        "alpha": ALPHA,
        "enable_rhyme_hint_examples": ENABLE_RHYME_HINT_EXAMPLES,
        "enable_local_meter_repair": ENABLE_LOCAL_METER_REPAIR,
        "meter_repair_variants_per_verse": METER_REPAIR_VARIANTS_PER_VERSE,
        "meter_repair_temperature": METER_REPAIR_TEMPERATURE,
        "meter_repair_num_predict": METER_REPAIR_NUM_PREDICT,
        "enable_outer_rhyme_repair": ENABLE_OUTER_RHYME_REPAIR,
        "outer_rhyme_repair_variants": OUTER_RHYME_REPAIR_VARIANTS,
        "outer_rhyme_repair_temperature": OUTER_RHYME_REPAIR_TEMPERATURE,
        "outer_rhyme_repair_num_predict": OUTER_RHYME_REPAIR_NUM_PREDICT,
        "enable_post_a_rhyme_meter_repair": ENABLE_POST_A_RHYME_METER_REPAIR,
        "post_a_rhyme_meter_repair_variants": (
            POST_A_RHYME_METER_REPAIR_VARIANTS
        ),
        "post_a_rhyme_meter_repair_temperature": (
            POST_A_RHYME_METER_REPAIR_TEMPERATURE
        ),
        "post_a_rhyme_meter_repair_num_predict": (
            POST_A_RHYME_METER_REPAIR_NUM_PREDICT
        ),
        "enable_inner_rhyme_repair": ENABLE_INNER_RHYME_REPAIR,
        "inner_rhyme_repair_variants": INNER_RHYME_REPAIR_VARIANTS,
        "inner_rhyme_repair_temperature": INNER_RHYME_REPAIR_TEMPERATURE,
        "inner_rhyme_repair_num_predict": INNER_RHYME_REPAIR_NUM_PREDICT,
        "inner_rhyme_repair_conditioned_by_candidate": (
            INNER_RHYME_REPAIR_CONDITIONED_BY_CANDIDATE
        ),
        "inner_rhyme_repair_variants_per_candidate": (
            INNER_RHYME_REPAIR_VARIANTS_PER_CANDIDATE
        ),
        "inner_rhyme_repair_max_candidate_words": (
            INNER_RHYME_REPAIR_MAX_CANDIDATE_WORDS
        ),
        "enable_post_b_rhyme_meter_repair": ENABLE_POST_B_RHYME_METER_REPAIR,
        "post_b_rhyme_meter_repair_variants": (
            POST_B_RHYME_METER_REPAIR_VARIANTS
        ),
        "post_b_rhyme_meter_repair_temperature": (
            POST_B_RHYME_METER_REPAIR_TEMPERATURE
        ),
        "post_b_rhyme_meter_repair_num_predict": (
            POST_B_RHYME_METER_REPAIR_NUM_PREDICT
        ),
    }

    execution_start = perf_counter()
    result = graph.invoke(initial_state)
    execution_time_seconds = perf_counter() - execution_start

    print("\n=== RESULTADO FINAL ===")
    for index, beam in enumerate(result["beams"]):
        print(f"\nBeam final {index}:")
        print(f"Score global final: {beam['score']:.3f}")
        print(f"Historial scores: {[round(s, 3) for s in beam['score_history']]}")
        print(f"Resumen metricas: {summarize_metrics(beam.get('metrics', {}))}")
        print(
            "Reparacion metrica: "
            f"{summarize_meter_repair_report(beam.get('meter_repair_report', {}))}"
        )
        print(
            "Reparacion rima A: "
            f"{summarize_outer_rhyme_repair_report(beam.get('outer_rhyme_repair_report', {}))}"
        )
        print(
            "Reparacion rima B: "
            f"{summarize_inner_rhyme_repair_report(beam.get('inner_rhyme_repair_report', {}))}"
        )
        print(
            "Rima exterior AXYA: "
            f"{summarize_outer_rhyme_diagnosis(beam.get('outer_rhyme_diagnosis', {}))}"
        )
        print(
            "Rima interior B: "
            f"{summarize_inner_rhyme_diagnosis(beam.get('inner_rhyme_diagnosis', {}))}"
        )

        print("Feedback final:")
        print("  ", get_main_feedback_line(beam.get("feedback", "")))

        print("Estrofa final:\n")
        stanza = beam.get("stanza")
        if isinstance(stanza, list) and stanza:
            for verse_number, verse in enumerate(stanza, start=1):
                print(f"  {verse_number}. {verse}")
        else:
            print("  [Sin estrofa generada todavia]")

        print("Ultima razon de generacion:")
        print("  ", beam.get("generation_reasoning", "Sin razon de generacion."))

    if result["beams"]:
        save_final_result(
            best_beam=result["beams"][0],
            trace=result.get("trace", []),
            run_parameters=run_parameters,
            execution_time_seconds=execution_time_seconds,
        )


if __name__ == "__main__":
    main()
