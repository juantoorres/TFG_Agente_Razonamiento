from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, TypedDict

import requests
from langgraph.graph import END, StateGraph

from sonnet_metrics import (
    TARGET_SYLLABLES_PER_VERSE,
    count_verse_syllables,
    evaluate_stanza_abba,
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
RETRY_TEMPERATURE = 0.1
NUM_PREDICT = 600
ALPHA = 1.0
EPSILON = 1e-6

# Fase experimental opcional: intenta reparar solo la metrica de versos que no
# tienen 11 silabas. No toca la rima ni bloquea versos correctos.
ENABLE_LOCAL_METER_REPAIR = True
METER_REPAIR_VARIANTS_PER_VERSE = 5
METER_REPAIR_TEMPERATURE = 0.4
METER_REPAIR_NUM_PREDICT = 200


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


def _build_stanza_generation_messages(
    question: str,
    previous_stanza: list[str] | None,
    feedback: str | None,
    num_candidates: int,
    strict: bool = False,
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
        user_prompt = (
            f"Tarea original:\n{question}\n\n"
            "Estrofa candidata anterior, verso a verso:\n"
            f"{format_numbered_verses(previous_stanza)}\n\n"
            "Feedback formal recibido:\n"
            f"{summarized_feedback}\n\n"
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


def build_meter_repair_report(enabled: bool, stanza: list[str]) -> dict[str, Any]:
    """Crea la estructura base del informe de reparacion metrica."""
    return {
        "enabled": enabled,
        "changed": False,
        "original_stanza": stanza.copy(),
        "repaired_stanza": stanza.copy(),
        "verse_repairs": [],
    }


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
# 8. Resumenes y trazas
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

    return {
        "stanza": stanza,
        "score": item.get("score", 0.0),
        "step_score": item.get("step_score", 0.0),
        "score_history": score_history,
        "metrics_summary": summarize_metrics(metrics),
        "meter_repair_summary": summarize_meter_repair_report(meter_repair_report),
        "meter_repair_report": meter_repair_report,
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


# =========================
# 8. Nodos de LangGraph
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
        evaluation = evaluate_stanza_abba(stanza)
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

        scored_candidate = {
            "stanza": stanza.copy(),
            "score": total_score,
            "step_score": step_score,
            "score_history": updated_history,
            "feedback": str(evaluation.get("feedback", "")),
            "metrics": evaluation,
            "meter_repair_report": meter_repair_report,
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

    sorted_candidates = sorted(
        state["candidates"],
        key=lambda beam: beam["score"],
        reverse=True,
    )
    top_k = sorted_candidates[: state["k"]]

    for index, beam in enumerate(top_k):
        print(f"\nBeam {index}:")
        print(f"Score global: {beam['score']:.3f}")
        print(f"Score ultimo paso: {beam.get('step_score', 0.0):.3f}")
        print(f"Historial scores: {[round(s, 3) for s in beam.get('score_history', [])]}")
        print(f"Resumen métricas: {summarize_metrics(beam.get('metrics', {}))}")
        print(
            "Reparación métrica: "
            f"{summarize_meter_repair_report(beam.get('meter_repair_report', {}))}"
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
# 9. Agregacion de puntuaciones
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
# 10. Decision de continuacion
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
# 11. Guardado de resultados
# =========================

def save_final_result(
    best_beam: dict[str, Any],
    trace: list[dict[str, Any]],
    run_parameters: dict[str, Any],
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

    text_lines = [
        "ESTROFA FINAL",
        "",
        f"Modelo: {run_parameters.get('model')}",
        f"Temperatura: {run_parameters.get('temperature')}",
        f"Num predict: {run_parameters.get('num_predict')}",
        f"k: {run_parameters.get('k')}",
        f"max_steps: {run_parameters.get('max_steps')}",
        f"alpha: {run_parameters.get('alpha')}",
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
        "",
        f"Score global: {float(best_beam.get('score', 0.0)):.3f}",
        f"Historial de scores: {[round(float(score), 3) for score in score_history]}",
        f"Resumen de metricas: {summarize_metrics(metrics)}",
        f"Resumen de reparacion metrica: {summarize_meter_repair_report(meter_repair_report)}",
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
# 12. Main
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
            "Genera una estrofa clásica en español de cuatro versos "
            "endecasílabos con rima consonante ABBA. El tema debe ser la "
            "nostalgia del paso del tiempo."
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
                "generation_reasoning": "Estado inicial sin estrofa.",
            }
        ],
        "candidates": [],
        "trace": [],
        "step": 0,
        "max_steps": 3,
        "k": 2,
    }

    run_parameters = {
        "model": GENERATION_MODEL,
        "temperature": TEMPERATURE,
        "num_predict": NUM_PREDICT,
        "k": initial_state["k"],
        "max_steps": initial_state["max_steps"],
        "alpha": ALPHA,
        "enable_local_meter_repair": ENABLE_LOCAL_METER_REPAIR,
        "meter_repair_variants_per_verse": METER_REPAIR_VARIANTS_PER_VERSE,
        "meter_repair_temperature": METER_REPAIR_TEMPERATURE,
        "meter_repair_num_predict": METER_REPAIR_NUM_PREDICT,
    }

    result = graph.invoke(initial_state)

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
        )


if __name__ == "__main__":
    main()
