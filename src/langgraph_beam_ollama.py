from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import requests
import sys

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from sonnet_metrics import evaluate_sonnet


BASE_URL = "http://localhost:11434"

# Antes teníamos llama3 
GENERATION_MODEL = "mistral:7b-instruct"
# Se mantiene para una futura fase de LLM-as-a-judge.
SCORING_MODEL = "llama3"

# Añadimos hiperparámetros global
ALPHA = 1.0
EPSILON = 1e-6
TARGET_SCORE = 0.95


# =========================
# 1. Estado del grafo
# =========================
class BeamSearchState(TypedDict):
    question: str
    beams: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    step: int
    max_steps: int
    k: int


# Cada beam/candidato representa una version candidata de soneto.
# Campos esperados en cada beam/candidato:
# - sonnet: list[str]
#   Lista de versos del soneto candidato.
# - score: float
#   Puntuacion global formal del soneto.
# - score_history: list[float]
#   Historial de puntuaciones por iteracion.
# - feedback: str
#   Feedback textual de evaluacion formal.
# - metrics: dict[str, Any]
#   Metricas detalladas de evaluacion.
# - generation_reasoning: str
#   Breve explicacion de como se genero esa version.


# =========================
# 2. Utilidad general Ollama
# =========================
def chat_ollama(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, num_predict: int = 300) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json", # CAMBIO para forzar al modelo que el formato de salida sea JSON
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
# 3. Generación con Ollama
# =========================
def clean_generated_verse(verse: str) -> str:
    """
    Limpia problemas superficiales de formato en un verso generado.

    Elimina espacios exteriores, numeracion inicial común, comillas exteriores y
    espacios duplicados. No elimina tildes ni modifica el contenido poético.
    """
    cleaned_verse = str(verse).strip()
    cleaned_verse = re.sub(r"^\s*\d+\s*(?:[.)]|-)\s*", "", cleaned_verse)

    quote_pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
        "«": "»",
    }
    if len(cleaned_verse) >= 2:
        first_character = cleaned_verse[0]
        last_character = cleaned_verse[-1]
        if quote_pairs.get(first_character) == last_character:
            cleaned_verse = cleaned_verse[1:-1].strip()

    return " ".join(cleaned_verse.split())


def clean_generated_sonnet(verses: list[str]) -> list[str]:
    """
    Limpia una lista de versos generados y elimina versos vacios.
    """
    cleaned_verses = []
    for verse in verses:
        cleaned_verse = clean_generated_verse(verse)
        if cleaned_verse:
            cleaned_verses.append(cleaned_verse)
    return cleaned_verses


def _parse_sonnet_payload(payload: Dict[str, Any]) -> list[str]:
    """
    Extrae y limpia la lista de versos de un objeto JSON ya parseado.
    """
    verses = payload.get("verses")
    if not isinstance(verses, list):
        raise ValueError("El campo 'verses' no existe o no es una lista.")

    return clean_generated_sonnet([str(verse) for verse in verses])


def parse_sonnet_response(raw_response: str) -> list[str]:
    """
    Parsea una respuesta JSON con la forma {"verses": [...]}.

    No exige que haya exactamente 14 versos; esa validación se hará después con
    las métricas formales del soneto.
    """
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("La respuesta del modelo no es un objeto JSON.")
    return _parse_sonnet_payload(parsed)


def parse_sonnet_candidates_response(
    raw_response: str,
    num_candidates: int,
) -> list[dict[str, Any]]:
    """
    Parsea una respuesta JSON global con candidatos de soneto.

    Espera un objeto con la clave "candidates". Cada candidato debe incluir
    "verses" y puede incluir "generation_reasoning".
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

        sonnet = _parse_sonnet_payload(item)
        if not sonnet:
            continue

        generation_reasoning = str(
            item.get("generation_reasoning", "Sin explicación.")
        ).strip()

        parsed_candidates.append(
            {
                "sonnet": sonnet,
                "generation_reasoning": (
                    generation_reasoning if generation_reasoning else "Sin explicación."
                ),
            }
        )

    if not parsed_candidates:
        raise ValueError("No se ha podido extraer ningún candidato válido.")

    return parsed_candidates


def format_numbered_verses(verses: list[str]) -> str:
    """
    Devuelve una lista de versos numerados para incluirla en prompts.
    """
    if not verses:
        return "[sin versos]"
    return "\n".join(
        f"{verse_number}. {verse}"
        for verse_number, verse in enumerate(verses, start=1)
    )


def summarize_feedback_for_prompt(feedback: str, max_chars: int = 2500) -> str:
    """
    Recorta el feedback formal para evitar prompts demasiado largos.
    """
    clean_feedback = str(feedback).strip()
    if len(clean_feedback) <= max_chars:
        return clean_feedback
    return f"{clean_feedback[:max_chars].rstrip()}\n[feedback recortado]"


def _build_sonnet_generation_messages(
    question: str,
    previous_sonnet: list[str],
    feedback: str,
    num_candidates: int,
    strict: bool = False,
) -> List[Dict[str, str]]:
    """
    Construye los mensajes para pedir candidatos de soneto a Ollama.
    """
    strict_rules = ""
    if strict:
        strict_rules = (
            "\nModo estricto de reintento:\n"
            "- Devuelve sólamente JSON valido.\n"
            "- No cortes el JSON. Devuelve todos los corchetes y llaves de cierre.\n"
            "- No incluyas markdown, comentarios, titulo ni texto fuera del JSON.\n"
            "- Cada candidato debe ser un soneto completo, no una lista parcial de correcciones.\n"
            "- Asegúrate de que cada candidato tenga las claves exactas "
            '"verses" y "generation_reasoning".\n'
        )

    rhyme_pattern_explanation = (
        "PATRON EXPLICITO ABBA ABBA CDC CDC:\n"
        "- Verso 1 termina con rima A.\n"
        "- Verso 2 termina con rima B.\n"
        "- Verso 3 termina con rima B.\n"
        "- Verso 4 termina con rima A.\n"
        "- Verso 5 termina con rima A.\n"
        "- Verso 6 termina con rima B.\n"
        "- Verso 7 termina con rima B.\n"
        "- Verso 8 termina con rima A.\n"
        "- Verso 9 termina con rima C.\n"
        "- Verso 10 termina con rima D.\n"
        "- Verso 11 termina con rima C.\n"
        "- Verso 12 termina con rima C.\n"
        "- Verso 13 termina con rima D.\n"
        "- Verso 14 termina con rima C.\n\n"
        "Equivalencias de rima:\n"
        "- Versos 1 y 4 deben rimar entre si.\n"
        "- Versos 2 y 3 deben rimar entre si.\n"
        "- Versos 5 y 8 deben rimar como 1 y 4.\n"
        "- Versos 6 y 7 deben rimar como 2 y 3.\n"
        "- Versos 9, 11, 12 y 14 deben compartir rima C.\n"
        "- Versos 10 y 13 deben compartir rima D.\n"
    )

    construction_strategy = (
        "ESTRATEGIA DE CONSTRUCCION:\n"
        "Antes de escribir, decide internamente:\n"
        "- dos rimas consonantes A y B para los cuartetos,\n"
        "- dos rimas consonantes C y D para los tercetos,\n"
        "- y reutilizalas estrictamente segun el patrón ABBA ABBA CDC CDC.\n"
        "No expliques esta decisión; solo úsala para construir los versos.\n"
    )

    verse_length_rules = (
        "LONGITUD DE VERSO:\n"
        "- Escribe versos breves.\n"
        "- Evita versos claramente largos.\n"
        "- Evita frases explicativas o narrativas extensas.\n"
        "- Prefiere sintagmas poéticos compactos.\n"
        "- Cada verso debe tender a 11 silabas métricas.\n"
    )

    json_format_rules = (
        "FORMATO JSON OBLIGATORIO:\n"
        "Devuelve exactamente un objeto JSON con esta forma:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "verses": ["...", "..."],\n'
        '      "generation_reasoning": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "No incluyas texto antes ni después del JSON.\n"
    )

    system_prompt = (
        "Eres un poeta y asistente de generación controlada de sonetos en un "
        "sistema de Beam Search.\n"
        "Debes proponer versiones candidatas de un soneto clásico en espanol.\n"
        "No añadas explicaciones fuera del JSON.\n"
        "Debes devolver JSON valido.\n\n"
        "RESTRICCIONES FORMALES PRIORITARIAS:\n"
        "- Exactamente 14 versos.\n"
        "- Cada verso debe intentar tener 11 sílabas métricas.\n"
        "- Rima consonante ABBA ABBA CDC CDC.\n"
        "- Sin título.\n"
        "- Sin explicaciones.\n"
        "- Sin numeración dentro de los versos.\n"
        "- Cada verso debe ser una cadena independiente.\n"
        "- Debe priorizar estructura, métrica y rima antes que belleza estética.\n"
        "- generation_reasoning debe ser breve.\n"
        "- Debe devolver una versión completa del soneto, no solo versos corregidos.\n\n"
        f"{construction_strategy}\n"
        f"{rhyme_pattern_explanation}\n"
        f"{verse_length_rules}\n"
        f"{strict_rules}\n"
        f"{json_format_rules}"
    )

    if previous_sonnet:
        summarized_feedback = summarize_feedback_for_prompt(feedback)
        user_prompt = (
            f"Tarea original:\n{question}\n\n"
            "Soneto candidato anterior, verso a verso:\n"
            f"{format_numbered_verses(previous_sonnet)}\n\n"
            "Feedback formal completo recibido:\n"
            f"{summarized_feedback}\n\n"
            f"Genera exactamente {num_candidates} versiones corregidas o "
            "mejoradas del soneto anterior.\n\n"
            "Instrucciones de corrección:\n"
            "- Devuelve una nueva versión completa del soneto, no solo los versos corregidos.\n"
            "- Mantén exactamente 14 versos siempre que sea posible.\n"
            "- Intenta que cada verso tenga 11 sílabas métricas.\n"
            "- Respeta la rima consonante ABBA ABBA CDC CDC.\n"
            "- Corrige específicamente los errores indicados en el feedback formal.\n"
            "- Conserva los aspectos que ya esten correctos y no los empeores.\n"
            "- Evita repetir literalmente el soneto anterior si el feedback contiene errores.\n"
            "- Prioriza primero estructura, métrica y rima antes que belleza estética.\n"
            "- No anadas título ni texto fuera del JSON.\n\n"
            "CORRECCION CON FEEDBACK:\n"
            "- Prioriza los errores mencionados en el feedback.\n"
            "- Si el feedback dice que un verso tiene demasiadas sílabas, acórtalo.\n"
            "- Si el feedback dice que un verso tiene pocas sílabas, alárgalo ligeramente.\n"
            "- Si falla la rima, cambia la palabra final del verso afectado.\n"
            "- Si sobran o faltan versos, devuelve exactamente 14.\n"
            "- No empeores versos o rimas que ya estaban correctos.\n\n"
            f"{construction_strategy}\n"
            f"{rhyme_pattern_explanation}\n"
            f"{verse_length_rules}\n"
            f"{json_format_rules}"
            "Responde solo con JSON valido."
        )
    else:
        user_prompt = (
            f"Tarea original:\n{question}\n\n"
            f"Genera exactamente {num_candidates} primeras versiones candidatas "
            "del soneto desde cero.\n"
            "Cada candidato debe ser un soneto completo, sin título, con 14 versos "
            "siempre que sea posible, endecasílabos y con rima consonante "
            "ABBA ABBA CDC CDC.\n\n"
            f"{construction_strategy}\n"
            f"{rhyme_pattern_explanation}\n"
            f"{verse_length_rules}\n"
            f"{json_format_rules}"
            "Responde solo con JSON valido."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_sonnet_candidates_with_ollama(
    question: str,
    previous_sonnet: list[str],
    feedback: str,
    num_candidates: int = 3,
) -> list[dict[str, Any]]:
    """
    Genera versiones candidatas de soneto usando Ollama.

    Hace un primer intento creativo y un reintento mas estricto si falla el
    parseo. Si ambos intentos fallan, devuelve candidatos fallback vacíos para
    mantener ejecutable el grafo.
    """
    attempts = [
        {
            "temperature": 0.7,
            "num_predict": 1200,
            "strict": False,
        },
        {
            "temperature": 0.1,
            "num_predict": 1200,
            "strict": True,
        },
    ]
    errors = []

    for attempt_number, attempt in enumerate(attempts, start=1):
        raw_response = ""
        messages = _build_sonnet_generation_messages(
            question=question,
            previous_sonnet=previous_sonnet,
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
            candidates = parse_sonnet_candidates_response(raw_response, num_candidates)
            generation_reasoning = (
                "Versión corregida usando feedback formal."
                if previous_sonnet
                else "Generación inicial desde cero."
            )
            for candidate in candidates:
                candidate["generation_reasoning"] = generation_reasoning
            return candidates
        except Exception as e:
            error_message = f"Intento {attempt_number}: {e}"
            errors.append(error_message)
            print(f"[WARN] Fallo generando sonetos con Ollama. {error_message}")
            if raw_response:
                print("[WARN] Respuesta recibida del generador:")
                print(raw_response)

    fallback_reason = (
        "Fallback: no se pudieron generar candidatos de soneto con Ollama. "
        f"Errores: {' | '.join(errors)}"
    )
    return [
        {
            "sonnet": [],
            "generation_reasoning": fallback_reason,
        }
        for _ in range(num_candidates)
    ]


def generate_candidates_with_ollama(
    question: str,
    path: List[str],
    num_candidates: int = 3,
) -> List[Dict[str, str]]:
    """
    Pide al modelo varias posibles continuaciones en formato JSON.

    TODO: adaptar esta utilidad para generar versiones candidatas de soneto en
    lugar de continuaciones de una trayectoria de razonamiento.
    """
    current_reasoning = "\n".join(f"- {step}" for step in path)

    system_prompt = (
        "Eres un asistente que genera siguientes pasos de razonamiento dentro de un sistema de Beam Search.\n"
        "Tu tarea es proponer posibles continuaciones útiles de una trayectoria de razonamiento.\n"
        "Debes responder únicamente con JSON válido.\n"
        "Todos los textos deben estar en español.\n"
        "No añadas explicaciones fuera del JSON.\n\n"

        "Reglas importantes:\n"
        "1. Cada candidato debe representar un avance real en la resolución.\n"
        "2. Evita repetir, reformular o resumir pasos ya presentes.\n"
        "3. Los candidatos deben ser distintos entre sí.\n"
        "4. Cada paso debe ser concreto, breve y útil.\n"
        "5. No propongas pasos triviales, vacíos o excesivamente genéricos.\n"
        "6. Si el problema es simple, no inventes complejidad innecesaria.\n"
        "7. Si el problema requiere cálculo o deducción, orienta los pasos hacia esa resolución.\n\n"

        "Devuelve un objeto JSON con esta forma exacta:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "next_step": "texto",\n'
        '      "reasoning": "explicación breve"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    user_prompt = (
        f"Pregunta original:\n{question}\n\n"
        f"Trayectoria actual de razonamiento:\n{current_reasoning}\n\n"
        f"Genera exactamente {num_candidates} posibles siguientes pasos.\n\n"
        "Condiciones:\n"
        "- cada candidato debe aportar algo nuevo,\n"
        "- no repitas ideas ya presentes en la trayectoria,\n"
        "- los candidatos deben ser diferentes entre sí,\n"
        "- cada paso debe acercar a la solución,\n"
        "- evita pasos demasiado vagos como 'pensar mejor', 'analizar más' o similares.\n\n"
        "Responde solo con JSON válido y en español."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = chat_ollama(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.2,
        num_predict=300,
    )

    try: # Metemos aquí try/except al parsear JSON
        parsed = json.loads(raw_response)
        candidates = parsed["candidates"]

        if not isinstance(candidates, list):
            raise ValueError("El campo 'candidates' no es una lista.")

        cleaned_candidates = []
        for item in candidates[:num_candidates]:
            if not isinstance(item, dict):
                continue

            next_step = str(item.get("next_step", "")).strip()
            reasoning = str(item.get("reasoning", "")).strip()

            if next_step:
                cleaned_candidates.append(
                    {
                        "next_step": next_step,
                        "reasoning": reasoning if reasoning else "Sin explicación",
                    }
                )

        if cleaned_candidates:
            return cleaned_candidates

    except Exception as e:
        print(f"[WARN] Error parseando JSON del generador: {e}")
        print("[WARN] Respuesta recibida del generador:")
        print(raw_response)

    return [
        {
            "next_step": f"{path[-1]} -> continuación fallback {i}", # Fallback si falla el parseo
            "reasoning": "Fallback por error de parseo JSON en generación.",
        }
        for i in range(num_candidates)
    ]


# =========================
# 4. Scoring con Ollama
# =========================
def score_candidate_with_ollama(
    question: str,
    candidate_path: List[str],
) -> Dict[str, Any]:
    """
    Pide al modelo una evaluación estructurada del candidato.
    Devuelve:
      - score: float entre 0 y 1
      - justification: texto breve

    TODO: adaptar esta utilidad para evaluar sonetos candidatos o sustituirla
    por evaluacion formal objetiva.
    """
    current_reasoning = "\n".join(f"- {step}" for step in candidate_path)

    system_prompt = (
        "Eres un evaluador crítico de pasos de razonamiento en un sistema de Beam Search.\n"
        "Tu tarea es evaluar la calidad de una trayectoria de razonamiento candidata.\n"
        "Debes responder únicamente con JSON válido.\n"
        "Todos los textos deben estar en español.\n"
        "No añadas explicaciones fuera del JSON.\n\n"

        "Evalúa la trayectoria según estos criterios:\n"
        "1. Coherencia: los pasos deben tener sentido entre sí.\n"
        "2. Utilidad: la trayectoria debe ayudar realmente a resolver el problema.\n"
        "3. Progreso: el último paso debe suponer un avance real, no una repetición.\n"
        "4. No redundancia: penaliza pasos repetitivos, reformulaciones innecesarias o pasos triviales.\n"
        "5. Precisión: premia pasos concretos y penaliza pasos vagos o genéricos.\n"
        "6. Corrección: penaliza cualquier paso que exprese incorrectamente una operación.\n\n"

        "IMPORTANTE SOBRE LA PUNTUACIÓN:\n"
        "- Debes asignar una puntuación continua entre 0 y 1.\n"
        "- NO uses únicamente valores redondeados como 0.7, 0.8 o 0.9 salvo que sea estrictamente necesario.\n"
        "- Usa decimales para reflejar diferencias pequeñas (por ejemplo: 0.73, 0.78, 0.81).\n"
        "- Dos trayectorias solo deben tener exactamente la misma puntuación si son prácticamente equivalentes en calidad.\n"
        "- Debes ser capaz de discriminar entre candidatos similares.\n\n"

        "Interpretación orientativa (no rígida):\n"
        "- 0.0 a 0.2: trayectoria muy mala o incorrecta.\n"
        "- 0.3 a 0.5: trayectoria débil o poco útil.\n"
        "- 0.6 a 0.8: trayectoria razonable.\n"
        "- 0.8 a 1.0: trayectoria muy buena.\n\n"

        "Sé estricto con la puntuación.\n"
        "No des notas altas a trayectorias redundantes, vagas o que no aportan avance real.\n"
        "Penaliza especialmente:\n"
        "- pasos que no avanzan la resolución,\n"
        "- pasos redundantes,\n"
        "- pasos incorrectos conceptualmente.\n\n"

        "Presta especial atención al ÚLTIMO paso:\n"
        "- debe aportar información nueva,\n"
        "- debe acercar claramente a la solución,\n"
        "- no debe ser una reformulación.\n\n"

        "Devuelve un objeto JSON con esta forma exacta:\n"
        "{\n"
        '  "score": 0.0,\n'
        '  "justification": "explicación breve"\n'
        "}\n"
    )

    user_prompt = (
        f"Pregunta original:\n{question}\n\n"
        f"Trayectoria candidata de razonamiento:\n{current_reasoning}\n\n"
        "Evalúa la calidad global de esta trayectoria intermedia.\n"
        "Presta especial atención a si el último paso:\n"
        "- aporta información nueva,\n"
        "- hace avanzar la resolución,\n"
        "- evita redundancia,\n"
        "- es concreto y útil.\n\n"
        "Devuelve una puntuación entre 0 y 1 y una justificación breve.\n"
        "Responde solo con JSON válido y en español."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = chat_ollama(
        model=SCORING_MODEL,
        messages=messages,
        temperature=0.1, # Parámetros que estudiamos justo al inicio del TFG (temperatura, creatividad, num_predict, etc.)
        num_predict=200,
    )

    try:
        parsed = json.loads(raw_response)

        raw_score = parsed.get("score", 0.0)
        justification = str(parsed.get("justification", "")).strip()

        score = float(raw_score)

        # Clamp por seguridad
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0

        return {
            "score": score,
            "justification": justification if justification else "Sin justificación",
        }

    except Exception as e:
        print(f"[WARN] Error parseando JSON del scorer: {e}")
        print("[WARN] Respuesta recibida del scorer:")
        print(raw_response)

    return {
        "score": 0.1,
        "justification": "Fallback por error de parseo JSON en scoring.",
    }


# =========================
# 5. Nodos del grafo
# =========================
def _get_nested_metric(metrics: dict[str, Any], section: str, key: str) -> Any:
    """
    Obtiene una métrica anidada de forma segura.
    """
    section_value = metrics.get(section, {})
    if not isinstance(section_value, dict):
        return None
    return section_value.get(key)


def summarize_metrics(metrics: dict[str, Any]) -> str:
    """
    Resume las métricas formales de evaluate_sonnet en una sola línea.

    El resumen es robusto ante claves ausentes para poder usarse en logs durante
    la ejecución de Beam Search.
    """
    if not isinstance(metrics, dict):
        metrics = {}

    actual_verses = _get_nested_metric(metrics, "verse_count", "actual_verses")
    expected_verses = _get_nested_metric(metrics, "verse_count", "expected_verses")
    correct_syllables = _get_nested_metric(metrics, "syllables", "correct_verses")
    correct_rhymes = _get_nested_metric(metrics, "rhyme", "correct_positions")
    score_20 = metrics.get("score_20")

    actual_verses_text = actual_verses if actual_verses is not None else "?"
    expected_verses_text = expected_verses if expected_verses is not None else "?"
    correct_syllables_text = (
        correct_syllables if correct_syllables is not None else "?"
    )
    correct_rhymes_text = correct_rhymes if correct_rhymes is not None else "?"
    score_20_text = score_20 if score_20 is not None else "?"

    return (
        f"versos={actual_verses_text}/{expected_verses_text} | "
        f"endecasílabos={correct_syllables_text}/14 | "
        f"rima={correct_rhymes_text}/14 | "
        f"score={score_20_text}/20"
    )


def build_trace_entry(item: dict[str, Any]) -> dict[str, Any]:
    """
    Construye una entrada compacta de traza para un candidato o beam.
    """
    metrics = item.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    sonnet = item.get("sonnet", [])
    if not isinstance(sonnet, list):
        sonnet = []

    score_history = item.get("score_history", [])
    if not isinstance(score_history, list):
        score_history = []

    return {
        "sonnet": sonnet,
        "score": item.get("score", 0.0),
        "step_score": item.get("step_score", 0.0),
        "score_history": score_history,
        "metrics_summary": summarize_metrics(metrics),
        "feedback": item.get("feedback", ""),
        "generation_reasoning": item.get("generation_reasoning", ""),
    }


def expand_node(state: BeamSearchState) -> dict:
    all_candidates: List[Dict[str, Any]] = []

    print(f"\n=== EXPAND STEP {state['step']} ===")

    for beam_index, beam in enumerate(state["beams"]):
        previous_sonnet = beam.get("sonnet", [])
        if not isinstance(previous_sonnet, list):
            previous_sonnet = []
        current_feedback = beam.get(
            "feedback",
            "Todavía no se ha generado ningún soneto.",
        )
        uses_feedback = bool(previous_sonnet and current_feedback)

        print(f"\nBeam origen {beam_index}:")
        print(
            "  Modo: "
            + (
                "corrección de soneto previo"
                if previous_sonnet
                else "generación inicial desde cero"
            )
        )
        print(f"  Score actual: {float(beam.get('score', 0.0)):.3f}")
        print(f"  Usa feedback para corregir: {uses_feedback}")

        generated_sonnets = generate_sonnet_candidates_with_ollama(
            question=state["question"],
            previous_sonnet=previous_sonnet,
            feedback=current_feedback,
            num_candidates=2,
        )

        for generated in generated_sonnets:
            candidate = {
                "sonnet": generated["sonnet"],
                "score": beam.get("score", 0.0),
                "score_history": beam.get("score_history", []).copy(),
                "feedback": beam.get(
                    "feedback",
                    "Todavía no se ha generado ningún soneto.",
                ),
                "metrics": beam.get("metrics", {}).copy(),
                "generation_reasoning": generated["generation_reasoning"],
            }
            all_candidates.append(candidate)

    print(f"Candidatos generados: {len(all_candidates)}")
    for i, candidate in enumerate(all_candidates):
        print(
            f"  Candidato {i}: "
            f"{len(candidate.get('sonnet', []))} versos generados"
        )

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
    scored_candidates: List[Dict[str, Any]] = []

    print(f"=== SCORE STEP {state['step']} ===")

    for i, candidate in enumerate(state["candidates"]):
        # El scoring actual es objetivo y se basa en metricas formales:
        # extension, computo silabico y rima. En una fase futura se podra
        # combinar con una evaluacion subjetiva mediante LLM-as-a-judge.
        sonnet = candidate.get("sonnet", [])
        if not isinstance(sonnet, list):
            sonnet = []
        evaluation = evaluate_sonnet(sonnet)

        step_score = float(evaluation.get("score", 0.0))
        updated_history = candidate.get("score_history", []) + [step_score]
        total_score = aggregate_scores(updated_history, alpha=ALPHA)

        score_20 = evaluation.get("score_20", round(step_score * 20, 2))
        errors = evaluation.get("errors", [])
        error_count = len(errors) if isinstance(errors, list) else 0

        print(f"\nCandidato {i}:")
        print(f"Número de versos: {len(sonnet)}")
        print(f"Score formal: {step_score:.3f} ({score_20}/20)")
        print(f"Errores detectados: {error_count}")
        print(f"Resumen métricas: {summarize_metrics(evaluation)}")

        scored_candidate = {
            "sonnet": sonnet.copy() if isinstance(sonnet, list) else [],
            "score": total_score,
            "step_score": step_score,
            "score_history": updated_history,
            "feedback": str(evaluation.get("feedback", "")),
            "metrics": evaluation,
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
    print(f"=== PRUNE STEP {state['step']} ===")

    sorted_candidates = sorted(
        state["candidates"],
        key=lambda beam: beam["score"],
        reverse=True,
    )

    top_k = sorted_candidates[: state["k"]]

    for i, beam in enumerate(top_k):
        print(f"\nBeam {i}:")
        print(f"Score global: {beam['score']:.3f}")
        print(f"Score último paso: {beam.get('step_score', 0.0):.3f}")
        print(f"Historial scores: {[round(s, 3) for s in beam.get('score_history', [])]}")
        print(f"Resumen métricas: {summarize_metrics(beam.get('metrics', {}))}")

        print("Primeros versos del soneto:")
        if beam.get("sonnet"):
            for verse_number, verse in enumerate(beam["sonnet"][:2], start=1):
                print(f"  {verse_number}. {verse}")
        else:
            print("  [Sin soneto generado todavía]")

        print("Razón de generación del último paso:")
        print("  ", beam.get("generation_reasoning", "Sin razón de generación."))

        feedback = beam.get("feedback", "Sin feedback formal todavía.")
        feedback_summary = str(feedback).splitlines()[0] if feedback else ""
        print("Feedback resumido:")
        print("  ", feedback_summary)

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

# ====================================================================================
# 6. Función auxiliar de agregación (para integrar la fórmula que me ha pasado Sergio)
# ====================================================================================

def aggregate_scores(score_history: List[float], alpha: float = ALPHA) -> float:
    """
    Agrega scores de calidad en [0,1] usando media geométrica normalizada.
    """
    if not score_history:
        return 0.0

    product = 1.0
    for s in score_history:
        safe_s = max(s, EPSILON)
        product *= safe_s

    T = len(score_history)
    return product ** (1.0 / (T ** alpha))

# =========================
# 6. Decisión de continuación
# =========================
def should_continue(state: BeamSearchState) -> str:
    if state["step"] >= state["max_steps"]:
        beams = state.get("beams", [])
        best_beam = max(beams, key=lambda beam: beam.get("score", 0.0)) if beams else {}
        best_score = float(best_beam.get("score", 0.0))
        trace = state.get("trace", [])
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

    beams = state.get("beams", [])
    if not beams:
        return "continue"

    best_beam = max(beams, key=lambda beam: beam.get("score", 0.0))
    best_score = float(best_beam.get("score", 0.0))
    if best_score >= TARGET_SCORE:
        trace = state.get("trace", [])
        trace.append(
            {
                "step": state["step"],
                "phase": "stop",
                "reason": "target_score",
                "best_score": best_score,
            }
        )
        state["trace"] = trace
        print(f"Fin: alcanzado TARGET_SCORE con score {best_score:.3f}.")
        return "end"

    return "continue"


def save_final_result(
    best_beam: dict[str, Any],
    trace: list[dict[str, Any]],
    output_dir: str = "outputs",
) -> None:
    """
    Guarda el mejor soneto final, sus métricas y la traza completa en archivos.

    Crea la carpeta de salida si no existe y usa un timestamp para evitar
    sobrescribir resultados de ejecuciones anteriores.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sonnet_path = output_path / f"final_sonnet_{timestamp}.txt"
    metrics_path = output_path / f"final_metrics_{timestamp}.json"
    trace_path = output_path / f"final_trace_{timestamp}.json"

    sonnet = best_beam.get("sonnet", [])
    if not isinstance(sonnet, list):
        sonnet = []

    metrics = best_beam.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    score_history = best_beam.get("score_history", [])
    if not isinstance(score_history, list):
        score_history = []

    text_lines = [
        "SONETO FINAL",
        "",
        f"Score global: {float(best_beam.get('score', 0.0)):.3f}",
        f"Historial de scores: {[round(float(score), 3) for score in score_history]}",
        f"Resumen de métricas: {summarize_metrics(metrics)}",
        "",
        "Feedback final:",
        str(best_beam.get("feedback", "Sin feedback formal todavía.")),
        "",
        "Soneto completo:",
    ]

    if sonnet:
        text_lines.extend(
            f"{verse_number}. {verse}"
            for verse_number, verse in enumerate(sonnet, start=1)
        )
    else:
        text_lines.append("[Sin soneto generado todavía]")

    sonnet_path.write_text("\n".join(text_lines), encoding="utf-8")

    json_payload = {
        "sonnet": sonnet,
        "score": best_beam.get("score", 0.0),
        "step_score": best_beam.get("step_score", 0.0),
        "score_history": score_history,
        "feedback": best_beam.get("feedback", ""),
        "metrics": metrics,
        "generation_reasoning": best_beam.get("generation_reasoning", ""),
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, ensure_ascii=False, indent=2)

    with trace_path.open("w", encoding="utf-8") as file:
        json.dump(trace, file, ensure_ascii=False, indent=2)

    print("\nArchivos generados:")
    print(f"  Soneto final: {sonnet_path}")
    print(f"  Métricas: {metrics_path}")
    print(f"  Traza: {trace_path}")


# =========================
# 7. Main
# =========================
def main():
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
            "Escribe un soneto clásico en español sobre el paso del tiempo y "
            "la memoria. Debe tener exactamente 14 versos endecasílabos y "
            "rima consonante ABBA ABBA CDC CDC."
        ),
        "beams": [
            {
                "sonnet": [],
                "score": 0.0,
                "score_history": [],
                "feedback": "Todavía no se ha generado ningún soneto.",
                "metrics": {},
                "generation_reasoning": "Estado inicial sin soneto.",
            }
        ],
        "candidates": [],
        "trace": [],
        "step": 0,
        "max_steps": 3,
        "k": 2,
    }

    result = graph.invoke(initial_state)

    print("\n=== RESULTADO FINAL ===")
    for i, beam in enumerate(result["beams"]):
        print(f"\nBeam final {i}:")
        print(f"Score global final: {beam['score']:.3f}")
        print(f"Historial scores: {[round(s, 3) for s in beam['score_history']]}")

        print("Feedback final:")
        print("  ", beam.get("feedback", "Sin feedback formal todavía."))

        print("Soneto final:")
        if beam.get("sonnet"):
            for verse_number, verse in enumerate(beam["sonnet"], start=1):
                print(f"  {verse_number}. {verse}")
        else:
            print("  [Sin soneto generado todavía]")

        print("Última razón de generación:")
        print("  ", beam["generation_reasoning"])

    if result["beams"]:
        # Guardamos el mejor resultado final y la traza para su analisis posterior.
        save_final_result(result["beams"][0], result.get("trace", []))


def run_cleaning_smoke_tests() -> None:
    """
    Ejecuta pruebas rápidas de limpieza superficial de versos generados.
    """
    examples = [
        "1. En sombra vuelve la memoria",
        "02) El tiempo late todavía",
        "\"La tarde guarda su secreto\"",
    ]

    print("=== PRUEBAS DE LIMPIEZA DE VERSOS ===")
    for example in examples:
        print(f"{example!r} -> {clean_generated_verse(example)!r}")


if __name__ == "__main__":
    if "--test-cleaning" in sys.argv:
        run_cleaning_smoke_tests()
    else:
        main()
