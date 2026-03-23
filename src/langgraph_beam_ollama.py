from __future__ import annotations

import json
import requests

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END


BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "llama3"
SCORING_MODEL = "llama3"


# =========================
# 1. Estado del grafo
# =========================
class BeamSearchState(TypedDict):
    question: str
    beams: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    step: int
    max_steps: int
    k: int


# =========================
# 2. Utilidad general Ollama
# =========================
def chat_ollama(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, num_predict: int = 300) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=90)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


# =========================
# 3. Generación con Ollama
# =========================
def generate_candidates_with_ollama(
    question: str,
    path: List[str],
    num_candidates: int = 3,
) -> List[Dict[str, str]]:
    """
    Pide al modelo varias posibles continuaciones en formato JSON.
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
        "5. Precisión: premia pasos concretos y penaliza pasos vagos o genéricos.\n\n"

        "Interpretación de la puntuación:\n"
        "- 0.0 a 0.2: trayectoria muy mala, irrelevante o claramente redundante.\n"
        "- 0.3 a 0.5: trayectoria débil, con utilidad limitada.\n"
        "- 0.6 a 0.8: trayectoria razonable y útil.\n"
        "- 0.9 a 1.0: trayectoria muy buena, clara, útil y con avance real.\n\n"

        "Sé estricto con la puntuación.\n"
        "No des notas altas a trayectorias redundantes, vagas o que no aportan avance real.\n"
        "Si el último paso repite esencialmente una idea ya presente, debes penalizarlo.\n"
        "Si la trayectoria parece correcta pero innecesariamente verbosa o circular, no debe superar 0.6 o 0.7.\n\n"

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
def expand_node(state: BeamSearchState) -> dict:
    all_candidates: List[Dict[str, Any]] = []

    print(f"\n=== EXPAND STEP {state['step']} ===")

    for beam in state["beams"]:
        current_path = beam["path"]

        generated_steps = generate_candidates_with_ollama(
            question=state["question"],
            path=current_path,
            num_candidates=3,
        )

        for generated in generated_steps:
            candidate = {
                "path": current_path + [generated["next_step"]],
                "score": beam["score"],
                "generation_reasoning": generated["reasoning"],
                "step_justification": None,
            }
            all_candidates.append(candidate)

    print(f"Candidatos generados: {len(all_candidates)}")

    return {"candidates": all_candidates}


def score_node(state: BeamSearchState) -> dict:
    scored_candidates: List[Dict[str, Any]] = []

    print(f"=== SCORE STEP {state['step']} ===")

    for candidate in state["candidates"]:
        score_result = score_candidate_with_ollama(
            question=state["question"],
            candidate_path=candidate["path"],
        )

        scored_candidate = {
            "path": candidate["path"],
            "score": candidate["score"] + score_result["score"], # Ahora mismo trayectorias largas pueden ganar por acumulación, aunque sus últimos pasos no sean especialmente buenos
            "generation_reasoning": candidate["generation_reasoning"],
            "step_justification": score_result["justification"],
        }
        scored_candidates.append(scored_candidate)

    return {"candidates": scored_candidates}


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
        print(f"Score acumulado: {beam['score']:.3f}")
        print("Path:")
        for step_text in beam["path"]:
            print("  ", step_text)
        print("Razón de generación del último paso:")
        print("  ", beam["generation_reasoning"])
        print("Justificación de scoring del último paso:")
        print("  ", beam["step_justification"])

    return {
        "beams": top_k,
        "step": state["step"] + 1,
    }


# =========================
# 6. Decisión de continuación
# =========================
def should_continue(state: BeamSearchState) -> str:
    if state["step"] >= state["max_steps"]:
        return "end"
    return "continue"


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
        "question": "En una clase hay 12 alumnos. Llegan 3 más y luego se van 4. ¿Cuántos quedan? Razona paso a paso.",
        "beams": [
            {
                "path": ["Inicio"],
                "score": 0.0,
                "generation_reasoning": "Estado inicial",
                "step_justification": "Sin evaluar todavía",
            }
        ],
        "candidates": [],
        "step": 0,
        "max_steps": 3,
        "k": 2,
    }

    result = graph.invoke(initial_state)

    print("\n=== RESULTADO FINAL ===")
    for i, beam in enumerate(result["beams"]):
        print(f"\nBeam final {i}:")
        print(f"Score final: {beam['score']:.3f}")
        print("Path final:")
        for step_text in beam["path"]:
            print("  ", step_text)
        print("Última razón de generación:")
        print("  ", beam["generation_reasoning"])
        print("Última justificación de scoring:")
        print("  ", beam["step_justification"])


if __name__ == "__main__":
    main()