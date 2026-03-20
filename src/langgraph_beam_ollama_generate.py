from __future__ import annotations

import json
import random
import requests

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END


BASE_URL = "http://localhost:11434"
MODEL = "llama3"


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
# 2. Llamada a Ollama
# =========================
def chat_ollama(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 300,
        },
    }

    response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


def generate_candidates_with_ollama(question: str, path: List[str], num_candidates: int = 3) -> List[Dict[str, str]]:
    """
    Pide al modelo varias posibles continuaciones en formato JSON.
    """
    current_reasoning = "\n".join(f"- {step}" for step in path)

    system_prompt = (
        "Eres un asistente que genera siguientes pasos de razonamiento.\n"
        "Responde exclusivamente en español.\n" # Lo añado ya que aunque le paso todo esto en español, el modelo me ha respondido en inglés. ¿Forzar a usar español?
        "Debes responder únicamente con JSON válido.\n"
        "No añadas explicaciones fuera del JSON.\n"
        "Devuelve un objeto con esta forma exacta:\n"
        '{\n'
        '  "candidates": [\n'
        '    {\n'
        '      "next_step": "texto",\n'
        '      "reasoning": "explicación breve"\n'
        '    }\n'
        '  ]\n'
        '}\n'
    )

    user_prompt = (
        f"Pregunta original:\n{question}\n\n"
        f"Razonamiento actual:\n{current_reasoning}\n\n"
        f"Genera exactamente {num_candidates} posibles siguientes pasos de razonamiento.\n"
        "Cada paso debe ser breve, coherente y distinto de los demás.\n"
        "Responde solo con JSON válido."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = chat_ollama(messages)

    try:
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
        print(f"[WARN] Error parseando JSON de Ollama: {e}")
        print("[WARN] Respuesta recibida:")
        print(raw_response)

    # Fallback simple por si el modelo no respeta el JSON
    return [
        {
            "next_step": f"{path[-1]} -> continuación fallback {i}",
            "reasoning": "Fallback por error de parseo JSON.",
        }
        for i in range(num_candidates)
    ]


# =========================
# 3. Score mock
# =========================
def mock_score(_: str) -> Dict[str, Any]:
    score_value = random.random()

    if score_value > 0.75:
        justification = "La propuesta parece muy prometedora."
    elif score_value > 0.4:
        justification = "La propuesta es razonable, aunque no especialmente fuerte."
    else:
        justification = "La propuesta parece débil frente a otras alternativas."

    return {
        "score": score_value,
        "justification": justification,
    }


# =========================
# 4. Nodos del grafo
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
        last_text = candidate["path"][-1]
        score_result = mock_score(last_text)

        scored_candidate = {
            "path": candidate["path"],
            "score": candidate["score"] + score_result["score"],
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
# 5. Decisión de continuación
# =========================
def should_continue(state: BeamSearchState) -> str:
    if state["step"] >= state["max_steps"]:
        return "end"
    return "continue"


# =========================
# 6. Main
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
        "question": "Un alumno tiene 3 manzanas y compra 2 más. Explica paso a paso cómo resolver cuántas tiene en total.",
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