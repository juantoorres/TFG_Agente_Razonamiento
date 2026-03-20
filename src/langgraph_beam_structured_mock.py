import random
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END


# PROGRAMA DE PRUEBA EN EL QUE LA SALIDA TIENE UNA ESTRUCTURA YA ALGO SIMILAR A LA QUE QUEREMOS QUE TENGA (JSON) CUANDO LO CONECTEMOS CON OLLAMA
# PUEDE ELIMINARSE SIN PROBLEMA


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
# 2. Funciones mock estructuradas
# =========================
def mock_generate(path: List[str], num_candidates: int = 3) -> List[Dict[str, str]]:
    """
    Simula la salida estructurada de un LLM generador.
    Cada candidato devuelve:
      - next_step: siguiente paso propuesto
      - reasoning: breve explicación de por qué propone ese paso
    """
    last_step = path[-1]

    outputs = []
    for i in range(num_candidates):
        outputs.append(
            {
                "next_step": f"{last_step} -> opción {i}",
                "reasoning": f"Propongo la opción {i} como posible continuación del razonamiento.",
            }
        )

    return outputs


def mock_score(candidate_text: str) -> Dict[str, Any]:
    """
    Simula la salida estructurada de un evaluador.
    Devuelve:
      - score: puntuación numérica
      - justification: justificación textual
    """
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
# 3. Nodos del grafo
# =========================
def expand_node(state: BeamSearchState) -> dict:
    """
    Expande cada beam actual en varios candidatos estructurados.
    """
    all_candidates: List[Dict[str, Any]] = []

    print(f"\n=== EXPAND STEP {state['step']} ===")

    for beam in state["beams"]:
        current_path = beam["path"]
        generated_steps = mock_generate(current_path, num_candidates=3)

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
    """
    Puntúa cada candidato y añade justificación estructurada.
    """
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
    """
    Ordena candidatos y conserva solo los k mejores.
    """
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
# 4. Decisión de continuación
# =========================
def should_continue(state: BeamSearchState) -> str:
    if state["step"] >= state["max_steps"]:
        return "end"
    return "continue"


# =========================
# 5. Programa principal
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
        "question": "Inicio",
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