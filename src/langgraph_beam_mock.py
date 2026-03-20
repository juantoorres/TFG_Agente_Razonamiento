import random
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

# ========== ¿Qué tenemos en este archivo? ============

# - beams           = ramas activas (que siguen vivas tras la poda)
# - candidates      = expansiones temporales (no sobreviven entre iteraciones; se reemplazan cada vez)
# - expand_node     = genera continuaciones (luego llamará a Ollama para proponer siguientes pasos de razonamiento)
# - score_node      = puntúa (ahora usa random.random(), después usará un LLM para puntuar cada nodo)
# - prune_node      = se queda con las k mejores (ordena y conserva top-k)
# - should_continue = decide si seguir o terminar
# - step            = porfundidad actual del razonamiento


# =========== ¿Qué debemos entender? ===============

# Hay una diferencia importante respecto al beam search “normal” de Python (archivo 'beam_search_mock.py'):

# - antes hacías todo dentro de una función beam_search(...)
# - ahora el algoritmo está partido en nodos del grafo

# - Pero la lógica es la misma:

# - expandir -> puntuar -> podar -> decidir -> repetir


# ============= ¿Por qué hemos separado 'expand -> score -> prune' en lugar de meterlo todo en un único nodo? ==============

# - Podemos depurar cada fase por separado
# - Podemos cambiar mocks por Ollama sin rehacer la arquitectura
# - Nos será mucho más fácil explicarlo en la memoria (del TFG)

# TAMBIÉN ES DE PRUEBA, PUEDE ELIMINARSE


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
# 2. Funciones mock
# =========================
def mock_generate(path: List[str], num_candidates: int = 3) -> List[str]:
    """
    Simula la generación de varios siguientes pasos
    para una rama concreta.
    """
    last_step = path[-1]
    return [f"{last_step} -> opción {i}" for i in range(num_candidates)]


def mock_score(_: str) -> float:
    """
    Simula una puntuación entre 0 y 1.
    """
    return random.random()


# =========================
# 3. Nodos del grafo
# =========================
def expand_node(state: BeamSearchState) -> dict:
    """
    Expande cada beam actual en varios candidatos.
    Aquí todavía no se puntúa nada.
    """
    all_candidates: List[Dict[str, Any]] = []

    print(f"\n=== EXPAND STEP {state['step']} ===")

    for beam in state["beams"]:
        current_path = beam["path"]

        generated_steps = mock_generate(current_path, num_candidates=3)

        for next_step in generated_steps:
            candidate = {
                "path": current_path + [next_step],
                "score": beam["score"],  # aún no sumamos nada
            }
            all_candidates.append(candidate)

    print(f"Candidatos generados: {len(all_candidates)}")

    return {"candidates": all_candidates}


def score_node(state: BeamSearchState) -> dict:
    """
    Asigna score a cada candidato.
    """
    scored_candidates: List[Dict[str, Any]] = []

    print(f"=== SCORE STEP {state['step']} ===")

    for candidate in state["candidates"]:
        last_text = candidate["path"][-1]
        step_score = mock_score(last_text)

        scored_candidate = {
            "path": candidate["path"],
            "score": candidate["score"] + step_score,
        }
        scored_candidates.append(scored_candidate)

    return {"candidates": scored_candidates}


def prune_node(state: BeamSearchState) -> dict:
    """
    Ordena candidatos y conserva solo los k mejores
    como beams para la siguiente iteración.
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
        print(f"Score: {beam['score']:.3f}")
        print("Path:")
        for step_text in beam["path"]:
            print("  ", step_text)

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
            }
        ],
        "candidates": [],
        "step": 0,
        "max_steps": 3, # Máximo 3 pasos
        "k": 2, # Nos quedamos solo con los 2 mejores
    }

    result = graph.invoke(initial_state)

    print("\n=== RESULTADO FINAL ===")
    for i, beam in enumerate(result["beams"]):
        print(f"\nBeam final {i}:")
        print(f"Score final: {beam['score']:.3f}")
        for step_text in beam["path"]:
            print("  ", step_text)


if __name__ == "__main__":
    main()

