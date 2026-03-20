from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# PROGRAMA TAMBIÉN DE PRUEBA
# PUEDE ELIMINARSE SIN PROBLEMA

# ¿Qué estamos comprobando aquí?
# Cada estado ya no guarda solo números, sino "pensamientos"
# Un nodo representa una iteración de razonamiento
# El bucle simula que el agente piensa varios pasos antes de terminar

# Estado compartido del grafo
class FakeLLMState(TypedDict):
    counter: int
    history: List[str]


# Nodo que simula un "pensamiento" de un LLM
def fake_llm_node(state: FakeLLMState) -> dict:
    step = state["counter"]

    thought = f"Paso {step}: estoy razonando sobre el problema."

    print(thought)

    return {
        "counter": step + 1,
        "history": state["history"] + [thought],
    }


# Decisión: seguir pensando o terminar
def should_continue(state: FakeLLMState) -> str:
    if state["counter"] < 5:
        return "continue"
    return "end"


def main():
    builder = StateGraph(FakeLLMState)

    builder.add_node("think", fake_llm_node)

    builder.set_entry_point("think")

    builder.add_conditional_edges(
        "think",
        should_continue,
        {
            "continue": "think",
            "end": END,
        },
    )

    graph = builder.compile()

    initial_state = {
        "counter": 0,
        "history": [],
    }

    result = graph.invoke(initial_state)

    print("\n--- HISTORIAL FINAL ---")
    for item in result["history"]:
        print(item)

    print("\nEstado final:", result)


if __name__ == "__main__":
    main()