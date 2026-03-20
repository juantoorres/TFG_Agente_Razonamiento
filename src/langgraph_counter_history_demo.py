from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# PROGRAMA TAMBIÉN DE PRUEBA
# AHORA UN ESTADO NO ES SIMPLEMENTE UN counter:int, SINO QUE TAMBIÉN TIENE MEMORIA history:List[str]
# PUEDE ELIMINARSE SIN PROBLEMA

# Antes solo había un número, no se guardaba nada
# Ahora se guarda TODO lo que pasa en cada paso

# 1. Estado con memoria
class CounterState(TypedDict):
    counter: int
    history: List[str]


# 2. Nodo: suma + guarda en historial
def increment_node(state: CounterState) -> dict:
    current = state["counter"]

    print(f"Valor actual: {current}")

    new_value = current + 1

    # Añadimos un mensaje al historial
    new_entry = f"He pasado de {current} a {new_value}"

    return {
        "counter": new_value,
        "history": state["history"] + [new_entry],
    }


# 3. Condición de parada
def should_continue(state: CounterState) -> str:
    if state["counter"] < 5:
        return "continue"
    return "end"


def main():
    builder = StateGraph(CounterState)

    builder.add_node("increment", increment_node)

    builder.set_entry_point("increment")

    builder.add_conditional_edges(
        "increment",
        should_continue,
        {
            "continue": "increment",
            "end": END,
        },
    )

    graph = builder.compile()

    # Estado inicial con historial vacío
    result = graph.invoke({
        "counter": 0,
        "history": []
    })

    print("\n--- HISTORIAL ---")
    for step in result["history"]:
        print(step)


if __name__ == "__main__":
    main()