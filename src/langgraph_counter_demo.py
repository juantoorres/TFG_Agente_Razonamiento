from typing import TypedDict
from langgraph.graph import StateGraph, END
import warnings

# ESTE ARCHIVO ES SIMPLEMENTE PARA ENTENDER LANGGRAPH
# PROGRAMA QUE HACE UN CONTADOR, PARA ESTUDIAR EL SALTO ENTRE LOS NODOS Y TAL, NO ES RELEVANTE
# PUEDE ELIMINARSE SIN PROBLEMA

warnings.filterwarnings("ignore") # Me sale un 'warning' con la versión del paquete 'Core Pydantic V1' de la librería 'langgraph'. Lo ignoramos


# 1. Definimos el estado compartido del grafo
class CounterState(TypedDict):
    counter: int


# 2. Nodo: suma 1 al contador
def increment_node(state: CounterState) -> dict:
    current = state["counter"]
    print(f"Valor actual del contador: {current}")
    return {"counter": current + 1}


# 3. Función de decisión: ¿seguimos o terminamos?
def should_continue(state: CounterState) -> str:
    if state["counter"] < 5:
        return "continue"
    return "end"


def main():
    # 4. Construimos el grafo
    builder = StateGraph(CounterState)

    # Añadimos el nodo
    builder.add_node("increment", increment_node)

    # Marcamos el nodo de entrada
    builder.set_entry_point("increment")

    # Añadimos la lógica condicional
    builder.add_conditional_edges(
        "increment",
        should_continue,
        {
            "continue": "increment",
            "end": END,
        },
    )

    # 5. Compilamos el grafo
    graph = builder.compile()

    # 6. Ejecutamos el grafo con el estado inicial
    result = graph.invoke({"counter": 0})

    print("\nEjecución terminada.")
    print("Estado final:", result)


if __name__ == "__main__":
    main()