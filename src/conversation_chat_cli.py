from __future__ import annotations

import requests
from typing import List, Dict

BASE_URL = "http://localhost:11434"
MODEL = "llama3"


def trim_memory(messages: List[Dict[str, str]], max_turns: int = 6):
    """
    Mantiene el mensaje 'system' y solo los últimos N turnos
    (un turno = user + assistant).
    """
    system = messages[0]

    conversation = messages[1:]

    # Cada turno son 2 mensajes (usuario-modelo)
    conversation = conversation[-max_turns * 2:]

    return [system] + conversation


def chat(messages: List[Dict[str, str]]):

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 200,
        },
    }

    r = requests.post(f"{BASE_URL}/api/chat", json=payload)
    r.raise_for_status()

    data = r.json()

    return data["message"]["content"].strip()


if __name__ == "__main__":

    messages = [
        {
            "role": "system",
            "content": "Responde en español de forma breve y clara."
        }
    ]

    print("\nChat con Ollama. Escribe 'salir' para terminar.\n")

    while True:

        user_input = input("Usuario: ")

        if user_input.lower() in ["salir", "exit", "quit"]:
            break

        # Añadimos mensaje del usuario
        messages.append({
            "role": "user",
            "content": user_input
        })

        # Recortamos memoria si es demasiado larga (puede aumentar notablemente el coste en tokens --> más costoso y lento)
        messages = trim_memory(messages)

        # Llamamos al modelo
        response = chat(messages)

        print("\nModelo:", response, "\n")

        # Guardamos respuesta del modelo en memoria (Se lo volveremos a pasar todo junto como contexto en la siguiente interacción)
        messages.append({
            "role": "assistant",
            "content": response
        })