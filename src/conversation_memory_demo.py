# src/conversation_memory_demo.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


BASE_URL = "http://localhost:11434"
MODEL = "llama3"

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RUNS_DIR / "conversation_memory_demo.jsonl"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat(messages: List[Dict[str, str]], *, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Llamada a /api/chat.
    - messages contiene el historial con roles: system/user/assistant
    - stream=False para recibir una sola respuesta final (más fácil de depurar)
    """
    payload: Dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options

    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "latency_ms": latency_ms,
        "raw": data,
        # Según la API, la respuesta del assistant viene en data["message"]["content"]
        "assistant_text": (data.get("message", {}) or {}).get("content", "").strip(),
    }


if __name__ == "__main__":
    # 1) Creamos el historial.
    #    (Opcional pero recomendable) un system message define reglas de estilo/idioma.
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": "Responde en español de forma breve y clara.",
        }
    ]

    # 2) Turno 1: “¿Qué es una pera?”
    user_1 = "¿Qué es una pera?"
    messages.append({"role": "user", "content": user_1})

    resp_1 = chat(messages, options={"temperature": 0.2, "num_predict": 120})
    assistant_1 = resp_1["assistant_text"]
    messages.append({"role": "assistant", "content": assistant_1})

    print("TURNO 1 (assistant):")
    print(assistant_1)
    print(f"(latency_ms={resp_1['latency_ms']})\n")

    append_jsonl(LOG_FILE, {
        "turn": 1,
        "user": user_1,
        "assistant": assistant_1,
        "latency_ms": resp_1["latency_ms"],
        "messages_sent": messages,  # incluye el historial tras el turno
        "raw": resp_1["raw"],
    })

    # 3) Turno 2: “¿Y de qué color es?”
    user_2 = "¿Y de qué color es?"
    messages.append({"role": "user", "content": user_2})

    resp_2 = chat(messages, options={"temperature": 0.2, "num_predict": 120})
    assistant_2 = resp_2["assistant_text"]
    messages.append({"role": "assistant", "content": assistant_2})

    print("TURNO 2 (assistant):")
    print(assistant_2)
    print(f"(latency_ms={resp_2['latency_ms']})\n")

    append_jsonl(LOG_FILE, {
        "turn": 2,
        "user": user_2,
        "assistant": assistant_2,
        "latency_ms": resp_2["latency_ms"],
        "messages_sent": messages,  # incluye el historial tras el turno
        "raw": resp_2["raw"],
    })

    print(f"Log guardado en: {LOG_FILE}")