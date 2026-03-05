# src/json_mode_demo.py
from __future__ import annotations

import json
import time
from typing import Any, Dict

import requests

# Añadimos el campo 'format' al request
# En vez de permitir el texto libre, le pedimos: "devuélveme un JSON"
# El programa parsea con 'json.loads()' y comprueba que es un JSON válido.

# Esto conecta directamente con lo que me dijo Sergio: "Si quieres que genere un JSON o texto (muy importante para obtener salidas estructuradas)"

BASE_URL = "http://localhost:11434"
MODEL = "llama3"


def generate_json(prompt: str, *, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Llama a /api/generate pidiendo que la respuesta sea JSON.
    Ojo: además de format="json", es importante que el prompt instruya claramente
    el formato esperado.
    """
    payload: Dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",  # <-- modo JSON simple (después lo hacemos también con el format: SCHEMA)
    }
    if options:
        payload["options"] = options

    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # El texto del modelo (data["response"]) DEBERÍA ser un JSON.
    raw_text = (data.get("response") or "").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            "El modelo NO devolvió un JSON válido.\n"
            f"Error JSON: {e}\n"
            f"Texto devuelto:\n{raw_text}"
        )

    return {
        "latency_ms": latency_ms,
        "parsed_json": parsed,
        "raw_response_text": raw_text,
        "ollama_raw": data,
    }


if __name__ == "__main__":
    prompt = """
Devuelve SOLO un JSON (sin texto extra) con estas claves:
- definition: string (2 frases)
- example: string (1 ejemplo muy simple de NLP)

Tema: beam search
""".strip()

    result = generate_json(
        prompt,
        options={"temperature": 0.2, "num_predict": 200},
    )

    print("JSON parseado correctamente:\n")
    print(json.dumps(result["parsed_json"], ensure_ascii=False, indent=2))
    print(f"\n(latency_ms={result['latency_ms']}, model={MODEL})")
