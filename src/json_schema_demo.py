# src/json_schema_demo.py
from __future__ import annotations

import json
import time
from typing import Any, Dict

import requests


BASE_URL = "http://localhost:11434"
MODEL = "llama3"


SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "definition": {"type": "string"},
        "example": {"type": "string"},
    },
    "required": ["definition", "example"],
    "additionalProperties": False,
}


def generate_structured(prompt: str, *, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": SCHEMA,  # <-- JSON schema (structured outputs)
    }
    if options:
        payload["options"] = options

    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    latency_ms = int((time.perf_counter() - t0) * 1000)

    raw_text = (data.get("response") or "").strip()
    parsed = json.loads(raw_text)  # debería ser JSON válido

    # Validación mínima (sin librerías externas):
    if not isinstance(parsed, dict):
        raise RuntimeError("El resultado no es un objeto JSON.")
    for k in SCHEMA["required"]:
        if k not in parsed:
            raise RuntimeError(f"Falta la clave requerida: {k}")
    extra = set(parsed.keys()) - set(SCHEMA["properties"].keys())
    if extra:
        raise RuntimeError(f"Claves extra no permitidas: {sorted(extra)}")

    return {
        "latency_ms": latency_ms,
        "parsed_json": parsed,
        "raw_response_text": raw_text,
        "ollama_raw": data,
    }


if __name__ == "__main__":
    # Recomendación de Ollama: además de pasar el schema en format,
    # también guiar con el schema en el prompt para “anclar” el formato. :contentReference[oaicite:3]{index=3}
    prompt = f"""
Devuelve un JSON que cumpla este JSON Schema:
{json.dumps(SCHEMA, ensure_ascii=False)}

Rellena:
- definition: 2 frases sobre qué es beam search
- example: 1 ejemplo muy simple de NLP
""".strip()

    result = generate_structured(
        prompt,
        options={"temperature": 0.2, "num_predict": 200},
    )

    print(json.dumps(result["parsed_json"], ensure_ascii=False, indent=2))
    print(f"\n(latency_ms={result['latency_ms']}, model={MODEL})")
