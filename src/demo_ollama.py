# src/demo_ollama.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import requests


BASE_URL = "http://localhost:11434"
MODEL = "llama3"

# Hacemos que la salida, llamada 'demo_ollama.jsonl' vaya a la carpeta runs/
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RUNS_DIR / "demo_ollama.jsonl"


def check_ollama_is_up() -> None:
    """
    Comprobación rápida de que el servidor de Ollama está accesible.
    Si esto falla, no tiene sentido intentar generar.
    """
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            "No puedo conectar con Ollama en http://localhost:11434.\n"
            "Causas típicas: Ollama no está ejecutándose, firewall, o URL incorrecta.\n"
            f"Error: {e}"
        )


def generate(prompt: str, *, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Llamada mínima al endpoint /api/generate.
    - Enviamos texto (prompt)
    - Recibimos texto (response)
    - options permite controlar parámetros del muestreo (temperature, num_predict, etc.)
    """
    payload: Dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,  # respuesta completa en un JSON
    }
    if options:
        payload["options"] = options

    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    t1 = time.perf_counter()

    data["_latency_ms"] = int((t1 - t0) * 1000)
    data["_sent_options"] = options or {}
    return data


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # 1) Comprobar que Ollama está levantado
    check_ollama_is_up()

    # 2) Prompt de prueba (simple y corto)
    prompt = "Resume en 3 bullets qué es beam search."

    # 3) Parámetros mínimos (de momento solo temperatura, para ver efecto más adelante)
    #    Nota: en el siguiente paso haremos un “laboratorio” variando esto (archivo 'params_lab').
    options = {"temperature": 0.2}

    # 4) Generamos la respuesta, pasandole como parámetros el objeto 'options' que hemos creado arriba
    result = generate(prompt, options=options)

    # 5) Mostrar por pantalla (feedback inmediato)
    print(result.get("response", "").strip())
    print(f"\n(latency_ms={result['_latency_ms']}, model={MODEL}, options={options})")

    # 6) Guardar log (para documentar experimentos)
    append_jsonl(
        LOG_FILE,
        {
            "model": MODEL,
            "prompt": prompt,
            "options": options,
            "latency_ms": result["_latency_ms"],
            "response": result.get("response", "").strip(),
            # guardamos también el raw por si luego quieres ver detalles adicionales
            "raw": {k: v for k, v in result.items() if not k.startswith("_")},
        },
    )

    # Confirmación de que hemos guardado la salida en el LOG_FILE indicado al inicio del código
    print(f"\nLog guardado en: {LOG_FILE}")
