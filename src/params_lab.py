# src/params_lab.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


BASE_URL = "http://localhost:11434"
MODEL = "llama3"

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RUNS_DIR / "params_lab.jsonl"


def check_ollama_is_up() -> None:
    """
    Verifica que Ollama responde. Si esto falla, el resto del experimento no tiene sentido.
    """
    r = requests.get(f"{BASE_URL}/api/tags", timeout=10)
    r.raise_for_status()


def generate(prompt: str, *, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta /api/generate con stream=False para obtener una respuesta final en JSON.
    """
    payload: Dict[str, Any] = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    t1 = time.perf_counter()

    data["_latency_ms"] = int((t1 - t0) * 1000)
    return data


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def short_preview(text: str, n: int = 120) -> str:
    t = " ".join(text.strip().split())
    return (t[:n] + "…") if len(t) > n else t


def make_experiments() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Devuelve una lista de (nombre_experimento, options).
    Importante: mantenemos el prompt fijo, y variamos options de forma controlada.
    """
    # Base “estable”: baja temperatura y longitud acotada
    base = {
        "temperature": 0.2,
        "num_predict": 120,
    }

    exps: List[Tuple[str, Dict[str, Any]]] = []

    # 1) Temperatura: misma longitud, distinta creatividad
    for temp in [0.0, 0.2, 0.7, 1.0]:
        opts = dict(base)
        opts["temperature"] = temp
        exps.append((f"temp_{temp}", opts))

    # 2) Longitud (num_predict): misma creatividad, distinta extensión máxima
    for n in [60, 120, 240]:
        opts = dict(base)
        opts["num_predict"] = n
        exps.append((f"num_predict_{n}", opts))

    # 3) Muestreo (top_p, top_k): explorar cómo cambia la diversidad/control
    #    (dejamos temp fija para aislar el efecto)
    sampling_variants = [
        ("top_p_0.9", {"top_p": 0.9}),
        ("top_p_0.95", {"top_p": 0.95}),
        ("top_k_20", {"top_k": 20}),
        ("top_k_50", {"top_k": 50}),
        ("top_k_50_top_p_0.9", {"top_k": 50, "top_p": 0.9}),
    ]
    for name, extra in sampling_variants:
        opts = dict(base)
        opts.update(extra)
        exps.append((name, opts))

    # 4) Reproducibilidad (seed): repetimos la MISMA config varias veces
    #    para ver si mantiene consistencia.
    seed_base = dict(base)
    seed_base["temperature"] = 0.7  # algo de aleatoriedad para que se note el seed
    for seed in [42, 42, 1234]:
        opts = dict(seed_base)
        opts["seed"] = seed
        exps.append((f"seed_{seed}", opts))

    return exps


if __name__ == "__main__":
    check_ollama_is_up()

    # Prompt fijo (control de variable)
    prompt = (
        "Define beam search en 2-3 frases y pon un ejemplo muy simple en NLP. "
        "No uses más de 8 líneas."
    )

    experiments = make_experiments()

    print(f"Ejecutando {len(experiments)} experimentos con model={MODEL}\n")
    print(f"Log -> {LOG_FILE}\n")

    for i, (name, options) in enumerate(experiments, start=1):
        result = generate(prompt, options=options)
        response_text = result.get("response", "").strip()

        # Mostrar feedback rápido en consola (no saturar)
        print(f"[{i:02d}/{len(experiments)}] {name:20s} latency_ms={result['_latency_ms']}")
        print(f"  options={options}")
        print(f"  preview={short_preview(response_text)}\n")

        # Guardar en JSONL para análisis posterior
        append_jsonl(
            LOG_FILE,
            {
                "experiment": name,
                "model": MODEL,
                "prompt": prompt,
                "options": options,
                "latency_ms": result["_latency_ms"],
                "response": response_text,
                # raw completo por trazabilidad (por si luego quieres ver tokens/estadísticas)
                "raw": {k: v for k, v in result.items() if not k.startswith("_")},
            },
        )

    print("OK: laboratorio terminado.")
