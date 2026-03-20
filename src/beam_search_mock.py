import random


# 1. Generador falso (simula LLM)
def mock_generate(step_text, num_candidates=3):
    return [
        f"{step_text} -> opción {i}"
        for i in range(num_candidates)
    ]


# 2. Evaluador falso (simula scoring)
def mock_score(text):
    return random.random()  # número entre 0 y 1


def beam_search(initial_prompt, k=2, max_steps=3):
    # Cada beam: (score acumulado, path)
    beams = [(0.0, [initial_prompt])]

    for step in range(max_steps):
        print(f"\n--- STEP {step} ---")

        all_candidates = []

        # 1. Expandir cada beam
        for score, path in beams:
            current_text = path[-1]

            candidates = mock_generate(current_text)

            for c in candidates:
                new_path = path + [c]
                new_score = score + mock_score(c) # score_total = score_anterior + score_nuevo
                # Las puntuaciones se acumulan, por eso vemos que pasan de 1 (cada puntuación individual es entre 0 y 1, pero después se van acumulando)
                # También podríamos normalizar por la media: score_total = (score_total * n + new_score) / (n + 1)

                # Nuestro score ahora significa "Qué buena es la trayectoria completa", no cada paso individual

                all_candidates.append((new_score, new_path))

        # 2. Ordenar por score
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # 3. Podar (quedarse con top-k)
        beams = all_candidates[:k]

        # Debug
        for i, (score, path) in enumerate(beams):
            print(f"\nBeam {i}:")
            print(f"Score: {score:.3f}")
            print("Path:")
            for p in path:
                print("  ", p)

    return beams


if __name__ == "__main__":
    final_beams = beam_search("Inicio", k=2, max_steps=3)

    print("\n=== RESULTADO FINAL ===")
    for score, path in final_beams:
        print(f"\nScore final: {score:.3f}")
        for step in path:
            print(" ", step)