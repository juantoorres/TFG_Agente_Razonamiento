# 🧠 Agente de razonamiento para generación de estrofas ABBA

Este proyecto desarrolla un agente capaz de generar una **estrofa poética de cuatro versos endecasílabos con rima consonante ABBA** usando un modelo de lenguaje local y un proceso de búsqueda iterativa.

La idea central es combinar la creatividad de un LLM con un mecanismo de control formal: el sistema no acepta simplemente la primera respuesta del modelo, sino que genera varias alternativas, las evalúa y conserva las más prometedoras mediante **Beam Search**.

## 🎯 Objetivo del proyecto

El objetivo es estudiar cómo un agente puede guiar a un modelo de lenguaje en una tarea de generación textual con restricciones estrictas:

- cuatro versos;
- once sílabas métricas por verso;
- rima consonante ABBA;
- coherencia con el tema indicado en el prompt.

El proyecto se sitúa en la intersección entre generación automática de texto, evaluación formal de poesía y algoritmos de búsqueda.

## 🔎 Idea general

El sistema trabaja de forma iterativa:

1. Genera varias estrofas candidatas con un modelo local.
2. Evalúa cada estrofa según criterios formales.
3. Selecciona las mejores mediante Beam Search.
4. Usa el feedback formal para intentar mejorar las siguientes versiones.

Además del Beam Search, el agente incorpora mecanismos de refinamiento local para ayudar al modelo cuando falla alguna restricción métrica o de rima. Estos refinamientos actúan sobre versos concretos, manteniendo el problema lo más acotado posible.

## ✨ Características principales

- 🤖 Generación local mediante Ollama.
- 🌿 Búsqueda iterativa con Beam Search.
- 📏 Evaluación automática de métrica y rima.
- 🧩 Refinamiento local de restricciones formales.
- 🛡️ Conservación de buenos candidatos mediante elitismo.
- 📊 Trazas y métricas para analizar cada ejecución.

## 🗂️ Archivos principales

El núcleo actual del proyecto está en:

```text
src/
+-- langgraph_beam_stanza.py
+-- sonnet_metrics.py
```

`langgraph_beam_stanza.py` contiene el agente principal: generación, Beam Search, evaluación de candidatos, refinamientos y guardado de resultados.

`sonnet_metrics.py` contiene las funciones de evaluación formal: cómputo silábico aproximado, análisis de rima y puntuación de la estrofa.

## 📁 Estructura del repositorio

```text
TFG_Agente_Razonamiento/
+-- src/
|   +-- langgraph_beam_stanza.py
|   +-- sonnet_metrics.py
+-- outputs/
+-- memoria/
+-- pasosImplementacionRimaDetallados/
+-- runs/
+-- requirements.txt
+-- INSTRUCCIONES_EJECUCION.md
+-- CONTEXTO.md
+-- README.md
```

## 📤 Resultados generados

Cada ejecución guarda sus resultados en la carpeta `outputs/`.

Los archivos más importantes son:

- `final_stanza_*.txt`: resumen legible de la estrofa final y sus métricas.
- `final_stanza_metrics_*.json`: información estructurada de la evaluación final.
- `final_stanza_trace_*.json`: traza completa del proceso de Beam Search.

Estos archivos permiten estudiar no solo la estrofa final, sino también cómo ha evolucionado el proceso de búsqueda.

## 🚀 Ejecución del proyecto

Las instrucciones completas de instalación, configuración de Ollama, selección de modelo, parámetros principales y ejecución están en:

```text
INSTRUCCIONES_EJECUCION.md
```

Ese documento es la guía práctica para reproducir las pruebas.

## 📚 Contexto académico

Este repositorio forma parte de un Trabajo de Fin de Grado en Ingeniería Informática.

El foco principal del trabajo es analizar cómo un algoritmo de búsqueda como Beam Search puede servir para estructurar el comportamiento de un agente basado en LLMs, especialmente cuando la salida debe cumplir restricciones formales difíciles de satisfacer de una sola vez.

## 👤 Autor

Proyecto desarrollado por **Juan Torres Gómez**, estudiante de Ingeniería Informática de la Universidad de Málaga.
