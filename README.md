# 🧠 Agente de Razonamiento con Beam Search y LLMs

Este proyecto implementa un prototipo de agente basado en modelos de lenguaje (LLMs) capaz de **generar sonetos en español** mediante un proceso iterativo de exploración y evaluación.

El sistema utiliza un enfoque de **Beam Search** para generar múltiples versiones de un soneto y seleccionar las más prometedoras en función de criterios de calidad.

El sistema combina:
- Generación de sonetos mediante un LLM
- Evaluación automática (en desarrollo) basada en métricas formales
- Poda de candidatos mediante Beam Search

---

## 🚀 Características principales

- Generación iterativa de sonetos mediante LLMs
- Uso de Beam Search para explorar múltiples versiones del poema
- Sistema de feedback para mejorar iterativamente las soluciones
- Arquitectura basada en grafos con LangGraph
- Ejecución completamente local mediante Ollama

---

## 🔧 Requisitos

- Python 3.10 o superior
- Ollama instalado y en ejecución

---

## 📦 Instalación de dependencias

Instalar las librerías necesarias:

```bash
pip install -r requirements.txt
```

Se recomienda utilizar un entorno virtual para evitar conflictos entre dependencias.

---

## 🤖 Configuración de Ollama

Instalar Ollama desde:

https://ollama.com

Descargar el modelo utilizado:

```bash
ollama pull mistral:7b-instruct
```

Asegurarse de que el servidor está activo:

```bash
ollama serve
```

---

## ▶️ Ejecución

Ejecutar el script principal:

```bash
python src/langgraph_beam_ollama.py
```

---

## 🧩 Descripción del funcionamiento

El sistema sigue un ciclo iterativo compuesto por tres fases:

**Expansión (Expand)**: Se generan múltiples versiones candidatas del soneto.
**Evaluación (Score)**: Cada soneto es evaluado para estimar su calidad.
**Poda (Prune)**: Se seleccionan los `k` mejores sonetos en función de su puntuación.

Este proceso se repite durante un número fijo de pasos (`max_steps`), permitiendo mejorar progresivamente los resultados.

---

## 📊 Evaluación de los sonetos

La evaluación del soneto se abordará en dos fases:

1. Métricas objetivas (en desarrollo)

Se pretende evaluar automáticamente los siguientes aspectos:

Extensión: el soneto debe tener exactamente 14 versos
Métrica: cada verso debe tener aproximadamente 11 sílabas
Rima: esquema ABBA ABBA CDC CDC

Estas métricas se implementarán mediante funciones en Python que permitirán analizar formalmente cada soneto generado.

2. Evaluación subjetiva (trabajo futuro)

Como mejora futura, se incorporará un enfoque de LLM-as-a-judge, donde un modelo de lenguaje evaluará aspectos más subjetivos como:

calidad estética
coherencia poética
riqueza léxica

Esta evaluación podrá combinarse con las métricas objetivas para obtener una puntuación más completa.l

---

## 🧪 Observaciones

- El sistema permite analizar cómo evolucionan los sonetos a lo largo de las iteraciones
- La generación de poesía con restricciones estrictas es un problema complejo para LLMs locales
- El enfoque permite introducir feedback estructurado para mejorar los resultados

---

## ⚙️ Dependencias principales

- langgraph → orquestación del grafo de ejecución
- langchain → integración con modelos de lenguaje
- requests → comunicación con Ollama

---

## 📁 Estructura del proyecto

```text
TFG_AGENTE_RAZONAMIENTO
├── memoria
├── runs
├── src
│   ├── langgraph_beam_ollama.py
│   └── sonnet_metrics.py
└── .gitignore
```

---

## 📌 Notas

- El sistema está diseñado como prueba de concepto para agentes con razonamiento iterativo
- Se centra en la generación controlada de texto bajo restricciones formales
- Las métricas de evaluación se están desarrollando de forma incremental

---

## 📚 Contexto académico

Este proyecto forma parte de un Trabajo de Fin de Grado en Ingeniería Informática, centrado en:

- Agentes con razonamiento deliberativo
- Uso de Beam Search en LLMs
- Evaluación automática de salidas generadas
- Generación de texto bajo restricciones formales

---

## 👨‍💻 Autor

Proyecto desarrollado por Juan Torres Gómez, estudiante de Ingeniería Informática de la Universidad de Málaga

