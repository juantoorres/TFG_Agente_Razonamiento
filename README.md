# 🧠 Agente de Razonamiento con Beam Search y LLMs

Este proyecto implementa un prototipo de agente con capacidad de razonamiento basado en modelos de lenguaje (LLMs), utilizando un enfoque de **Beam Search** para explorar múltiples trayectorias de razonamiento y seleccionar las más prometedoras.

El sistema combina:
- Generación de pasos de razonamiento mediante un LLM
- Evaluación crítica de cada trayectoria mediante un segundo proceso (también basado en LLM)
- Poda de hipótesis basada en una función de puntuación agregada

---

## 🚀 Características principales

- Generación de múltiples candidatos en paralelo (Beam Search)
- Evaluación automática de calidad de razonamientos intermedios
- Uso de puntuaciones en el rango [0,1] interpretadas como calidad heurística
- Agregación de puntuaciones mediante media geométrica normalizada
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
ollama pull llama3
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

1. **Expansión (Expand)**  
   Se generan múltiples posibles continuaciones del razonamiento actual.

2. **Evaluación (Score)**  
   Cada trayectoria candidata es evaluada mediante un modelo LLM que asigna:
   - una puntuación de calidad
   - una justificación textual

3. **Poda (Prune)**  
   Se seleccionan las mejores `k` trayectorias según una función de puntuación agregada.

Este proceso se repite durante un número fijo de pasos (`max_steps`).

---

## 📊 Puntuación de trayectorias

Las puntuaciones devueltas por el modelo están en el rango [0,1], pero:

> No representan probabilidades, sino una estimación heurística de calidad.

Para evitar sesgos hacia trayectorias más largas, se emplea una media geométrica normalizada:

```text
score = (∏ q_t)^(1 / T^α)
```

donde:
- q_t es la calidad de cada paso
- T es la longitud de la trayectoria
- α es un hiperparámetro de control

---

## 🧪 Observaciones

- El sistema permite analizar cómo evolucionan distintas trayectorias de razonamiento
- Se pueden detectar comportamientos como:
  - redundancia
  - falta de progreso
  - errores conceptuales
- La calidad del resultado depende en gran medida del modelo evaluador

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
└── .gitignore
```

---

## 📌 Notas

- El sistema está diseñado como prueba de concepto para el desarrollo de agentes con razonamiento deliberativo
- El uso de un mismo modelo para generación y evaluación puede introducir sesgos
- En futuras mejoras se puede emplear un modelo distinto para la fase de evaluación

---

## 📚 Contexto académico

Este proyecto forma parte de un Trabajo de Fin de Grado en Ingeniería Informática, centrado en:

- Razonamiento en modelos de lenguaje
- Métodos de exploración de hipótesis (Beam Search)
- Evaluación automática de trayectorias de pensamiento

---

## 👨‍💻 Autor

Proyecto desarrollado por Juan Torres Gómez, estudiante de Ingeniería Informática de la Universidad de Málaga

