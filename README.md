# EdgeCardio-LoRA

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Data: DVC](https://img.shields.io/badge/data-DVC-945DD6.svg)](https://dvc.org/)
[![Runtime: ONNX](https://img.shields.io/badge/runtime-ONNX%20INT8-005CED.svg)](https://onnxruntime.ai/)

> **Pipeline End-to-End para diagnóstico automático de anomalías en señales de Electrocardiograma (ECG), optimizado para dispositivos de computación en el borde (Edge AI) mediante Parameter-Efficient Fine-Tuning (LoRA) y Cuantización a INT8, con arquitectura preparada para Aprendizaje Federado.**

---

## 🫀 Visión General

La monitorización ambulatoria continua de ECG (ej. parches Holter y wearables médicos) genera flujos de señales biomédicas donde la latencia de respuesta y la privacidad del paciente son críticas. El envío continuado de señales a la nube compromete la confidencialidad médica (regulada por GDPR y HIPAA), además de requerir conectividad constante y un consumo energético inviable en hardware portátil.

**EdgeCardio-LoRA** aborda este desafío ejecutando la inferencia en tiempo real directamente en microprocesadores de bajo consumo (ARM Cortex-A / x86-64), combinando:
- **Inferencia en el Extremo (< 10 ms):** Clasificación de latidos según el estándar clínico **AAMI EC57** ($N, S, V, F, Q$).
- **Personalización Eficiente (LoRA):** Calibración a la morfología cardíaca específica de cada paciente entrenando $< 1\%$ de parámetros adicionales con HuggingFace `peft`, mitigando el olvido catastrófico (*catastrophic forgetting*).
- **Compresión Extrema (ONNX INT8):** Modelo cuantizado con un peso en disco inferior a $2\text{ MB}$ y consumo en RAM menor a $35\text{ MB}$, prescindiendo de dependencias pesadas de entrenamiento en tiempo de ejecución.
- **Diseño Privacy-by-Design:** La señal cruda no sale del dispositivo. En ciclos federados, el cliente transmite exclusivamente las matrices de bajo rango ($\Delta W$).

> Para un desglose formal de los diagramas C4, árboles de decisión y flujos de secuencia, consulta la [Documentación de Arquitectura](docs/architecture/ARCHITECTURE.md).

---

## 🏗️ Arquitectura del Pipeline

```
  [Sensor ECG Lead II (360 Hz)]
               │
               ▼
┌──────────────────────────────────────────────┐
│        Nodo Edge Local (EdgeCardio)          │
│                                              │
│   FastAPI Gateway  ───>  Signal Core         │
│    (/v1/infer/beat)      (Filter & Z-score)  │
│                                │             │
│                                ▼             │
│                       ONNX Runtime (INT8)    │
│                       [1D-CNN + Merged LoRA] │
│                                │             │
│                                ▼             │
│                      Diagnóstico (< 10 ms)   │
│                      Clases AAMI: N, S, V, F │
└──────────────────────────────────────────────┘
               │
               └─ [Canal Seguro / Futuro FL] ──> Solo matrices ΔW (LoRA)
```

### Características Técnicas

- **Partición Inter-Paciente Estricta:** Implementación del protocolo de **de Chazal et al. (2004)** sobre el dataset MIT-BIH (22 registros DS1 para entrenamiento, 22 registros DS2 para test independiente), evitando el *data leakage* habitual en benchmarks de series temporales médicas.
- **Backbone 1D-CNN:** Extractor de características convolucionales unidimensionales optimizado para relaciones temporales locales de alta frecuencia (complejo QRS, alteraciones del segmento ST).
- **Servicio Asíncrono de Inferencia:** Microservicio FastAPI con validación estricta de esquemas numéricos en Pydantic v2 y motor desacoplado ONNX Runtime C++.
- **Gobernanza MLOps:** Versionado y linaje de datos mediante **DVC**, seguimiento de hiperparámetros y modelos con **MLflow** local (SQLite), y configuración centralizada en YAML.

---

## 🗺️ Roadmap de Implementación

- [x] **Fase 0: Arquitectura y Diseño:** Especificación técnica C4, estándares clínicos AAMI y decisiones arquitectónicas (ADRs).
- [ ] **Fase A: Data Ingestion & MLOps:** Pipeline determinista con `wfdb`, filtrado Butterworth (0.5 - 45 Hz), segmentación de picos R y linaje con DVC.
- [ ] **Fase B: Baseline & Tracking:** Implementación de 1D-CNN en PyTorch, entrenamiento con partición inter-paciente y registro en MLflow.
- [ ] **Fase C: Personalización Eficiente:** Inyección de adaptadores LoRA (PEFT) para calibración por paciente sobre el modelo base congelado.
- [ ] **Fase D: Edge Compression:** Fusión de adaptadores LoRA, exportación a ONNX y cuantización INT8 evaluando tamaño, latencia y macro F1.
- [ ] **Fase E: Edge Deployment:** Microservicio FastAPI empaquetado en contenedor Docker optimizado para procesadores empotrados.

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.10 o superior (compatible con 3.10, 3.11 y 3.12)
- Git instalado en el sistema

### 1. Clonar el repositorio y configurar el entorno virtual

```bash
git clone https://github.com/Diegodepab/EdgeCardio-LoRA.git
cd EdgeCardio-LoRA

# Con uv (recomendado):
uv venv
source .venv/bin/activate
uv pip install -e ".[ml,dev]"

# O mediante pip tradicional:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[ml,dev]"
```

### 2. Inicializar el control de versiones de datos (DVC)

```bash
dvc init
```

### 3. Ejecución de la Fase A: Ingesta y Preprocesado de Datos

```bash
# Descarga automatizada y verificación de integridad del MIT-BIH Arrhythmia Database
python -m src.data.ingest

# Preprocesado, filtrado paso-banda y segmentación inter-paciente
python -m src.data.preprocess

# O reproducir el pipeline formal mediante DVC
dvc repro
```

---

## 🧪 Calidad de Código y Testing

El repositorio incorpora estándares de tipado estático, formateo y tests unitarios:

```bash
# Análisis estático y formateo con Ruff
ruff check .
ruff format . --check

# Verificación de tipos estáticos con Mypy
mypy src

# Ejecución de la suite de tests con cobertura
pytest tests/ -v --cov=src
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
