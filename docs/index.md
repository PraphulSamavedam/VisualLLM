# VisualLLM - Visual Question Answering

This repository explores **Visual Question Answering (VQA)** using Large Language Models without explicit fine-tuning. The project investigates how different pipeline configurations combining vision models (BLIP, YOLO) with LLMs (Llama-2, Mistral) perform on VQA tasks through prompt engineering and in-context learning.

**Institution:** Northeastern University
**Focus Areas:** Visual Question Answering, Multi-modal AI, Prompt Engineering, In-Context Learning

## What is Visual Question Answering?

Visual Question Answering (VQA) is the challenging task of answering natural language questions about images. The model must understand both the visual content and the question to generate accurate answers.

**Examples:**

| Question | Image | Answer |
|----------|-------|--------|
| How many children are in bed? | ![Children in bed] | 2 |
| Who is wearing glasses? | ![People with glasses] | Man |
| Is the umbrella upside down? | ![Upside down umbrella] | Yes |

This is particularly significant for making LLMs truly multi-modal and enabling them to process and reason about visual information.

---

## Project Overview

This research project has **two main aspects**:

### 1. Pipeline Performance Exploration
Investigate different combinations of vision models and LLMs without fine-tuning:
- **Vision Models:** BLIP (image captioning), YOLO (object detection)
- **Language Models:** Llama-2-7B, Mistral-7B
- **Configurations:** BLIP+LLM, YOLO+LLM, BLIP+YOLO+LLM

### 2. Search System Enhancement
Use the best-performing pipeline to improve image search and recommendation systems.

---

## Key Findings

### Best Configuration
**BLIP + YOLO + Mistral** achieves the highest performance:
- **Exact Match Accuracy:** 51.9%
- **Semantic Match Accuracy:** 55.1%
- **Optimal Settings:** Max 3 tokens, single-word prompt instruction

### Model Comparison
- **Llama-2** performs better in most pipeline configurations
- **Mistral** excels specifically in the BLIP+YOLO+LLM pipeline
- Quantization has minimal impact on accuracy

### In-Context Learning
- **1-shot:** Best performance (48% exact, 59% semantic)
- **3-shot:** Slight degradation (43-44% exact, 51-54% semantic)
- **5-shot:** Significant drop (14-24% exact, 18-29% semantic)
- **Insight:** More examples ≠ better (context relevance matters)

### Baseline Comparison
**BLIP-VQA** (fine-tuned model): 90.5% exact, 96.1% semantic

**Gap:** Generic pipelines achieve ~50% of fine-tuned performance without training

---

## Technologies

- **Python 3.8+** - Primary programming language
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - LLM inference
- **BLIP** - Image captioning model
- **YOLOv5** - Object detection
- **Llama-2-7B-chat** - Language model
- **Mistral-7B-Instruct** - Language model
- **VQA v2.0 Dataset** - Evaluation benchmark

---

## Quick Navigation

### [Pipeline Exploration](pipelines.md)
Comparison of different vision model + LLM combinations.

**Key Topics:** BLIP+LLM, YOLO+LLM, BLIP+YOLO+LLM, performance analysis

---

### [Prompt Templates](prompt-templates.md)
Investigation of 7 different prompt templates and their impact on accuracy.

**Key Topics:** Template design, instruction clarity, output formatting

---

### [In-Context Learning](in-context-learning.md)
Analysis of few-shot learning with 1, 3, and 5 example demonstrations.

**Key Topics:** ICL effectiveness, example selection, context limitations

---

### [Results & Analysis](results.md)
Comprehensive experimental results with accuracy metrics and insights.

**Key Topics:** Performance tables, error analysis, model comparison

---

### [Getting Started](getting-started.md)
Setup instructions, usage examples, and code walkthrough.

**Key Topics:** Installation, running experiments, reproducing results

---

## Repository Structure

```
VisualLLM/
├── src/                          # Source code
│   ├── blip_captions.py         # BLIP image captioning
│   ├── yolo_detections.py       # YOLO object detection
│   ├── blip_llama.py            # BLIP + Llama pipeline
│   ├── blip_mistral.py          # BLIP + Mistral pipeline
│   ├── yolo_llama.py            # YOLO + Llama pipeline
│   ├── yolo_mistral.py          # YOLO + Mistral pipeline
│   ├── blip_yolo_llama.py       # BLIP + YOLO + Llama pipeline
│   ├── blip_yolo_mistral.py     # BLIP + YOLO + Mistral pipeline
│   ├── llama_templates.py       # Prompt templates for Llama
│   ├── mistral_templates.py     # Prompt templates for Mistral
│   ├── icl.py                   # In-context learning experiments
│   └── constants.py             # Configuration constants
│
├── data/                         # Dataset and preprocessed data
│   ├── validation/              # VQA v2.0 validation set
│   ├── train/                   # VQA v2.0 training set
│   └── *.csv                    # Preprocessed captions/detections
│
├── notebooks/                    # Jupyter notebooks
│   └── Experiment.ipynb         # Exploratory analysis
│
├── results/                      # Experimental outputs
├── report/                       # Project report and figures
└── docs/                         # MkDocs documentation
```

---

## Quick Start

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r src/requirements.txt
```

### Run Experiments
```bash
cd src

# Generate BLIP captions
python blip_captions.py

# Generate YOLO detections
python yolo_detections.py

# Run BLIP + YOLO + Mistral pipeline
python blip_yolo_mistral.py

# Run in-context learning experiments
python icl.py
```

For detailed setup and usage, see [Getting Started](getting-started.md).

---

## Research Context

This project was developed as part of coursework exploring modern approaches to multi-modal AI. The work demonstrates:

✅ **Zero-shot VQA** - Answer visual questions without task-specific training
✅ **Pipeline modularity** - Mix and match vision + language components
✅ **Prompt engineering** - Optimize LLM performance through careful prompting
✅ **In-context learning** - Leverage few-shot examples for improved accuracy
✅ **Comparative analysis** - Systematic evaluation of different approaches

---

## Contact

**Author:** Praphul Samavedam
**GitHub:** [@PraphulSamavedam](https://github.com/PraphulSamavedam)

---

## References

- **VQA Paper:** [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468)
- **BLIP:** [Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- **YOLO:** [You Only Look Once](https://arxiv.org/abs/1506.02640)
- **Llama-2:** [Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- **Mistral:** [Mistral 7B](https://arxiv.org/abs/2310.06825)
- **LLaVA:** [Visual Instruction Tuning](https://github.com/haotian-liu/LLaVA)
