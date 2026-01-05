# Pipeline Overview

## Introduction

This project explores three distinct pipeline architectures for Visual Question Answering (VQA) without fine-tuning. Each pipeline combines vision models (BLIP for captioning and/or YOLO for object detection) with Large Language Models (Llama-2 or Mistral) in different configurations.

The goal is to understand which combination of vision understanding provides the best context for LLMs to answer visual questions accurately.

---

## Pipeline Architectures

### Three Main Approaches

```
┌─────────────────────────────────────────────────────────────────┐
│              VISUALLLM PIPELINE CONFIGURATIONS                  │
└─────────────────────────────────────────────────────────────────┘

Pipeline A: BLIP + LLM
┌───────┐     ┌──────┐     ┌─────┐     ┌────────┐
│ Image │ ──► │ BLIP │ ──► │ LLM │ ──► │ Answer │
└───────┘     └──────┘     └─────┘     └────────┘
              Caption      Natural
              (Global)     Language
                          Reasoning

Pipeline B: YOLO + LLM
┌───────┐     ┌──────┐     ┌─────┐     ┌────────┐
│ Image │ ──► │ YOLO │ ──► │ LLM │ ──► │ Answer │
└───────┘     └──────┘     └─────┘     └────────┘
              Objects      Object
              (Local)      Reasoning

Pipeline C: BLIP + YOLO + LLM (Hybrid)
┌───────┐     ┌──────┐  ┐
│ Image │ ──► │ BLIP │  ├─► Combined ──► ┌─────┐ ──► ┌────────┐
└───────┘  ┌─►│ YOLO │  ┘   Context      │ LLM │     │ Answer │
           │  └──────┘                    └─────┘     └────────┘
           └───────────┘
           Parallel      Caption +         Multi-modal
           Processing    Detections        Reasoning
```

---

## Pipeline Comparison

### Performance Summary

| Pipeline | Llama-2 Exact | Llama-2 Semantic | Mistral Exact | Mistral Semantic |
|----------|---------------|------------------|---------------|------------------|
| **BLIP + LLM** | 45.4% | 59.3% | 42.9% | 52.1% |
| **YOLO + LLM** | 47.1% | 63.0% | 29.2% | 39.6% |
| **BLIP + YOLO + LLM** | 46.1% | 64.2% | **51.9%** ⭐ | 59.3% |

**Best Configuration:** BLIP + YOLO + Mistral (Template 7)

---

### Strengths & Weaknesses

#### Pipeline A: BLIP + LLM

**✅ Strengths:**
- Captures overall scene understanding
- Good for descriptive questions ("What is happening?")
- Natural language context for LLM
- Understands mood, atmosphere, context
- Computationally lighter (single vision model)

**⚠️ Limitations:**
- Misses objects not mentioned in caption
- Poor for counting tasks ("How many X?")
- Limited spatial information
- May miss small or background objects
- Caption quality depends on BLIP's focus

**Best For:**
- Scene-level questions
- Qualitative assessments
- Questions about actions or relationships
- General image understanding

---

#### Pipeline B: YOLO + LLM

**✅ Strengths:**
- Excellent for object presence ("Is there a X?")
- Accurate counting ("How many X?")
- Provides spatial locations (bounding boxes)
- Detects all objects in frame
- Precise object identification

**⚠️ Limitations:**
- No overall scene context
- Limited to 80 COCO classes
- Poor for abstract questions (mood, style, color nuances)
- Doesn't capture relationships between objects
- Miss actions or events happening

**Best For:**
- Object-centric questions
- Counting queries
- Spatial reasoning
- Presence/absence questions

---

#### Pipeline C: BLIP + YOLO + LLM (Hybrid)

**✅ Strengths:**
- Comprehensive image understanding (global + local)
- Handles diverse question types
- Best accuracy across most configurations
- Combines caption context with object details
- Provides both "what's happening" and "what objects exist"

**⚠️ Limitations:**
- Higher computational cost (2 vision models)
- Longer prompts (more context tokens)
- Potential information redundancy
- Requires careful prompt engineering to balance both sources

**Best For:**
- Complex questions requiring multiple reasoning types
- When computational cost is acceptable
- Production systems requiring robustness
- Diverse question distributions

---

## Design Rationale

### Why Multiple Pipelines?

Each pipeline explores different hypothesis about what visual information is most useful for VQA:

1. **BLIP (Caption-based):**
   - *Hypothesis:* Natural language descriptions provide intuitive context
   - *Result:* Good for scene understanding, misses object details

2. **YOLO (Detection-based):**
   - *Hypothesis:* Object lists with locations suffice for most questions
   - *Result:* Excellent for object queries, poor for context

3. **BLIP + YOLO (Hybrid):**
   - *Hypothesis:* Combining both modalities covers more question types
   - *Result:* Best overall, slight overhead acceptable

### Key Insights

1. **Complementary Information:**
   - BLIP provides "story" (what's happening)
   - YOLO provides "inventory" (what's present)
   - Together: comprehensive understanding

2. **Model Choice Matters:**
   - Llama-2 better for BLIP/YOLO individual pipelines
   - Mistral better for hybrid BLIP+YOLO pipeline
   - Different models process context differently

3. **Prompt Engineering Critical:**
   - Template structure significantly impacts accuracy (11% vs 51%)
   - Single-word instruction essential
   - Model attribution helps (mentioning "BLIP model", "YOLO")

---

## Pipeline Selection Guide

### Decision Tree

```
Start: What type of questions am I answering?

├─ Primarily object-centric (counting, presence)?
│  └─► Use YOLO + LLM Pipeline
│     Best for: "How many?", "Is there a?", "What objects?"
│
├─ Primarily scene-understanding (actions, context)?
│  └─► Use BLIP + LLM Pipeline
│     Best for: "What is happening?", "What is the mood?", "Describe the scene?"
│
├─ Mixed question types or production system?
│  └─► Use BLIP + YOLO + LLM Pipeline (Hybrid)
│     Best for: Diverse questions, robust performance
│
└─ Computational constraints?
   ├─ Limited resources → BLIP + LLM or YOLO + LLM (single vision model)
   └─ Resources available → BLIP + YOLO + LLM (best accuracy)
```

---

## Implementation Details

### Common Components

All pipelines share:

1. **Preprocessing:**
   - Load VQA dataset
   - Sample question-answer pairs
   - Extract image IDs

2. **Vision Processing (Cached):**
   - Run vision models once
   - Store results in CSV
   - Reuse for multiple LLM experiments

3. **LLM Inference:**
   - Load Llama-2 or Mistral
   - Apply prompt template
   - Generate answer (max 3 tokens)

4. **Evaluation:**
   - Exact match accuracy
   - Semantic match accuracy
   - Compare against ground truth

### Differences

| Aspect | BLIP + LLM | YOLO + LLM | BLIP + YOLO + LLM |
|--------|------------|------------|-------------------|
| **Vision Input** | Caption only | Detections only | Both |
| **Context Type** | Descriptive text | Object list | Rich multi-modal |
| **Prompt Length** | Short (~50 tokens) | Medium (~100 tokens) | Long (~150 tokens) |
| **Best LLM** | Llama-2 | Llama-2 | Mistral |
| **Computation** | Low | Low | Medium |
| **Accuracy** | 45-59% | 47-63% | 46-64% (Llama), **52-59%** (Mistral) |

---

## Next Steps

Explore detailed architecture and implementation for each pipeline:

- **[BLIP + LLM Pipeline](blip-llm-pipeline.md)** - Caption-based approach
- **[YOLO + LLM Pipeline](yolo-llm-pipeline.md)** - Object detection approach
- **[BLIP + YOLO + LLM Pipeline](blip-yolo-llm-pipeline.md)** - Hybrid approach

Each page includes:
- Detailed architecture diagrams
- Data flow visualization
- Code implementation
- Performance characteristics
- Use case recommendations
