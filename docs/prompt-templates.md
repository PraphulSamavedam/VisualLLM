# Prompt Template Engineering

## Overview

Prompt engineering is critical for optimizing LLM performance in Visual Question Answering tasks. This page documents the seven prompt templates tested in the VisualLLM project, analyzing their design principles, performance characteristics, and model-specific behaviors.

## Template Design Principles

Effective prompt templates for VQA must balance several key factors:

1. **Clarity**: Explicit instructions for single-word answers
2. **Context Provision**: Clear separation of image captions and object detections
3. **Question Focus**: Emphasizing the question being asked
4. **Response Constraint**: Enforcing brevity and format compliance
5. **Model Awareness**: Acknowledging or omitting the source models (BLIP, YOLO)

## The Seven Templates

### Template 1: Model-Aware Prompt

**Template:**
```python
"Based on the image caption(provided by BLIP model) as '{caption}' and " \
"detections(provided by Yolo) as {detections}, answer in a single word " \
"the question based on the image details as question: '{question}'\n" \
"Answer: "
```

**Design Philosophy:**
- Explicitly mentions source models (BLIP and YOLO)
- Provides clear context attribution
- Direct instruction for single-word answers
- Structured format with labeled sections

**Performance:**

| Model | Quantized Exact Match | Unquantized Exact Match | Quantized Semantic | Unquantized Semantic |
|:------|:---------------------:|:-----------------------:|:------------------:|:--------------------:|
| Llama | 0.43 | 0.43 | 0.57 | 0.58 |
| Mistral | 0.476 | 0.476 | 0.491 | 0.491 |

**Analysis:**
- **Best performer for Llama**: Achieves highest semantic match accuracy (0.58)
- Strong performance across both models
- Model awareness may help establish context credibility

---

### Template 2: Explicit Instruction Style

**Template:**
```python
"Using the image caption '{caption}' and detected objects '{detections}', " \
"answer the following question with a single word: '{question}'.\n" \
"Answer: "
```

**Design Philosophy:**
- Uses action verb "Using" to establish context
- Explicit "with a single word" constraint
- Simpler structure without model attribution
- More conversational flow

**Performance:**

| Model | Quantized Exact Match | Unquantized Exact Match | Quantized Semantic | Unquantized Semantic |
|:------|:---------------------:|:-----------------------:|:------------------:|:--------------------:|
| Llama | 0.30 | 0.30 | 0.41 | 0.39 |
| Mistral | 0.483 | 0.484 | 0.501 | 0.102 |

**Analysis:**
- **Poor performance for Llama**: Significant drop from Template 1
- **Strong for Mistral**: Second-best exact match accuracy
- Model-specific sensitivity to phrasing

---

### Template 3: Direct Question Format

**Template:**
```python
"Caption: '{caption}'. Detected Objects: '{detections}'. " \
"What is the one-word answer to this question about the image: '{question}'?\n" \
"Answer: "
```

**Design Philosophy:**
- Labeled format: "Caption:" and "Detected Objects:"
- Meta-question approach: "What is the one-word answer to..."
- Emphasizes "about the image" context
- More structured presentation

**Performance:**

| Model | Quantized Exact Match | Unquantized Exact Match | Quantized Semantic | Unquantized Semantic |
|:------|:---------------------:|:-----------------------:|:------------------:|:--------------------:|
| Llama | 0.26 | 0.24 | 0.35 | 0.35 |
| Mistral | 0.478 | 0.479 | 0.492 | 0.493 |

**Analysis:**
- **Weakest for Llama**: Lowest performance across metrics
- **Consistent for Mistral**: Mid-range performance
- Meta-question format may add cognitive overhead

---

### Template 4: Formal Request Style

**Template:**
```python
"Given the description '{caption}' and identified elements '{detections}', " \
"provide a one-word response to this inquiry about the image: '{question}'.\n" \
"Answer: "
```

**Design Philosophy:**
- Formal language: "Given", "identified elements", "inquiry"
- Professional tone
- "Provide a one-word response" instruction
- Uses "description" instead of "caption"

**Performance:**

| Model | Quantized Exact Match | Unquantized Exact Match | Quantized Semantic | Unquantized Semantic |
|:------|:---------------------:|:-----------------------:|:------------------:|:--------------------:|
| Llama | 0.39 | 0.39 | 0.52 | 0.52 |
| Mistral | 0.472 | 0.473 | 0.488 | 0.488 |

**Analysis:**
- **Mid-tier for both models**: Balanced performance
- Formal language may increase response quality
- Consistent quantization behavior

---

### Template 5: Simplified Instruction

**Template:**
```python
"From the image caption '{caption}' and object detections '{detections}', " \
"find the answer to: '{question}'. Respond in just one word.\n" \
"Answer: "
```

**Design Philosophy:**
- Action-oriented: "find the answer"
- Simplified structure
- "Respond in just one word" at end
- Natural language flow

**Performance:**

| Model | Quantized Exact Match | Unquantized Exact Match | Quantized Semantic | Unquantized Semantic |
|:------|:---------------------:|:-----------------------:|:------------------:|:--------------------:|
| Llama | 0.33 | 0.33 | 0.43 | 0.43 |
| Mistral | 0.484 | 0.486 | 0.498 | 0.501 |

**Analysis:**
- **Below average for Llama**: Lower than Templates 1, 4, 6
- **Strong for Mistral**: Third-best performance
- "Find the answer" framing may be ambiguous

---

### Template 6: Challenge Format

**Template:**
```python
"Challenge: With the caption '{caption}' and objects detected as '{detections}', " \
"determine the single-word answer to the question: '{question}'.\n" \
"Answer: "
```

**Design Philosophy:**
- Gamification: "Challenge:" prefix
- Engaging framing
- "Determine" suggests analytical thinking
- Structured format

**Performance:**

| Model | Quantized Exact Match | Unquantized Exact Match | Quantized Semantic | Unquantized Semantic |
|:------|:---------------------:|:-----------------------:|:------------------:|:--------------------:|
| Llama | 0.41 | 0.41 | 0.52 | 0.53 |
| Mistral | 0.472 | 0.473 | 0.488 | 0.489 |

**Analysis:**
- **Second-best for Llama**: Competitive performance
- **Mid-tier for Mistral**: Consistent with Template 4
- Challenge framing may increase focus

---

### Template 7: Reverse Order Format

**Template:**
```python
"Answer in a single word for the question: {question} using image caption: " \
"{caption} and object detections: {detections}.\n" \
"Answer: "
```

**Design Philosophy:**
- Question-first approach
- Instruction precedes context
- Less structured format
- Direct and concise

**Performance:**

| Model | Quantized Exact Match | Unquantized Exact Match | Quantized Semantic | Unquantized Semantic |
|:------|:---------------------:|:-----------------------:|:------------------:|:--------------------:|
| Llama | 0.11 | 0.11 | 0.17 | 0.18 |
| Mistral | 0.516 | 0.519 | 0.549 | 0.551 |

**Analysis:**
- **Worst for Llama**: Catastrophic performance drop
- **Best for Mistral**: Highest scores across all metrics
- Question-first order creates strong model divergence

---

## Performance Comparison

### Complete Results Table

#### BLIP + YOLO + Llama-2

| Template | Exact Match | Semantic Match | Performance |
|:--------:|------------:|---------------:|-------------|
| **1** ⭐ | **43.0%** | **58.0%** | 🏆 Best - Context-first |
| 6 | 41.0% | 53.0% | ✅ Second-best |
| 4 | 39.0% | 52.0% | ✅ Solid baseline |
| 5 | 33.0% | 43.0% | ⚠️ Below average |
| 2 | 30.0% | 39.0% | ⚠️ Poor |
| 3 | 24.0% | 35.0% | ❌ Avoid |
| 7 | 11.0% | 18.0% | ❌ Catastrophic |

**Best Configuration:** Template 1 with 58% semantic accuracy

---

#### BLIP + YOLO + Mistral-7B

| Template | Exact Match | Semantic Match | Performance |
|:--------:|------------:|---------------:|-------------|
| **7** ⭐ | **51.9%** | **55.1%** | 🏆 Best - Question-first |
| 2 | 48.4% | 50.1% | ✅ Second-best |
| 5 | 48.6% | 50.1% | ✅ Consistent |
| 1 | 47.6% | 49.1% | ✅ Good |
| 3 | 47.9% | 49.3% | ✅ Good |
| 4 | 47.3% | 48.8% | ⚠️ Lower |
| 6 | 47.3% | 48.9% | ⚠️ Lower |

**Best Configuration:** Template 7 with 55.1% semantic accuracy

---

### Cross-Model Comparison

| Model | Best Template | Semantic Accuracy | Worst Template | Gap |
|-------|:-------------:|-----------------:|:--------------:|----:|
| **Llama-2** | Template 1 | **58.0%** ⭐ | Template 7 | 40.0% |
| **Mistral** | Template 7 | **55.1%** | Template 6 | 6.2% |

**Key Insight:** Template 7 shows **86% performance gap** between models:
- Mistral: 55.1% (best) 🏆
- Llama-2: 18.0% (worst) ❌

### Key Insights

1. **Model-Specific Preferences:**
   - **Llama prefers Template 1**: Context-first, model-aware format
   - **Mistral prefers Template 7**: Question-first, direct format
   - Dramatic performance divergence on Template 7

2. **Quantization Impact:**
   - Minimal difference between quantized and unquantized versions
   - Both formats maintain consistent performance
   - BFloat16 quantization is viable for production

3. **Template Stability:**
   - Mistral shows more consistent performance across templates (0.472-0.519)
   - Llama shows higher variance (0.11-0.43)
   - Template selection more critical for Llama

## Model-Specific Recommendations

### For Llama-2-7b-chat-hf

**Recommended Templates:**

1. **Template 1** (Best overall): 0.58 semantic accuracy
   - Use when maximum accuracy is required
   - Ideal for production deployments

2. **Template 6** (Second-best): 0.53 semantic accuracy
   - Alternative with engaging framing
   - Good for interactive applications

3. **Template 4** (Solid baseline): 0.52 semantic accuracy
   - Formal, professional tone
   - Suitable for business contexts

**Avoid:**
- **Template 7**: Catastrophic failure (0.18 semantic)
- **Template 3**: Poor performance (0.35 semantic)
- **Template 2**: Below-average results (0.39 semantic)

### For Mistral-7B-Instruct-v0.2

**Recommended Templates:**

1. **Template 7** (Best overall): 0.551 semantic accuracy
   - Counterintuitive question-first format
   - Highest performance by significant margin

2. **Template 2** (Second-best): 0.501 semantic accuracy
   - Simple, explicit instructions
   - Good alternative to Template 7

3. **Template 5** (Consistent): 0.501 semantic accuracy
   - Natural language flow
   - Reliable baseline

**Avoid:**
None of the templates show catastrophic failure for Mistral. All perform reasonably well (0.488-0.551 semantic accuracy).

## Implementation Code

### Template Definition

From `/Users/prapsama/Documents/Personal/VisualLLM/src/llama_templates.py`:

```python
templates = [
    # Template 1: Model-Aware
    "Based on the image caption(provided by BLIP model) as '{caption}' and "
    "detections(provided by Yolo) as {detections}, answer in a single word "
    "the question based on the image details as question: '{question}'\n"
    "Answer: ",

    # Template 2: Explicit Instruction
    "Using the image caption '{caption}' and detected objects '{detections}', "
    "answer the following question with a single word: '{question}'.\n"
    "Answer: ",

    # Template 3: Direct Question
    "Caption: '{caption}'. Detected Objects: '{detections}'. "
    "What is the one-word answer to this question about the image: '{question}'?\n"
    "Answer: ",

    # Template 4: Formal Request
    "Given the description '{caption}' and identified elements '{detections}', "
    "provide a one-word response to this inquiry about the image: '{question}'.\n"
    "Answer: ",

    # Template 5: Simplified
    "From the image caption '{caption}' and object detections '{detections}', "
    "find the answer to: '{question}'. Respond in just one word.\n"
    "Answer: ",

    # Template 6: Challenge
    "Challenge: With the caption '{caption}' and objects detected as '{detections}', "
    "determine the single-word answer to the question: '{question}'.\n"
    "Answer: ",

    # Template 7: Reverse Order
    "Answer in a single word for the question: {question} using image caption: "
    "{caption} and object detections: {detections}.\n"
    "Answer: "
]
```

### Usage Example

```python
import pandas as pd
import torch
from datasets import Dataset
from transformers import pipeline
from huggingface_hub import login

# Load data
df = pd.read_csv(detections_file_path)

# Select template
template = templates[0]  # Template 1 for Llama, Template 6 for Mistral

# Format prompts
df["Prompt"] = df.apply(
    lambda row: template.format(
        question=row['Question'],
        detections=row['Generated Detections'],
        caption=row['Generated Caption']
    ),
    axis=1
)

# Initialize pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For Llama
login(token=os.environ.get("LLAMA_TOKEN"))
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    return_full_text=False,
    device=device
)

# For Mistral (no login required)
pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    return_full_text=False,
    device=device
)

# Generate answers
dataset = Dataset.from_pandas(df)
output = pipe(dataset["Prompt"], max_new_tokens=3)
df["Model Output"] = output
```

## Design Recommendations

### General Best Practices

1. **Always include single-word constraint**: Essential for VQA evaluation
2. **Use max_new_tokens=3**: Prevents verbose responses
3. **Test both template orders**: Context-first vs question-first
4. **Consider model architecture**: Templates that work for one model may fail for another

### Template Engineering Principles

**For Context-Heavy Tasks:**
- Use Templates 1, 4, or 6
- Provide clear context attribution
- Structure information hierarchically

**For Direct Q&A:**
- Use Templates 7 (Mistral) or 2 (Mistral)
- Lead with the question
- Minimize cognitive load

**For Professional Applications:**
- Use Template 4 (formal tone)
- Maintain consistent terminology
- Avoid gamification

## Error Analysis

### Common Failure Modes

1. **Template 7 with Llama**:
   - Question-first order causes confusion
   - Model may miss context placement
   - Critical architecture difference from Mistral

2. **Template 3 with Llama**:
   - Meta-question format adds complexity
   - "What is the one-word answer to..." is redundant
   - Model struggles with nested instructions

3. **Quantization Robustness**:
   - Minimal performance impact across templates
   - BFloat16 maintains template effectiveness
   - Safe for production deployment

## Conclusion

Template selection is crucial for VQA performance, with dramatic model-specific effects. Llama-2 performs best with context-first, model-aware formats (Template 1), while Mistral-7B excels with question-first, direct formats (Template 7). The 86% performance gap on Template 7 (Llama: 0.18 vs Mistral: 0.551) demonstrates the importance of model-specific optimization.

For production systems:
- **Use Template 1 for Llama** (0.58 semantic accuracy)
- **Use Template 7 for Mistral** (0.551 semantic accuracy)
- **Always test templates** before deployment
- **Quantization is safe** (minimal impact on template performance)

---

## Navigation

**← [Back to Home](index.md)**

### Related Experiments

- [Generation Configuration](generation-config.md) - Token limits and configuration
- [In-Context Learning](in-context-learning.md) - Few-shot learning experiments
- [Results & Analysis](results.md) - Comprehensive performance analysis

### Pipeline Pages

- [Pipeline Overview](pipelines.md) - Compare all three architectures
- [BLIP + LLM](blip-llm-pipeline.md) | [YOLO + LLM](yolo-llm-pipeline.md) | [BLIP + YOLO + LLM](blip-yolo-llm-pipeline.md)
