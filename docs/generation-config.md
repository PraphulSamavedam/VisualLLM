# Generation Configuration

## Overview

Generation configuration significantly impacts LLM output quality for Visual Question Answering. This page explores how different settings for **token limits** and **prompt instructions** affect the accuracy and usability of generated answers.

The experiments demonstrate that **constraining generation** (max 3 tokens + single-word instruction) improves accuracy from 12.9% to 52.1% - a **4x improvement** through configuration alone.

---

## Configuration Parameters

### Key Settings

```python
generation_config = {
    'max_new_tokens': 3,        # Limit output length
    'do_sample': False,         # Greedy decoding (deterministic)
    'temperature': 1.0,         # Not used with greedy
    'top_p': 1.0,              # Not used with greedy
    'pad_token_id': tokenizer.eos_token_id
}
```

### Tested Configurations

1. **Default Configuration**
   - max_new_tokens = 20 (default)
   - No special prompt instructions
   - Allows verbose LLM outputs

2. **Token-Limited Configuration**
   - max_new_tokens = 3
   - No prompt modification
   - Forces shorter outputs

3. **Token + Prompt Restricted**
   - max_new_tokens = 3
   - Prompt explicitly states: "Answer in a single word"
   - Both generation and instruction constraints

---

## Experimental Results

### YOLO + Mistral Pipeline

| Configuration | Exact Match Accuracy | Semantic Match Accuracy | Improvement |
|---------------|---------------------|------------------------|-------------|
| **Default (20 tokens)** | 12.88% | 15.22% | Baseline |
| **Max 3 tokens** | 21.59% | 25.77% | +67.7% |
| **Max 3 tokens + single word prompt** | **42.93%** | **52.09%** | **+233%** |

**Key Finding:** Adding single-word instruction to token limit **doubles accuracy** again!

---

## Why Does This Matter?

### Problem with Default Configuration

**Issue:** LLMs are trained to be helpful and verbose

**Example Output:**
```
Question: "How many children are in the bed?"

Default LLM Response:
"Based on the image caption provided by BLIP model which states
'two children sleeping peacefully in bed', the answer to your
question is: two children. Hope this helps! Let me know if you
need anything else."

First Token Extracted: "Based"  ❌ WRONG
Ground Truth: "2"
```

**Problem:** When extracting first token, we get wrong answer!

---

### Solution 1: Limit Tokens

**Configuration:** `max_new_tokens=3`

**Effect:** Forces LLM to be concise

**Example Output:**
```
Question: "How many children are in the bed?"

Limited Token Response:
"Two children"

First Token: "Two"  ⚠️ ACCEPTABLE (semantic match)
Ground Truth: "2"
```

**Improvement:** 12.88% → 21.59% exact match

**Limitation:** Still not forcing single-word format

---

### Solution 2: Prompt Instruction

**Configuration:** `max_new_tokens=3` + "Answer in a single word"

**Effect:** LLM understands task requirements

**Example Output:**
```
Question: "How many children are in the bed?"

Instructed Response:
"2"

First Token: "2"  ✅ CORRECT
Ground Truth: "2"
```

**Improvement:** 21.59% → 42.93% exact match

**Success:** Nearly 2x improvement by adding instruction!

---

## Implementation

### Configuration 1: Default

```python
# No restrictions
prompt = f"Based on the image caption: '{caption}', Answer the question '{question}'\nAnswer: "

output = llm_pipeline(
    prompt,
    max_new_tokens=20  # Default, allows verbose
)
```

**Characteristics:**
- Natural LLM behavior
- Verbose, explanatory responses
- Poor for structured extraction
- Accuracy: ~13-15%

---

### Configuration 2: Token Limited

```python
# Restrict output length
prompt = f"Based on the image caption: '{caption}', Answer the question '{question}'\nAnswer: "

output = llm_pipeline(
    prompt,
    max_new_tokens=3  # Force brevity
)
```

**Characteristics:**
- Forces shorter outputs
- Still may include preamble
- Better but not optimal
- Accuracy: ~22-26%

---

### Configuration 3: Token + Prompt Restricted

```python
# Restrict both generation and instruction
prompt = f"Based on the image caption: '{caption}'. Answer in a single word the question: '{question}'\nAnswer: "

output = llm_pipeline(
    prompt,
    max_new_tokens=3  # Hard limit
)
```

**Characteristics:**
- Explicit single-word instruction
- LLM follows instruction
- Clean, parseable output
- Accuracy: **~43-52%** ⭐

---

### Configuration 4: Enhanced Prompt (Tested but verbose)

```python
# Very explicit instructions
prompt = f"""Based on the image caption: '{caption}'.
Answer in a single word the question: '{question}'
In plain simple English language without any emoticons or icons or font colors
or punctuation marks. I strongly state do not repeat the question, prompt used,
disclaimer, explanation or anything apart from answer, that is just provide the
answer in a single word in lowercase.
Remember if you are unable to answer based on the caption, mention 'NA'.
Answer: """

output = llm_pipeline(
    prompt,
    max_new_tokens=3
)
```

**Result:** Over-specification didn't improve beyond Configuration 3
**Learning:** Concise, clear instructions > verbose explanations

---

## Recommendations

### Best Practices

1. **Always Use Token Limits:**
   ```python
   max_new_tokens=3  # For single-word answers
   max_new_tokens=10  # For short phrases
   ```

2. **Explicit Output Format:**
   ```python
   "Answer in a single word: "
   "Respond with only the number: "
   "One-word answer only: "
   ```

3. **Greedy Decoding for Consistency:**
   ```python
   do_sample=False  # Deterministic outputs
   ```

4. **Balance Instruction Clarity:**
   - Be explicit but concise
   - Avoid over-explaining
   - Test simpler instructions first

---

## Comparative Analysis

### Configuration Impact Across Pipelines

| Pipeline | Default (20 tokens) | Max 3 Tokens | Max 3 + Instruction | Total Improvement |
|----------|--------------------:|-------------:|--------------------:|------------------:|
| **BLIP + Llama** | 15% | 23% (+53%) | 45% (+96%) | **+200%** 🎯 |
| **YOLO + Llama** | 18% | 28% (+56%) | 47% (+68%) | **+161%** |
| **BLIP + YOLO + Mistral** | 13% | 22% (+69%) | **52%** (+136%) ⭐ | **+300%** 🏆 |

**Key Findings:**

- 🎯 **Best Absolute Performance:** BLIP + YOLO + Mistral at **52%**
- 🏆 **Highest Relative Gain:** 4x improvement (13% → 52%)
- ⚡ **Most Effective Step:** Adding single-word instruction (2x boost)
- 📊 **Consistent Pattern:** All pipelines benefit dramatically from proper configuration

---

## Technical Details

### Why Greedy Decoding?

```python
do_sample = False  # Greedy: Always pick highest probability token
```

**Advantages:**
- Deterministic outputs (reproducible)
- No randomness from sampling
- Consistent evaluation

**Alternative (Sampling):**
```python
do_sample = True
temperature = 0.7  # Lower = more conservative
top_p = 0.9       # Nucleus sampling
```

**Use When:**
- Want diverse outputs
- Creative tasks
- Multiple valid answers

---

### Token Budget Analysis

| max_new_tokens | Avg Output Length | Exact Match | Semantic Match | Notes |
|---------------:|------------------:|------------:|---------------:|-------|
| 20 (default) | 15-20 tokens | 12.88% | 15.22% | ❌ Too verbose |
| 10 | 8-10 tokens | 18.5% | 22.3% | Still wordy |
| 5 | 4-5 tokens | 20.1% | 24.5% | Better |
| **3** | 2-3 tokens | 21.59% | 25.77% | ✅ Good balance |
| **1** ⭐ | 1 token | **38.2%** | **44.1%** | 🏆 Best with instruction |

**Key Insight:** Single token (`max_new_tokens=1`) achieves **2.5x better** accuracy than default when paired with clear instructions!

---

## Prompt Instruction Impact

### Without Instruction

```
Prompt: "Based on caption: 'two children sleeping', answer: 'how many children?'"
Output: "Based on the caption..."  ❌
```

### With Instruction

```
Prompt: "Caption: 'two children sleeping'. Answer in a single word: 'how many children?'"
Output: "2"  ✅
```

### Instruction Variations Tested

| Instruction Phrasing | Effectiveness |
|---------------------|---------------|
| "Answer in a single word" | ⭐⭐⭐⭐⭐ Best |
| "One-word answer only" | ⭐⭐⭐⭐ Good |
| "Respond with just one word" | ⭐⭐⭐⭐ Good |
| "Answer: " (no instruction) | ⭐⭐ Poor |
| "Please provide a concise response in lowercase..." | ⭐⭐⭐ Over-specified |

**Best:** Simple, direct instruction works best!

---

## Code Implementation

### Complete Configuration Setup

```python
from transformers import pipeline, AutoTokenizer
import torch

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    device=device,
    return_full_text=False  # Only return new tokens
)

# Optimal configuration
def generate_answer(prompt):
    """Generate answer with optimal settings"""
    output = llm_pipeline(
        prompt,
        max_new_tokens=3,
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = output[0]['generated_text']
    answer = generated_text.strip().split()[0]  # First token
    return answer.lower()

# Construct prompt with instruction
def build_prompt(caption, detections, question):
    """Build prompt with single-word instruction"""
    return f"""Based on the image caption: '{caption}' and detections: '{detections}'.
Answer in a single word the question: '{question}'
Answer: """
```

---

## Lessons Learned

### Key Insights

1. **LLMs Need Explicit Output Format:**
   - They're trained to be conversational
   - Must override default behavior
   - Clear instructions > token limits alone

2. **Token Limits Are Necessary But Insufficient:**
   - Prevent verbose outputs
   - Don't guarantee correct format
   - Combine with prompt instructions

3. **Simpler Instructions Work Better:**
   - "Answer in a single word" > lengthy explanations
   - Over-specification doesn't help
   - Clear and concise wins

4. **First Token Strategy:**
   - Extract first token as answer
   - Works well with proper configuration
   - Consistent across models

---

## References

- **Transformers Documentation:** [Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- **Generation Parameters:** [Generation Configuration](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)
- **Prompting Guide:** [Best Practices](https://www.promptingguide.ai/)

---

## Navigation

**← [Back to Home](index.md)**

### Related Experiments

- [Prompt Templates](prompt-templates.md) - Template engineering and optimization
- [In-Context Learning](in-context-learning.md) - Few-shot learning experiments
- [Results & Analysis](results.md) - Comprehensive performance analysis

### Pipeline Pages

- [Pipeline Overview](pipelines.md) - Compare all three architectures
- [BLIP + LLM](blip-llm-pipeline.md) | [YOLO + LLM](yolo-llm-pipeline.md) | [BLIP + YOLO + LLM](blip-yolo-llm-pipeline.md)
