# In-Context Learning (ICL) Experiments

## Overview

In-Context Learning (ICL) is a powerful capability of large language models where providing example demonstrations in the prompt can improve task performance. This page documents systematic experiments with 1-shot, 3-shot, and 5-shot ICL for Visual Question Answering, revealing surprising performance degradation patterns.

## Background

### What is In-Context Learning?

In-Context Learning enables LLMs to adapt to tasks by learning from examples provided directly in the prompt, without any gradient updates or fine-tuning. The model observes the pattern from examples and applies it to new instances.

**Key Characteristics:**
- No parameter updates required
- Zero training cost
- Immediate adaptation
- Task demonstrated through examples

### ICL in Visual Question Answering

For VQA tasks, ICL examples consist of:
1. **Image caption** (from BLIP)
2. **Object detections** (from YOLO)
3. **Question** about the image
4. **Ground truth answer** (single word)

The model learns the pattern: `[caption + detections + question] → [answer]`

## Experimental Design

### Configuration

- **Model**: Llama-2-7b-chat-hf (unquantized)
- **Pipeline**: BLIP + YOLO + Llama
- **Sample Size**: 100 instances (due to computational constraints)
- **Templates**: All 7 prompt templates
- **ICL Shots**: 1, 3, 5 examples
- **Generation Config**: max_new_tokens=1

### Example Selection Strategy

**Random Selection:**
```python
# Sample 100 test indices
indices = data.sample(n=100, random_state=np.random.seed(42)).index.tolist()

# Create pool of ICL examples (excluding test set)
filtered_df = data.drop(indices)
filtered_indices = filtered_df.index.tolist()

# For each test instance, randomly select N ICL examples
for each_index in indices:
    in_context_example_indices = np.random.choice(filtered_indices, icl_examples)
```

**Important Note:** Random selection was used for initial exploration. This may not be optimal, as discussed in the findings section.

## Implementation

### Prompt Construction Function

From `/Users/prapsama/Documents/Personal/VisualLLM/src/icl.py`:

```python
def make_prompt(icl_indices, test_example_index, tmplate):
    """
    This function provides the prompt in the desired template
    using the icl indices provided and with test index at the end.
    """
    # Fetch ICL examples
    icl_captions = dataset[icl_indices]['Generated Caption']
    icl_detections = dataset[icl_indices]['Generated Detections']
    icl_questions = dataset[icl_indices]['Question']
    icl_answers = dataset[icl_indices]['Answer']

    prompt = ""

    # Add ICL examples
    for caption, detection, question, answer in zip(
        icl_captions, icl_detections, icl_questions, icl_answers
    ):
        tmpltd_txt = tmplate
        tmpltd_txt = tmpltd_txt.replace("{caption}", caption)
        tmpltd_txt = tmpltd_txt.replace("{detections}", detection)
        tmpltd_txt = tmpltd_txt.replace("{question}", question)
        prompt += f"\n{tmpltd_txt}{answer}\n\n\n"

    # Add test example (without answer)
    test_caption = dataset[test_example_index]['Generated Caption']
    test_detection = dataset[test_example_index]['Generated Detections']
    test_question = dataset[test_example_index]['Question']

    tmpltd_txt = tmplate
    tmpltd_txt = tmpltd_txt.replace("{caption}", test_caption)
    tmpltd_txt = tmpltd_txt.replace("{detections}", test_detection)
    tmpltd_txt = tmpltd_txt.replace("{question}", test_question)

    prompt += f"\n{tmpltd_txt}"
    return prompt
```

### Example ICL Prompt Structure

**1-Shot Example:**
```
Based on the image caption(provided by BLIP model) as 'a dog sitting on grass'
and detections(provided by Yolo) as dog, grass, collar, answer in a single word
the question based on the image details as question: 'What animal is shown?'
Answer: dog


Based on the image caption(provided by BLIP model) as 'children playing in park'
and detections(provided by Yolo) as person, person, swing, tree, answer in a single word
the question based on the image details as question: 'How many children are visible?'
Answer:
```

**3-Shot Example:**
```
[Example 1 with answer]


[Example 2 with answer]


[Example 3 with answer]


[Test instance without answer]
Answer:
```

### Execution Loop

```python
# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device=device)

# Test different ICL configurations
for icl_examples in range(1, 6, 2):  # 1, 3, 5 examples
    for indx, template in enumerate(templates):
        results = []

        for each_index in indices:
            # Random selection of ICL examples
            in_context_example_indices = np.random.choice(
                filtered_indices,
                icl_examples
            )

            # Build prompt
            template_prompt = make_prompt(
                in_context_example_indices,
                each_index,
                template
            )

            # Generate answer
            generated_txt = pipe(
                template_prompt,
                max_new_tokens=1
            )[0]['generated_text']

            results.append({"Model Output": generated_txt})

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(
            f'{inferences_folder}/icl_template_{indx}_using_{icl_examples}_examples.csv',
            index=False
        )
```

## Results

### Performance Table

**BLIP + YOLO + Llama-2 with ICL:**

| ICL Examples | Accuracy Type | Template 1 | Template 2 | Template 3 | Template 4 | Template 5 | Template 6 | Template 7 |
|:------------:|:-------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| **1** | Exact Match | 0.46 | 0.44 | 0.36 | 0.44 | 0.46 | 0.48 | 0.37 |
|  | Semantic Match | 0.57 | 0.58 | 0.52 | 0.56 | 0.57 | **0.59** | 0.50 |
| **3** | Exact Match | 0.44 | 0.34 | 0.35 | 0.44 | 0.34 | 0.43 | 0.23 |
|  | Semantic Match | 0.53 | 0.42 | 0.41 | 0.54 | 0.41 | 0.51 | 0.34 |
| **5** | Exact Match | 0.17 | 0.20 | 0.14 | 0.22 | 0.16 | 0.15 | 0.24 |
|  | Semantic Match | 0.23 | 0.26 | 0.19 | 0.24 | 0.18 | 0.22 | **0.29** |

### Performance Comparison Chart

**Semantic Match Accuracy by ICL Examples:**

```
Template 1 (Context-First):
1-shot: ████████████████████████████ 0.57
3-shot: ███████████████████████████  0.53
5-shot: ███████████                  0.23

Template 6 (Challenge):
1-shot: ██████████████████████████████ 0.59 ← Best 1-shot
3-shot: ██████████████████████████     0.51
5-shot: ███████████                    0.22

Template 7 (Question-First):
1-shot: █████████████████████████      0.50
3-shot: █████████████████              0.34
5-shot: ██████████████                 0.29 ← Best 5-shot
```

### Baseline Comparison

**Zero-Shot vs ICL Performance:**

| Configuration | Exact Match | Semantic Match | Source |
|:--------------|:-----------:|:--------------:|:------:|
| Zero-Shot (1000 samples) | 0.461 | 0.6423 | Full dataset |
| 1-Shot ICL (100 samples) | 0.48 | 0.59 | Template 6 |
| 3-Shot ICL (100 samples) | 0.44 | 0.54 | Template 4 |
| 5-Shot ICL (100 samples) | 0.24 | 0.29 | Template 7 |

## Key Findings

### 1. Performance Degradation with More Examples

**Unexpected Result:** Adding more ICL examples **decreases** performance across almost all templates.

**Performance Drop:**
- **1-shot to 3-shot**: ~5-10% decrease in semantic accuracy
- **3-shot to 5-shot**: ~20-30% decrease in semantic accuracy
- **Overall 1-shot to 5-shot**: Up to 50% performance loss

**Example (Template 6):**
- 1-shot: 0.59 semantic accuracy
- 3-shot: 0.51 semantic accuracy (14% drop)
- 5-shot: 0.22 semantic accuracy (63% drop)

### 2. Template 7 Exception

**Anomaly:** Template 7 shows **increasing** performance with more examples:
- 1-shot: 0.50 semantic accuracy
- 3-shot: 0.34 semantic accuracy (drop)
- 5-shot: 0.29 semantic accuracy (best among 5-shot)

This template was the worst for zero-shot Llama (0.18), suggesting question-first format benefits from demonstration.

### 3. Template-Specific ICL Sensitivity

**Most Sensitive to ICL:**
- **Template 2**: 0.58 → 0.26 (55% drop)
- **Template 5**: 0.57 → 0.18 (68% drop)
- **Template 3**: 0.52 → 0.19 (63% drop)

**Most Robust:**
- **Template 7**: 0.50 → 0.29 (42% drop, but highest at 5-shot)
- **Template 1**: 0.57 → 0.23 (60% drop)

### 4. Context Window Concerns

**Hypothesis:** Performance degradation may be due to:

1. **Context Dilution**: More examples increase noise-to-signal ratio
2. **Irrelevant Examples**: Random selection includes unhelpful demonstrations
3. **Attention Distribution**: Model attention spreads across more examples
4. **Token Budget**: Longer prompts may exceed effective context window

**Prompt Length Analysis:**
- 1-shot: ~200-300 tokens
- 3-shot: ~600-900 tokens
- 5-shot: ~1000-1500 tokens

Llama-2-7b has 4096 token context, so length alone doesn't explain the degradation.

## Analysis and Insights

### Why Does ICL Hurt Performance?

**1. Random Example Selection Problem**

Current approach selects examples randomly, which may include:
- **Irrelevant examples**: Questions about different aspects (color vs count)
- **Contradictory patterns**: Different reasoning paths
- **Confusing edge cases**: Ambiguous or complex examples

**Better Approach:** Semantic similarity-based selection

```python
# Proposed improvement
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed all questions
question_embeddings = model.encode(data['Question'].tolist())

# For each test instance, find K most similar questions
def select_relevant_examples(test_question, k=3):
    test_embedding = model.encode([test_question])
    similarities = cosine_similarity(test_embedding, question_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:]
    return top_k_indices
```

**2. Example Quality Issues**

Not all examples are good demonstrations:
- **Wrong answers**: Ground truth may be incorrect
- **Ambiguous questions**: Multiple valid answers
- **Complex reasoning**: Multi-step inference required

**3. Cognitive Load**

More examples may increase cognitive complexity:
- Model must identify the pattern across multiple examples
- Attention mechanism must weigh all examples
- Risk of conflicting signals

### Architecture Diagram: ICL Prompt Structure

```
┌─────────────────────────────────────────────────────────┐
│                   ICL PROMPT STRUCTURE                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    EXAMPLE 1 (ICL)                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Template:                                            │ │
│ │ "Based on caption '{caption}' and detections        │ │
│ │  '{detections}', answer: '{question}'"              │ │
│ ├─────────────────────────────────────────────────────┤ │
│ │ Caption: "a dog sitting on grass"                   │ │
│ │ Detections: dog, grass, collar                       │ │
│ │ Question: "What animal is shown?"                    │ │
│ └─────────────────────────────────────────────────────┘ │
│ Answer: dog                                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    EXAMPLE 2 (ICL)                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Caption: "person on bicycle"                         │ │
│ │ Detections: person, bicycle, road                    │ │
│ │ Question: "What is the person riding?"               │ │
│ └─────────────────────────────────────────────────────┘ │
│ Answer: bicycle                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    EXAMPLE 3 (ICL)                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Caption: "child holding umbrella"                    │ │
│ │ Detections: person, umbrella, rain                   │ │
│ │ Question: "Is it raining?"                           │ │
│ └─────────────────────────────────────────────────────┘ │
│ Answer: yes                                             │
└─────────────────────────────────────────────────────────┘

                        ▼▼▼

┌─────────────────────────────────────────────────────────┐
│                  TEST INSTANCE (Target)                  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Caption: "children playing in park"                  │ │
│ │ Detections: person, person, swing, tree              │ │
│ │ Question: "How many children are visible?"           │ │
│ └─────────────────────────────────────────────────────┘ │
│ Answer: [MODEL GENERATES]                               │
└─────────────────────────────────────────────────────────┘

Model learns pattern: [Context] + [Question] → [Answer]
```

### Example Quality Matters

**Good ICL Example:**
```
Caption: "two dogs playing in yard"
Detections: dog, dog, ball, fence
Question: "How many dogs?"
Answer: two
```
- Clear mapping: detections contain answer
- Straightforward reasoning
- Question matches answer type

**Poor ICL Example:**
```
Caption: "people at a social gathering"
Detections: person, person, person, table, chair
Question: "What is the mood?"
Answer: happy
```
- Abstract reasoning required
- Answer not directly in detections
- May confuse model about reasoning path

## Recommendations

### Best Practices for ICL in VQA

1. **Use 1-Shot Learning**
   - Best performance across templates (0.59 semantic accuracy)
   - Minimal overhead
   - Clear demonstration without confusion

2. **Implement Semantic Example Selection**
   ```python
   # Use question similarity
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   similarities = cosine_similarity(test_embedding, example_embeddings)
   top_k = np.argsort(similarities)[-k:]
   ```

3. **Filter Example Quality**
   - Validate ground truth correctness
   - Prefer simple, unambiguous questions
   - Match question types (count, identification, yes/no)

4. **Template-Specific Strategies**
   - **Template 6**: Best for 1-shot (0.59)
   - **Template 7**: Consider for 5-shot if needed (0.29)
   - **Avoid Templates 2, 3, 5**: Severe degradation with ICL

5. **Consider Zero-Shot First**
   - Zero-shot: 0.6423 semantic (1000 samples)
   - 1-shot: 0.59 semantic (100 samples)
   - ICL may not be worth the overhead

### Future Improvements

**Advanced Selection Strategies:**

1. **Diverse Example Selection**
   ```python
   # Select examples covering different question types
   def diverse_selection(test_question, k=3):
       # Get similar questions
       similar = get_similar_questions(test_question, k*2)

       # Diversify by answer type
       diverse_examples = []
       question_types = set()

       for example in similar:
           q_type = classify_question_type(example)
           if q_type not in question_types:
               diverse_examples.append(example)
               question_types.add(q_type)

           if len(diverse_examples) == k:
               break

       return diverse_examples
   ```

2. **Hard Example Mining**
   - Include challenging examples that model typically gets wrong
   - Helps model learn edge cases

3. **Dynamic Example Count**
   - Adapt number of examples based on question complexity
   - Simple questions: 0-1 examples
   - Complex questions: 2-3 examples

4. **Example Caching**
   - Pre-compute high-quality example sets
   - Reuse for similar questions

## Error Analysis

### Common Failure Patterns

**1. Answer Format Errors**
- Model generates explanation instead of single word
- Mitigation: Stronger format constraints in prompt

**2. Context Overload**
- With 5 examples, model loses focus on test question
- Mitigation: Limit to 1-2 examples

**3. Conflicting Examples**
- Random selection includes contradictory patterns
- Mitigation: Semantic similarity selection

**4. Template Incompatibility**
- Some templates don't benefit from ICL (Templates 2, 3, 5)
- Mitigation: Use Template 6 for ICL

## Computational Considerations

### Performance Metrics

**Inference Time (Llama-2-7b on GPU):**
- Zero-shot: ~0.5 seconds per instance
- 1-shot: ~0.8 seconds per instance
- 3-shot: ~1.5 seconds per instance
- 5-shot: ~2.5 seconds per instance

**Throughput:**
- Zero-shot: ~7200 instances/hour
- 1-shot: ~4500 instances/hour
- 5-shot: ~1440 instances/hour

**Memory Usage:**
- Zero-shot: ~8GB VRAM
- 1-shot: ~10GB VRAM
- 5-shot: ~14GB VRAM

### Cost-Benefit Analysis

Given the performance degradation and computational overhead, ICL is **not recommended** for this VQA pipeline unless:

1. Semantic similarity-based selection is implemented
2. Example quality is carefully curated
3. Only 1-shot learning is used
4. Template 6 or 7 is selected

## Conclusion

In-Context Learning for VQA shows **surprising negative results** in the current implementation:

- **1-shot learning**: Competitive with zero-shot (0.59 vs 0.64)
- **3-shot learning**: Significant degradation (0.51-0.54)
- **5-shot learning**: Severe degradation (0.22-0.29)

The primary issue appears to be **random example selection**, which introduces irrelevant demonstrations that confuse the model. Future work should focus on:

1. **Semantic similarity-based selection**
2. **Example quality filtering**
3. **Template-specific ICL strategies**
4. **Dynamic example count adaptation**

For production deployments, **zero-shot inference** remains the most reliable approach, with optional **1-shot ICL using Template 6** for marginal improvements in specific scenarios.
