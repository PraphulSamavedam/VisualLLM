# Getting Started with VisualLLM

## Overview

This guide provides complete setup and usage instructions for the VisualLLM project, including environment setup, data preparation, pipeline execution, and experiment reproduction.

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.8+
- 16GB RAM
- 50GB disk space (for datasets)

**Recommended:**
- Python 3.9+
- 32GB RAM
- NVIDIA GPU with 16GB+ VRAM (for optimal performance)
- 100GB disk space

### Software Dependencies

- CUDA 11.8+ (for GPU acceleration)
- cuDNN 8.6+
- Git

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/VisualLLM.git
cd VisualLLM
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n visualllm python=3.9
conda activate visualllm
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
```
transformers>=4.30.0
torch>=2.0.0
datasets>=2.12.0
pandas>=1.5.0
numpy>=1.24.0
pillow>=9.5.0
opencv-python>=4.7.0
huggingface_hub>=0.15.0
tqdm>=4.65.0
```

**For YOLO:**
```bash
# Install YOLOv5
pip install yolov5
```

**For evaluation:**
```bash
pip install scikit-learn>=1.2.0
sentence-transformers>=2.2.0  # For semantic similarity
```

---

## Environment Setup

### 1. Set Hugging Face Token (for Llama)

Llama-2 models require authentication with Hugging Face:

**Method 1: Environment Variable**
```bash
export LLAMA_TOKEN="your_huggingface_token_here"
```

**Method 2: .env File**
```bash
# Create .env file
echo "LLAMA_TOKEN=your_huggingface_token_here" > .env
```

**Get Token:**
1. Visit [Hugging Face](https://huggingface.co/settings/tokens)
2. Create a new token with read access
3. Accept Llama-2 license at [model page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

### 2. Configure Paths

Edit `/Users/prapsama/Documents/Personal/VisualLLM/src/constants.py`:

```python
# Data folder paths
data_folder = "data"
train_folder = f"{data_folder}/train"
test_folder = f"{data_folder}/test"
validation_folder = f"{data_folder}/validation"

# Output paths
inferences_folder = "inferences"
results_folder = "results"

# Input file paths
sampled_qa_file_path = f"{data_folder}/sampled_qa.csv"
captions_file_path = f"{data_folder}/captions.csv"
detections_file_path = f"{data_folder}/detections.csv"
```

---

## Data Setup

### Automated Setup (Recommended)

Run the setup script to download and organize all datasets:

```bash
chmod +x setup.sh
./setup.sh
```

**What it does:**
1. Creates data directory structure
2. Downloads MSCOCO images (train, test, validation)
3. Downloads VQA v2.0 annotations and questions
4. Extracts and organizes files
5. Validates dataset integrity

**Directory Structure After Setup:**
```
data/
├── train/
│   ├── images/           # COCO train2014 (82,783 images)
│   ├── annotations.json  # VQA annotations
│   └── questions.json    # VQA questions
├── test/
│   ├── images/           # COCO test2015 (81,434 images)
│   └── questions.json    # Test questions (no annotations)
├── validation/
│   ├── images/           # COCO val2014 (40,504 images)
│   ├── annotations.json  # VQA annotations
│   └── questions.json    # VQA questions
└── indices.csv           # Sampled indices for ICL
```

### Manual Setup (Alternative)

If automated setup fails, follow these steps:

**1. Create Directory Structure:**
```bash
mkdir -p data/{train,test,validation}
mkdir -p inferences results
```

**2. Download COCO Images:**
```bash
# Training images (13GB)
cd data/train
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip && mv train2014 images
rm train2014.zip

# Validation images (6GB)
cd ../validation
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip && mv val2014 images
rm val2014.zip

# Test images (6GB)
cd ../test
wget http://images.cocodataset.org/zips/test2015.zip
unzip test2015.zip && mv test2015 images
rm test2015.zip
cd ../../
```

**3. Download VQA Annotations:**
```bash
# Training annotations
cd data/train
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
mv v2_mscoco_train2014_annotations.json annotations.json
rm v2_Annotations_Train_mscoco.zip
rmdir v2_Annotations_Train_mscoco

# Training questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
mv v2_OpenEnded_mscoco_train2014_questions.json questions.json
rm v2_Questions_Train_mscoco.zip

# Validation annotations
cd ../validation
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
mv v2_mscoco_val2014_annotations.json annotations.json
rm v2_Annotations_Val_mscoco.zip

# Validation questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
mv v2_OpenEnded_mscoco_val2014_questions.json questions.json
rm v2_Questions_Val_mscoco.zip
cd ../../
```

### Verify Setup

```bash
python -c "
import os
folders = ['data/train/images', 'data/validation/images', 'data/test/images']
files = ['data/train/annotations.json', 'data/train/questions.json',
         'data/validation/annotations.json', 'data/validation/questions.json']

print('Checking directories...')
for folder in folders:
    exists = os.path.exists(folder)
    print(f'{folder}: {'✓' if exists else '✗'}')

print('\nChecking files...')
for file in files:
    exists = os.path.exists(file)
    print(f'{file}: {'✓' if exists else '✗'}')
"
```

---

## Data Preprocessing

### 1. Generate Image Captions (BLIP)

```bash
cd src
python blip_captions.py
```

**What it does:**
- Loads BLIP-base model
- Processes images from validation set
- Generates descriptive captions
- Saves to `data/captions.csv`

**Output format:**
```csv
Image ID,Image file,Generated Caption
123456,COCO_val2014_000000123456.jpg,"a dog sitting on grass"
```

**Expected time:** ~2-3 hours for full validation set (GPU)

### 2. Generate Object Detections (YOLO)

```bash
python yolo_detections.py
```

**What it does:**
- Loads YOLOv5 model
- Detects objects in images
- Extracts bounding boxes and labels
- Saves to `data/detections.csv`

**Output format:**
```csv
Image ID,Image file,Generated Detections
123456,COCO_val2014_000000123456.jpg,"dog, grass, collar, person"
```

**Expected time:** ~1-2 hours for full validation set (GPU)

### 3. Generate Question-Answer Mapping

```bash
python generate_mapping_data.py
```

**What it does:**
- Loads VQA annotations and questions
- Merges with captions and detections
- Creates unified dataset
- Saves to `data/qa_mapping.csv`

**Output format:**
```csv
Image ID,Question,Answer,Generated Caption,Generated Detections
123456,"What animal?","dog","a dog on grass","dog, grass, collar"
```

### 4. Sample Dataset (Optional)

For quick experiments, create a smaller sample:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/qa_mapping.csv')
sample = df.sample(n=1000, random_state=42)
sample.to_csv('data/sampled_qa.csv', index=False)
print('Created 1000-sample dataset')
"
```

---

## Running Pipelines

### Basic Pipeline: BLIP + Llama

**Zero-shot inference on validation set:**

```bash
cd src
python blip_llama.py
```

**Output files:**
- `inferences/blip_llama/10k_default_answers.csv`
- `inferences/blip_llama/10k_generation_cfg_answers.csv`
- `inferences/blip_llama/10k_generation_cfg_prompt_restriction_answers.csv`

**Key parameters to modify:**

```python
# In blip_llama.py

# Adjust sample size
df = df[:1000]  # Process only 1000 samples

# Change generation config
max_new_tokens = 3  # Limit output length

# Modify prompt template
df["Prompt"] = "Your custom prompt template"
```

### YOLO + Llama Pipeline

```bash
python yolo_llama.py
```

Uses object detections instead of captions. Same output structure.

### Combined Pipeline: BLIP + YOLO + Llama

```bash
python blip_yolo_llama.py
```

**Best performing pipeline.** Combines both vision models.

### Mistral Pipelines

Replace Llama with Mistral (no token required):

```bash
python blip_mistral.py       # BLIP only
python yolo_mistral.py       # YOLO only
python blip_yolo_mistral.py  # Combined
```

### Template Experiments

Test all 7 prompt templates:

**For Llama:**
```bash
python llama_templates.py
```

**For Mistral:**
```bash
python mistral_templates.py
```

**Output:**
- Quantized results: `inferences/blip_yolo_llama_quantized_templates/`
- Unquantized results: `inferences/blip_yolo_llama_unquantized_templates/`

### In-Context Learning Experiments

```bash
python icl.py
```

**Configuration options:**

```python
# In icl.py

# Change sample size
samples = 100  # Default

# Modify ICL shot count
for icl_examples in range(1, 6, 2):  # 1, 3, 5 shots
    # Change to: range(1, 8, 2) for 1, 3, 5, 7 shots

# Custom template
templates = [your_custom_template]
```

---

## Evaluation

### Calculate Metrics

```bash
python evaluate.py
```

**Metrics computed:**
- Exact Match Accuracy
- Semantic Match Accuracy
- Per-template performance
- Confusion matrix

**Example evaluation code:**

```python
import pandas as pd
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util

# Load results
df = pd.read_csv('inferences/blip_yolo_llama/results.csv')

# Exact match
exact_match = accuracy_score(
    df['Ground Truth'].str.lower(),
    df['Generated Answer'].str.lower()
)

# Semantic match
model = SentenceTransformer('all-MiniLM-L6-v2')
gt_embeddings = model.encode(df['Ground Truth'].tolist())
pred_embeddings = model.encode(df['Generated Answer'].tolist())

similarities = util.cos_sim(gt_embeddings, pred_embeddings)
semantic_match = (similarities.diagonal() > 0.7).float().mean()

print(f"Exact Match: {exact_match:.4f}")
print(f"Semantic Match: {semantic_match:.4f}")
```

---

## Reproducing Paper Results

### 1. Generation Configuration Experiment

```bash
# Run YOLO + Mistral with different configs
python yolo_mistral.py

# Results will match Table 1 in README:
# Default: 15.22% semantic
# Max 3 tokens: 25.77% semantic
# Max 3 + single word: 52.09% semantic
```

### 2. Pipeline Comparison

```bash
# Run all pipelines
python blip_llama.py
python yolo_llama.py
python blip_yolo_llama.py
python blip_mistral.py
python yolo_mistral.py
python blip_yolo_mistral.py

# Results match Table 2:
# Best: BLIP + YOLO + Llama (64.23%)
```

### 3. Template Experiments

```bash
# Test all templates
python llama_templates.py
python mistral_templates.py

# Results match Tables 3-4:
# Llama best: Template 1 (58%)
# Mistral best: Template 7 (55.1%)
```

### 4. ICL Experiments

```bash
# Run ICL with 1, 3, 5 shots
python icl.py

# Results match Table 5:
# 1-shot: 59% (Template 6)
# 3-shot: 54% (Template 4)
# 5-shot: 29% (Template 7)
```

---

## Usage Examples

### Example 1: Quick Test on Single Image

```python
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline

# Load models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llama_pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device=device)

# Load image
image = Image.open("path/to/image.jpg")

# Generate caption
inputs = blip_processor(image, return_tensors="pt")
caption_ids = blip_model.generate(**inputs)
caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

# Ask question
question = "How many people are in the image?"
prompt = f"Based on the image caption: '{caption}', answer in a single word: '{question}'\nAnswer:"

answer = llama_pipe(prompt, max_new_tokens=3)[0]['generated_text']
print(f"Caption: {caption}")
print(f"Answer: {answer}")
```

### Example 2: Batch Processing

```python
import pandas as pd
from datasets import Dataset

# Load data
df = pd.read_csv('data/qa_mapping.csv')[:100]

# Create prompt
template = "Based on caption '{caption}' and detections '{detections}', answer: '{question}'"
df["Prompt"] = df.apply(
    lambda row: template.format(
        caption=row['Generated Caption'],
        detections=row['Generated Detections'],
        question=row['Question']
    ),
    axis=1
)

# Batch inference
dataset = Dataset.from_pandas(df)
outputs = llama_pipe(dataset["Prompt"], max_new_tokens=3, batch_size=8)

df["Generated Answer"] = [o[0]['generated_text'] for o in outputs]
df.to_csv('results.csv', index=False)
```

### Example 3: Custom Template

```python
# Define your template
custom_template = (
    "Image shows: {caption}. "
    "Objects detected: {detections}. "
    "Question: {question} "
    "Reply with ONE WORD only.\n"
    "Answer:"
)

# Apply to dataset
df["Prompt"] = df.apply(
    lambda row: custom_template.format(
        caption=row['Generated Caption'],
        detections=row['Generated Detections'],
        question=row['Question']
    ),
    axis=1
)

# Generate answers
outputs = llama_pipe(df["Prompt"].tolist(), max_new_tokens=3)
```

### Example 4: With ICL

```python
def create_icl_prompt(examples, test_instance, template):
    """Create prompt with in-context examples."""
    prompt = ""

    # Add examples
    for ex in examples:
        prompt += template.format(
            caption=ex['caption'],
            detections=ex['detections'],
            question=ex['question']
        )
        prompt += f"{ex['answer']}\n\n"

    # Add test instance
    prompt += template.format(
        caption=test_instance['caption'],
        detections=test_instance['detections'],
        question=test_instance['question']
    )

    return prompt

# Usage
icl_examples = [
    {"caption": "dog on grass", "detections": "dog, grass", "question": "What animal?", "answer": "dog"},
    {"caption": "two children playing", "detections": "person, person", "question": "How many children?", "answer": "two"}
]

test = {"caption": "cat on sofa", "detections": "cat, sofa", "question": "Where is the cat?"}

prompt = create_icl_prompt(icl_examples, test, template)
answer = llama_pipe(prompt, max_new_tokens=3)[0]['generated_text']
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```python
# Reduce batch size
batch_size = 4  # Default: 8

# Use quantization
import torch
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.bfloat16  # Saves ~50% memory
)

# Clear cache between runs
torch.cuda.empty_cache()
```

**2. Hugging Face Token Error**

```bash
# Verify token is set
echo $LLAMA_TOKEN

# Login manually
python -c "from huggingface_hub import login; login(token='your_token')"

# Check model access
# Visit: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# Accept license if not already done
```

**3. Slow Inference**

```python
# Use GPU
device = torch.device("cuda")

# Increase batch size (if memory allows)
batch_size = 16

# Use quantization
torch_dtype = torch.bfloat16

# Use smaller sample
df = df[:1000]  # Instead of full 10k
```

**4. Module Not Found**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.8+

# Verify installation
python -c "import transformers; print(transformers.__version__)"
```

**5. Data Download Fails**

```bash
# Check internet connection
ping images.cocodataset.org

# Resume interrupted download
wget -c http://images.cocodataset.org/zips/train2014.zip

# Alternative: Use curl
curl -O http://images.cocodataset.org/zips/train2014.zip

# Manual download
# Visit: https://cocodataset.org/#download
```

---

## Performance Optimization

### GPU Utilization

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Monitor GPU usage
# In terminal: watch -n 1 nvidia-smi
```

### Memory Management

```python
# Clear cache before large operations
torch.cuda.empty_cache()

# Use gradient checkpointing (for training)
model.gradient_checkpointing_enable()

# Limit max token generation
max_new_tokens = 3  # Optimal for VQA

# Process in smaller batches
for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    # Process batch
    torch.cuda.empty_cache()
```

### Parallelization

```python
# Use multiple GPUs
from torch.nn import DataParallel
model = DataParallel(model)

# Multi-processing for data loading
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, num_workers=4)
```

---

## Best Practices

### 1. Start Small
- Test on 100 samples first
- Verify pipeline works correctly
- Check output format
- Then scale to full dataset

### 2. Monitor Resources
- Watch GPU memory usage
- Track inference time
- Log errors and warnings
- Save intermediate results

### 3. Version Control
- Commit changes regularly
- Tag important experiments
- Document configuration changes
- Save model outputs

### 4. Reproducibility
```python
# Set random seeds
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### 5. Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting experiment...")
```

---

## Configuration Reference

### Model Configurations

```python
# Llama-2-7b-chat-hf
llama_config = {
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "max_new_tokens": 3,
    "temperature": 0.7,
    "top_p": 0.9,
    "return_full_text": False,
    "torch_dtype": torch.bfloat16  # For quantization
}

# Mistral-7B-Instruct-v0.2
mistral_config = {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "max_new_tokens": 3,
    "temperature": 0.7,
    "top_p": 0.9,
    "return_full_text": False,
    "torch_dtype": torch.bfloat16
}
```

### Pipeline Configurations

```python
# BLIP configuration
blip_config = {
    "model": "Salesforce/blip-image-captioning-base",
    "max_length": 50
}

# YOLO configuration
yolo_config = {
    "model": "yolov5s",  # yolov5s, yolov5m, yolov5l
    "conf_threshold": 0.25,
    "iou_threshold": 0.45
}
```

---

## Next Steps

After successful setup:

1. **Read Documentation:**
   - [Prompt Templates Guide](prompt-templates.md)
   - [In-Context Learning Analysis](in-context-learning.md)
   - [Results and Findings](results.md)

2. **Run Experiments:**
   - Start with quick test (100 samples)
   - Try different templates
   - Compare models
   - Experiment with ICL

3. **Analyze Results:**
   - Calculate metrics
   - Visualize performance
   - Identify error patterns
   - Optimize configuration

4. **Customize:**
   - Design new templates
   - Try different models
   - Implement improvements
   - Share findings

---

## Support

### Resources

- **Documentation**: `/docs/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Paper**: See `report/` directory

### Contact

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review code comments
- Consult troubleshooting section

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{visualllm2024,
  title={VisualLLM: Visual Question Answering with Large Language Models},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/VisualLLM}
}
```

---

## Navigation

**← [Back to Home](index.md)**

### Documentation

- [Pipeline Overview](pipelines.md) - Architecture comparison
- [BLIP + LLM](blip-llm-pipeline.md) | [YOLO + LLM](yolo-llm-pipeline.md) | [BLIP + YOLO + LLM](blip-yolo-llm-pipeline.md)
- [Experiments](generation-config.md) - Configuration and optimization
- [Results & Analysis](results.md) - Performance evaluation
