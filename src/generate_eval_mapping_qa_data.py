import json
import numpy as np
import pandas as pd
import os
from constants import annotations_file_path, questions_file_path, sampled_qa_file_path

# Setup
SEED = 42
token = os.environ.get("LLAMA_TOKEN", "Missing LLAMA_TOKEN in environment variable")

# login(token=token) ## This is bound to fail, add your token from chat and run

print("Obtaining the images to annotations")
# Load annotations and questions
with open(annotations_file_path, 'r') as file:
    annotations_data = json.load(file)['annotations']
with open(questions_file_path, 'r') as file:
    questions_data = json.load(file)['questions']

# Convert to numpy arrays for efficient operations
annotations = np.array(annotations_data)
questions = np.array(questions_data)

# Sample random 10000 annotations and questions
np.random.seed(SEED)
indices = np.random.choice(len(annotations), 10000, replace=False)
annotations_data = annotations[indices]
questions_data = questions[indices]

# Prepare data
print("Prepare data")
data = [{
    "Image ID": annotation_info["image_id"],
    "Image file": f"../data/validation/images/COCO_val2014_{str(annotation_info['image_id']).zfill(12)}.jpg",
    "Question ID": annotation_info["question_id"],
    "Question": questions_data[indx]['question'],
    "Answer": annotation_info['multiple_choice_answer'],
    "Plausible answers": set(x['answer'] for x in annotation_info['answers']),
    "Question Type": annotation_info['question_type'],
    "Answer Type": annotation_info['answer_type']
} for indx, annotation_info in enumerate(annotations_data)]

# Write data to a CSV file
df = pd.DataFrame(data)
df.to_csv(sampled_qa_file_path, index=False)
