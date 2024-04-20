"""
This file generates the required mapping of question, answers for 
easier processing of evaluations of our approaches

Version: 1.0.0
"""

import json # For parsing the json files
from constants import mapped_qa_file_path, questions_file_path, annotations_file_path
import pandas as pd


debug = True

# Load annotations and questions data
if debug:
     print("Obtaining the questions data for these annotations")
with open(annotations_file_path, 'r') as file:
    annotations_data = json.load(file)['annotations']
with open(questions_file_path, 'r') as file:
    questions_data = json.load(file)['questions']

# Get the mapping of questions and answers to store them into .csv file for quick lookup
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
df.to_csv(mapped_qa_file_path, index=False)
print(f"Successfully written to {mapped_qa_file_path}")
