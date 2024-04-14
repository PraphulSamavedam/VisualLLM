import torch 
import os # For all os level functions
import json # For parsing the json files
import csv
from constants import validation_folder, validation_images_folder, samples
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


### ==========================

### ==========================


# Define CSV file path
csv_file_path = 'mapped_question_answers.csv'
parquet_file_path = 'data.parquet'

validation_image_files = [os.path.join(validation_images_folder, x) for x in os.listdir(validation_images_folder)]
image_files = validation_image_files

if samples is not None:
    image_files = image_files[:samples]

print("Obtain the image ids for the images")
image_ids = [int(image_file.split(os.sep)[-1].split(".")[0].split("_")[-1]) for image_file in image_files]
print(len(set(image_ids)))

print("Obtaining the image annotations data")
annotations_file_path = f"{validation_folder}/annotations.json"
with open(annotations_file_path, 'r') as file:
        annotations_data = json.load(file)
annotations_data = annotations_data['annotations']

print("Obtaining the image questions data")
questions_file_path = f"{validation_folder}/questions.json"
with open(questions_file_path, 'r') as file:
        questions_data = json.load(file)
questions_data = questions_data['questions']

### 
# image_files = []
mapped_question_answers = []
results = []
for indx, annotation_info in enumerate(annotations_data):
    q_id = annotation_info["question_id"]
    question = questions_data[indx]['question']
    answer = annotation_info['multiple_choice_answer']
    plausile_answers = set(x['answer'] for x in annotation_info['answers'])
    image_file = f"../data/validation/images/COCO_val2014_{str(annotation_info['image_id']).zfill(12)}.jpg"
    mapped_question_answers.append([annotation_info['image_id'], image_file, q_id, question, answer, plausile_answers, annotation_info['question_type'], annotation_info['answer_type']])
    # image_files.append(image_file)
    results.append({"question_id": q_id, "question": question,
                    "answer": answer,
                    "plausible_answers": plausile_answers,
                    "image_id": annotation_info['image_id'],
                    "question_type" :annotation_info['question_type'],
                    "answer_type": annotation_info['answer_type'],
                    "image_file": image_file})

# Generate the mapping of the image, question, answer and store it
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Image ID', 'Image file', 'Question ID', "Question", "Answer", "Plausible answers", "Question Type", "Anwer Type"])
    # Write the data
    writer.writerows(mapped_question_answers)

print(f"Successfully written to {csv_file_path}")

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(results)

# Define the PyArrow schema
fields = [pa.field(name, pa.string()) for name in df.columns]
schema = pa.schema(fields)
# Convert the DataFrame to a PyArrow Table
table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
# Write the table to a Parquet file
pq.write_table(table, parquet_file_path)

print(f"Data has been written to parquet file {parquet_file_path}")