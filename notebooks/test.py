# %% [markdown]
# ### Base line model

# %%
# !pip uninstall -y torch torchvision torchaudio 
# huggingface transformers

# %%
#### CPU setup
# !pip install huggingface transformers torch torchvision

#### GPU setup
# !nvcc --version
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu112/torch_stable.html
# !pip install torch==1.13.1 torchvision==0.9.1 torchaudio===0.8.1 -f https://download.pytorch.org/whl/cu121/torch_stable.html
    
# !pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 torchdata==0.7.1 torchsummary==1.5.1 torchtext==0.17.1 torchvision==0.17.1+cu121 tornado==6.3.3 tqdm==4.66.2

# %%
import torch 
from transformers import pipeline  # Use a pipeline as a high-level helper
import os # For all os level functions
import json # For parsing the json files

import csv
# %%
### Setup the questions and answers for the image files
def obtain_questions_for_image_ids(image_ids: list, question_file_path: str= "questions.json"):
    """This function obtains the questions for the list of the image ids passed from the file"""
    with open(question_file_path) as file:
        questions_data = json.load(file)
    result = []
    for question_details in questions_data['questions']:
        if question_details['image_id'] in image_ids:
            result.append(question_details)
    return result


def obtain_annotations_for_image_ids(image_ids: list, annotations_file_path: str = "annotations.json"):
    """This function obtians the annotations for the list of the image ids passed from the file"""
    with open(annotations_file_path, 'r') as file:
        annotations_data = json.load(file)
    result = []
    for annotation_details in annotations_data['annotations']:
        if annotation_details['image_id'] in image_ids:
            result.append(annotation_details)
    return result


def get_refined_results_from_lists(relevant_questions:list, relevant_annotations:list):
    """This provides refined format of the question, image, plausible answers and other meta info"""
    result = []
    for question_info in relevant_questions:
        q_id = question_info['question_id']
        question = question_info['question']
        # print(f"Question ID: {q_id}, Question: {question}")
        for annotation_info in relevant_annotations:
            if annotation_info['question_id'] == q_id:
                answer = annotation_info["multiple_choice_answer"]
                plausile_answers = set(x['answer'] for x in annotation_info['answers'])
                result.append({"question_id": q_id, "question": question, 
                               "answer": answer, "plausible_answers": plausile_answers,
                               "image_id": annotation_info['image_id'], "question_type" :annotation_info['question_type'],
                               "answer_type": annotation_info['answer_type']})

    return result

def get_results(image_ids: list, folder: str):
    relevant_annotations = obtain_annotations_for_image_ids(image_ids, annotations_file_path=f"{folder}/annotations.json")
    relevant_questions = obtain_questions_for_image_ids(image_ids, question_file_path=f"{folder}/questions.json")
    return get_refined_results_from_lists(relevant_questions=relevant_questions, relevant_annotations=relevant_annotations)



# %%
validation_folder = "../data/validation"
validation_images_folder = f"{validation_folder}/images"

validation_image_files = [os.path.join(validation_images_folder, x) for x in os.listdir(validation_images_folder)]

sample_validation_image_files = validation_image_files

# %% [markdown]
# #### Setup for analysis

# %%
# sample_validation_image_files

# %%
image_files = sample_validation_image_files
image_ids = [int(image_file.split(os.sep)[-1].split(".")[0].split("_")[-1]) for image_file in image_files]

print(len(set(image_ids)))
# relevant_annotations = obtain_annotations_for_image_ids(image_ids, annotations_file_path=f"{validation_folder}/annotations.json")
# relevant_questions = obtain_questions_for_image_ids(image_ids, question_file_path=f"{validation_folder}/questions.json")

print("Obtaining the images to annotations")
annotations_file_path = f"{validation_folder}/annotations.json"
with open(annotations_file_path, 'r') as file:
        annotations_data = json.load(file)
annotations_data = annotations_data['annotations']

questions_file_path = f"{validation_folder}/questions.json"
with open(questions_file_path, 'r') as file:
        questions_data = json.load(file)
print(questions_data.keys())
questions_data = questions_data['questions']

image_files = []
mapped_question_answers = []
for indx, annotation_info in enumerate(annotations_data):
    q_id = annotation_info["question_id"]
    question = questions_data[indx]['question']
    answer = annotation_info['multiple_choice_answer']
    plausile_answers = set(x['answer'] for x in annotation_info['answers'])
    image_file = f"../data/validation/images/COCO_val2014_{str(annotation_info['image_id']).zfill(12)}.jpg"
    mapped_question_answers.append([annotation_info['image_id'], image_file, q_id, question, answer, plausile_answers, annotation_info['question_type'], annotation_info['answer_type']])
    image_files.append(image_file)
    break
    
# Define CSV file path
csv_file_path = 'mapped_question_answers.csv'

# Write data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Image ID', 'Image file', 'Question ID', "Question", "Answer", "Plausible answers", "Question Type", "Anwer Type"])
    # Write the data
    writer.writerows(mapped_question_answers)

print(f"Data has been written to {csv_file_path}")
                               

# annotations_file_path = f"{validation_folder}/questions.json"
# with open(annotations_file_path, 'r') as file:
#         annotations_data = json.load(file)
# print(annotations_data['questions'][45]['image_id'], 
#      annotations_data['questions'][45]['question_id'],
#      len(annotations_data['questions']))

# results = get_results(image_ids, validation_folder)