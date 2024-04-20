# %% [markdown]
# ### Base line model



# %%
import torch 
from transformers import pipeline  # Use a pipeline as a high-level helper
import os # For all os level functions
import json # For parsing the json files
# import pandas
# import parquet

# %%
import torch
print(torch.version.cuda)
import csv

# %% [markdown]
# #### Util functions
# 
# 

# %%
### Setup the questions and answers for the image files
# def obtain_questions_for_image_ids(image_ids: list, question_file_path: str= "questions.json"):
#     """This function obtains the questions for the list of the image ids passed from the file"""
#     with open(question_file_path) as file:
#         questions_data = json.load(file)
#     result = []
#     for question_details in questions_data['questions']:
#         if question_details['image_id'] in image_ids:
#             result.append(question_details)
#     return result


# def obtain_annotations_for_image_ids(image_ids: list, annotations_file_path: str = "annotations.json"):
#     """This function obtians the annotations for the list of the image ids passed from the file"""
#     with open(annotations_file_path, 'r') as file:
#         annotations_data = json.load(file)
#     result = []
#     for annotation_details in annotations_data['annotations']:
#         if annotation_details['image_id'] in image_ids:
#             result.append(annotation_details)
#     return result


# def get_refined_results_from_lists(relevant_questions:list, relevant_annotations:list):
#     """This provides refined format of the question, image, plausible answers and other meta info"""
#     result = []
#     for question_info in relevant_questions:
#         q_id = question_info['question_id']
#         question = question_info['question']
#         # print(f"Question ID: {q_id}, Question: {question}")
#         for annotation_info in relevant_annotations:
#             if annotation_info['question_id'] == q_id:
#                 answer = annotation_info["multiple_choice_answer"]
#                 plausile_answers = set(x['answer'] for x in annotation_info['answers'])
#                 result.append({"question_id": q_id, "question": question, 
#                                "answer": answer, "plausible_answers": plausile_answers,
#                                "image_id": annotation_info['image_id'], "question_type" :annotation_info['question_type'],
#                                "answer_type": annotation_info['answer_type']})

#     return result

# def get_results(image_ids: list, folder: str):
#     relevant_annotations = obtain_annotations_for_image_ids(image_ids, annotations_file_path=f"{folder}/annotations.json")
#     relevant_questions = obtain_questions_for_image_ids(image_ids, question_file_path=f"{folder}/questions.json")
#     return get_refined_results_from_lists(relevant_questions=relevant_questions, relevant_annotations=relevant_annotations)



# %%
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
questions_data = questions_data['questions']

image_files = []
mapped_question_answers = []
results = []
for indx, annotation_info in enumerate(annotations_data):
    q_id = annotation_info["question_id"]
    question = questions_data[indx]['question']
    answer = annotation_info['multiple_choice_answer']
    plausile_answers = set(x['answer'] for x in annotation_info['answers'])
    image_file = f"../data/validation/images/COCO_val2014_{str(annotation_info['image_id']).zfill(12)}.jpg"
    mapped_question_answers.append([annotation_info['image_id'], image_file, q_id, question, answer, plausile_answers, annotation_info['question_type'], annotation_info['answer_type']])
    image_files.append(image_file)
    
    results.append({"question_id": q_id, "question": question, 
                               "answer": answer, "plausible_answers": plausile_answers,
                               "image_id": annotation_info['image_id'], "question_type" :annotation_info['question_type'],
                               "answer_type": annotation_info['answer_type']})
    
    
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

# device = torch. ### Write code for the torch cuda if GPU is present
print("Generating captions using Blip Pipeline")
blipPipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device = 'cuda')
generated_captions = blipPipe(image_files)



### Get the generated text along with image_id, image_file
captions = [[image_files[indx], x[0]['generated_text']] for indx, x in enumerate(generated_captions)]

#     answers.append([img_id, caption, question, prompt, gen_text[0]['generated_text']])
    
# Define CSV file path
csv_file_path = 'blip_captions.csv'

# Write data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Image File', 'Caption'])
    # Write the data
    writer.writerows(captions)

print(f"Data has been written to {csv_file_path}")

# captaions_df = pd.DataFrame()
# for each_caption in captions:
#     captions_df.append()




lookup = dict()
for caption, image_id, image_file in zip(captions, image_ids, image_files):
    lookup[image_id]= (image_file, caption)
    

# %%
### Hugging Face token for running the application

import os
from huggingface_hub import login

login(token=token) ## This is bound to fail, add your token from chat and run

# %%


# %%

# Use a pipeline as a high-level helper
from transformers import pipeline

llamaPipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device = 'cuda')

# %%
### Zero Shot Performance
from tqdm import tqdm

# number_of_samples = 10
# sampled_results = results[:number_of_samples]

# %%
import csv

print(f"Zero Shot Performance")
print("=="* 50)

print(f"Default behavior")
print("=="* 50)

answers = []
for instance in tqdm(results):
    # instance = results[indx]
    img_id = instance['image_id']
    question = instance['question']
    file, caption = lookup[img_id]
    prompt = f"Based on the image caption (generated by BLIP model): {caption}, Answer the question '{question}' \n Answer: "
    gen_text = llamaPipe(prompt)
#     print(f"Generated text: {gen_text[0]['generated_text']}")
    answers.append([img_id, caption, question, prompt, gen_text[0]['generated_text']])
    
# Define CSV file path
csv_file_path = 'default_answers.csv'

# Write data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Image ID', 'Caption', 'Question', 'Prompt', 'Generated Answer'])
    # Write the data
    writer.writerows(answers)

print(f"Data has been written to {csv_file_path}")


### Generation configuration Limited to 3 tokens
print("=="* 50)
print(f"Zero Shot Performance with configuration limit only")
print("=="* 50)

generation_cfg_answers = []
for instance in tqdm(results):
    img_id = instance['image_id']
    question = instance['question']
    file, caption = lookup[img_id]
    prompt = f"Based on the image caption (generated by BLIP model): {caption}. Answer the question: '{question}'\n Answer: "
    gen_text = llamaPipe(prompt, max_new_tokens = 3)
#     print(f"Generated text: {gen_text[0]['generated_text']}")
    generation_cfg_answers.append([img_id, caption, question, prompt, gen_text[0]['generated_text']])

# Define CSV file path
csv_file_path = 'generation_cfg_answers.csv'

# Write data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Image ID', 'Caption', 'Question', 'Prompt', 'Generated Answer'])
    # Write the data
    writer.writerows(generation_cfg_answers)

print(f"Data has been written to {csv_file_path}")

#####
### Restriction by configuration to restrict to 3 tokens with restriction to single word in prompt
print("=="* 50)
print(f"Restriction by configuration to restrict to 3 tokens with restriction to single word in prompt")
print("=="* 50)

generation_cfg_prompt_restriction_answers = []
for instance in tqdm(results):
    img_id = instance['image_id']
    question = instance['question']
    file, caption = lookup[img_id]
    prompt = f"Based on the image caption (generated by BLIP model): {caption}. Answer in a single word the question: '{question}'\n Answer: "
    gen_text = llamaPipe(prompt, max_new_tokens = 3)
#     print(f"Generated text: {gen_text}")
    generation_cfg_prompt_restriction_answers.append(gen_text)

# Define CSV file path
csv_file_path = 'generation_cfg_prompt_restriction_answers.csv'

# Write data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Image ID', 'Caption', 'Question', 'Prompt', 'Generated Answer'])
    # Write the data
    writer.writerows(generation_cfg_prompt_restriction_answers)

print(f"Data has been written to {csv_file_path}")    

# %%


# %%
#### Configuration to generate only 3 tokens at max and Enhanced prompt
print("=="* 50)
print(f"Configuration to generate only 3 tokens at max and Enhanced prompt")
print("=="* 50)
prompt_with_generation_cfg_answers = []
for instance in tqdm(results):
    img_id = instance['image_id']
    question = instance['question']
    file, caption = lookup[img_id]
    prompt = f"Based on the image caption (generated by BLIP model): {caption}. \n In plain simple English language without any emoticons or icons or font colors or punctuation marks. I strongly state do not repeat the question, prompt used, disclaimer, explanantion or anything apart from answer, that is just provide the answer in a single word in lowercase, the question: '{question}'. Remember if you are unable to answer the question based on the caption provided by BLIP, mention 'NA'.\nAnswer: "
    gen_text = llamaPipe(prompt, max_new_tokens = 3)
#     print(f"Generated text: {gen_text}")
    prompt_with_generation_cfg_answers.append(gen_text)

# Define CSV file path
csv_file_path = 'prompt_with_generation_cfg_answers.csv'

# Write data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the headers
    writer.writerow(['Image ID', 'Caption', 'Question', 'Prompt', 'Generated Answer'])
    # Write the data
    writer.writerows(prompt_with_generation_cfg_answers)

print(f"Data has been written to {csv_file_path}")    

#### ToDo 
### Write a function to evaluate the model performance as accuracy or the model

### 

# %%
def evaluate_results(ground_truths: list[str], generated_texts: list[dict]):
    """ This function provides the results based on the two lists """
    pass




