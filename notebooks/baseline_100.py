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
# import pandas
# import parquet
from tqdm import tqdm
# %%
import torch
print(torch.version.cuda)
import csv
import numpy as np


validation_folder = "../data/validation"
validation_images_folder = f"{validation_folder}/images"


print("Obtaining the annotations")
annotations_file_path = f"{validation_folder}/annotations.json"
with open(annotations_file_path, 'r') as file:
        annotations_data = json.load(file)
annotations_data = annotations_data['annotations']

print("Obtaining the questions")
questions_file_path = f"{validation_folder}/questions.json"
with open(questions_file_path, 'r') as file:
        questions_data = json.load(file)
questions_data = questions_data['questions']

#### Sample random 100 annotations and questions
annotations = np.array(annotations_data)
questions = np.array(questions_data)

indices = np.random.choice(len(annotations), 100, replace=False)

# Use the indices to slice the lists
annotations_data = annotations[indices]
questions_data = questions[indices]

mapped_question_answers = []
results = []
image_ids = []
image_files = []
for indx, annotation_info in enumerate(annotations_data):
    q_id = annotation_info["question_id"]
    question = questions_data[indx]['question']
    answer = annotation_info['multiple_choice_answer']
    plausile_answers = set(x['answer'] for x in annotation_info['answers'])
    image_file = f"../data/validation/images/COCO_val2014_{str(annotation_info['image_id']).zfill(12)}.jpg"

    image_ids.append(annotation_info['image_id'])
    image_files.append(image_file)
    mapped_question_answers.append([annotation_info['image_id'], image_file, q_id, question, answer, plausile_answers, annotation_info['question_type'], annotation_info['answer_type']])
    results.append({"question_id": q_id, "question": question, 
                               "answer": answer, "plausible_answers": plausile_answers,
                               "image_id": annotation_info['image_id'], "question_type" :annotation_info['question_type'],
                               "answer_type": annotation_info['answer_type']})
    
    
# Define CSV file path
csv_file_path = '../data/100_mapped_question_answers.csv'

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
blipPipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device = 'cuda', max_new_tokens=15)

generated_captions = blipPipe(image_files)

# generated_captions = []
# for image in tqdm(image_files):
#     generated_caption = blipPipe(image_files)
#     generated_captions.append(generated_caption)



### Get the generated text along with image_id, image_file
captions = [[image_files[indx], x[0]['generated_text']] for indx, x in enumerate(generated_captions)]

#     answers.append([img_id, caption, question, prompt, gen_text[0]['generated_text']])
    
# Define CSV file path
csv_file_path = '100_blip_captions.csv'

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
token = "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"

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
csv_file_path = '100_default_answers.csv'

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
csv_file_path = '100_generation_cfg_answers.csv'

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
csv_file_path = '100_generation_cfg_prompt_restriction_answers.csv'

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
csv_file_path = '100_prompt_with_generation_cfg_answers.csv'

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



