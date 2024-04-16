import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login

import torch

torch.cuda.empty_cache()

captions_file = "../data/10k_blip_captions.csv"

detections_file = "../data/10k_yolo_detections.csv"

mapped_question_answers_file = "../data/10k_mapped_question_answers.csv"

captions_data = pd.read_csv(captions_file)
detections_data = pd.read_csv(detections_file)
qa_data = pd.read_csv(mapped_question_answers_file)


data = detections_data
print(data.head(4))

# Create a lookup dictionary
lookup = data.set_index('Image ID')[['Image file', 'Generated Caption', 'Generated Detections']].T.to_dict('list')

results = data.to_dict('records')

# Initialize Llama pipeline
# llamaPipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device='cuda')

pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device='cuda')

# print("Running Zero shot on LLama")
# # Zero Shot Performance with 20 tokens
# answers = [{
#     'Image ID': instance['Image ID'],
#     'Caption': lookup[instance['Image ID']][1],
#     'Detections': lookup[instance['Image ID']][2],
#     'Question': instance['Question'],
#     'Prompt': f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]}, and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}, Answer the question '{instance['Question']}' \n Answer: ",
#     'Generated Answer': pipe(f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]}, and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}, Answer the question '{instance['Question']}' \n Answer: ", max_new_tokens=20)[0]['generated_text']
# } for instance in tqdm(results)]

# Write data to a CSV file
# pd.DataFrame(answers).to_csv('10k_default_answers_with_detections_mistral.csv', index=False)

# Zero Shot Performance with configuration limit only
# print("Running Zero Shot Performance with configuration limit only")
# generation_cfg_answers = [{
#     'Image ID': instance['Image ID'],
#     'Caption': lookup[instance['Image ID']][1],
#     'Detections': lookup[instance['Image ID']][2],
#     'Question': instance['Question'],
#     'Prompt': f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]} and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}, Answer the question: '{instance['Question']}'\n Answer: ",
#     'Generated Answer': pipe(f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]} and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}, Answer the question: '{instance['Question']}'\n Answer: ", max_new_tokens=3)[0]['generated_text']
# } for instance in tqdm(results)]

# # Write data to a CSV file
# pd.DataFrame(generation_cfg_answers).to_csv('10k_generation_cfg_answers_with_detections_mistral.csv', index=False)

# Restriction by configuration to restrict to 3 tokens with restriction to single word in prompt
print("Running Zero Shot Performance with Restriction by configuration to restrict to 3 tokens with restriction to single word in prompt")
generation_cfg_prompt_restriction_answers = [{
    'Image ID': instance['Image ID'],
    'Caption': lookup[instance['Image ID']][1],
    'Detections' : lookup[instance['Image ID']][2],
    'Question': instance['Question'],
    'Prompt': f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]} and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}, Answer in a single word the question: '{instance['Question']}'\n Answer: ",
    'Generated Answer': pipe(f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]} and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}, Answer in a single word the question: '{instance['Question']}'\n Answer: ", max_new_tokens=3)[0]['generated_text']
} for instance in tqdm(results)]

# Write data to a CSV file
pd.DataFrame(generation_cfg_prompt_restriction_answers).to_csv('10k_generation_cfg_prompt_restriction_answers_with_detections_mistral.csv', index=False)

# Configuration to generate only 3 tokens at max and Enhanced prompt
print("Running Zero Shot Performance with Restriction by Configuration to generate only 3 tokens at max and Enhanced prompt")
prompt_with_generation_cfg_answers = [{
    'Image ID': instance['Image ID'],
    'Caption': lookup[instance['Image ID']][1],
    'Detections' : lookup[instance['Image ID']][2],
    'Question': instance['Question'],
    'Prompt': f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]} and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}. \n In plain simple English language without any emoticons or icons or font colors or punctuation marks. I strongly state do not repeat the question, prompt used, disclaimer, explanantion or anything apart from answer, that is just provide the answer in a single word in lowercase, the question: '{instance['Question']}'. Remember if you are unable to answer the question based on the caption provided by BLIP, mention 'NA'.\nAnswer: ",
    'Generated Answer': pipe( f"Based on the image caption (generated by BLIP model): {lookup[instance['Image ID']][1]} and and Object detections (generated by YOLOv5): {lookup[instance['Image ID']][2]}. \n In plain simple English language without any emoticons or icons or font colors or punctuation marks. I strongly state do not repeat the question, prompt used, disclaimer, explanantion or anything apart from answer, that is just provide the answer in a single word in lowercase, the question: '{instance['Question']}'. Remember if you are unable to answer the question based on the caption provided by BLIP, mention 'NA'.\nAnswer: ", max_new_tokens=3)[0]['generated_text']
} for instance in tqdm(results)]

# Write data to a CSV file
pd.DataFrame(prompt_with_generation_cfg_answers).to_csv('10k_enhanced_prompt_with_generation_cfg_answers_with_detections_mistral.csv', index=False)