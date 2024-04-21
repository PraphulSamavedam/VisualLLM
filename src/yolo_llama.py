"""
This file processes using YOLO and LLama for the sampled qa files and stores the generated answers
in the folder yolo_llama under the inferences folder.
"""
import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login
import torch
from constants import detections_file_path, inferences_folder


torch.cuda.empty_cache()

# File Names:
default_answers_file = "yolo_llama/10k_default_answers.csv"
gen_cfg_restricted_file = "yolo_llama/10k_generation_cfg_answers.csv"
gen_cfg_and_prompt_restricted_file = "yolo_llama/10k_generation_cfg_prompt_restriction_answers.csv"
gen_cfg_with_enhanced_prompt_file = "yolo_llama/10k_enhanced_prompt_with_generation_cfg_answers.csv"


### Processing the data
print("Obtaining the required data")
df = pd.read_csv(detections_file_path)

# Sets the Hugging Face token for running LLama
token = os.environ.get("LLAMA_TOKEN", "Missing LLAMA_TOKEN in environment variables")
login(token=token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llamaPipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device = device)
# results = df.to_dict('records')


### Zero Shot Performance with 20 tokens
print("Running Zero shot on LLama")

# Basic Prompt
df["Prompt"] = "Based on the Object detections (generated by YOLOv5): '" \
                + df["Generated Detections"] + "', Answer the question '" \
                + df["Question"] + "' \n Answer: "

answers = [{
    'Image ID': instance['Image ID'],
    'Detections': instance['Generated Detections'],
    'Question': instance['Question'],
    'Prompt': instance['Prompt'],
    'Generated Answer': llamaPipe(instance['Prompt'], max_new_tokens=20)[0]['generated_text']
} for _, instance in tqdm(df.iterrows())]

# Store the results
pd.DataFrame(answers).to_csv(f"{inferences_folder}/{default_answers_file}", index=False)
print(f"Data has been written to {inferences_folder}/{default_answers_file}")

### Zero Shot Performance with configuration limit to 3 tokens only
print("=="* 50)
print("Zero Shot Performance with configuration limit to 3 tokens only")
print("=="* 50)
# Basic Prompt is already in df["Prompt"]

generation_cfg_answers = [{
    'Image ID': instance['Image ID'],
    'Detections': instance['Generated Detections'],
    'Question': instance['Question'],
    'Prompt': instance['Prompt'],
    'Generated Answer': llamaPipe(instance['Prompt'], max_new_tokens=3)[0]['generated_text']
} for _, instance in tqdm(df.iterrows())]

# Store the results
pd.DataFrame(generation_cfg_answers).to_csv(f'{inferences_folder}/{gen_cfg_restricted_file}', index=False)
print(f"Data has been written to f{inferences_folder}/{gen_cfg_restricted_file}")


### Restriction by configuration to restrict to 3 tokens with restriction to single word in prompt
print("=="* 50)
print("Running Zero Shot Performance with Restriction by configuration to restrict to \
       3 tokens with restriction to single word in prompt")
print("=="* 50)

# Updates the prompt to include the single word restriction along with configuration restriction
df["Prompt"] = "Based on the Object detections (generated by YOLOv5): '" \
                + df["Generated Detections"] + "'. Answer in a single word the question: " \
                + df["Question"] + "'\n Answer: "

generation_cfg_prompt_restriction_answers = [{
    'Image ID': instance['Image ID'],
    'Detections': instance['Generated Detections'],
    'Question': instance['Question'],
    'Prompt': instance['Prompt'],
    'Generated Answer': llamaPipe(instance['Prompt'], max_new_tokens=3)[0]['generated_text']
} for _, instance in tqdm(df.iterrows())]

# Store the results
pd.DataFrame(generation_cfg_prompt_restriction_answers).to_csv(f"{inferences_folder}/{gen_cfg_and_prompt_restricted_file}", index=False)
print(f"Data has been written to {inferences_folder}/{gen_cfg_and_prompt_restricted_file}")


### Configuration to generate only 3 tokens at max and enhanced prompt
print("Running Zero Shot Performance with Restriction by Configuration to generate only 3 tokens at max and Enhanced prompt")

# Update the prompt to use the enhanced prompt
df["Prompt"] = "Based on the Object detections (generated by YOLOv5): '" \
                + df["Generated Detections"] + "'. Answer in a single word the question: '" \
                + df["Question"] + "' \n In plain simple English language without any emoticons or icons or font colors "\
                + "or punctuation marks. I strongly state do not repeat the question, prompt used, disclaimer, explanantion" \
                + "or anything apart from answer, that is just provide the answer in a single word in lowercase, the question: '" \
                + df["Question"] \
                + "'. Remember if you are unable to answer the question based on the caption provided by BLIP, mention 'NA'.\nAnswer: "

prompt_with_generation_cfg_answers = [{
    'Image ID': instance["Image ID"],
    'Detections': df["Generated Detections"],
    'Question': instance['Question'],
    'Prompt': instance["Prompt"],
    'Generated Answer': llamaPipe(instance["Prompt"], max_new_tokens=3)[0]['generated_text']
} for _, instance in tqdm(df.iterrows())]

# Store the results
pd.DataFrame(prompt_with_generation_cfg_answers).to_csv(f"{inferences_folder}/{gen_cfg_with_enhanced_prompt_file}", index=False)
print(f"Data has been written to {inferences_folder}/{gen_cfg_with_enhanced_prompt_file}")
