# %%
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login
import random
from datasets import Dataset

import torch
import time

# torch.cuda.empty_cache()

detections_file = "../data/10k_yolo_detections.csv"
samples = 100

detections_data = pd.read_csv(detections_file)
data = detections_data

indices = data.sample(n=samples, random_state = np.random.seed(42)).index.tolist()

# Convert the list of indices to a DataFrame
indices_df = pd.DataFrame(indices, columns=['Index'])

# Save the DataFrame to a CSV file
indices_df.to_csv('indices.csv', index=False)

print("Indices saved to 'indices.csv'")

filtered_df = data.drop(indices)
filtered_indices = filtered_df.index.tolist()

dataset = Dataset.from_pandas(data)

in_context_example_indices = np.random.choice(filtered_indices, 3)

dataset[in_context_example_indices]

# %%
pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device='cuda')

templates = ["Based on the image caption(provided by BLIP model) as '{caption}' and detections(provided by Yolo) as {detections}, answer in a single word the question based on the image details as question: '{question}'\nAnswer: ",
             "Using the image caption '{caption}' and detected objects '{detections}', answer the following question with a single word: '{question}'.\nAnswer: ",
             "Caption: '{caption}'. Detected Objects: '{detections}'. What is the one-word answer to this question about the image: '{question}'?\nAnswer: ",
             "Given the description '{caption}' and identified elements '{detections}', provide a one-word response to this inquiry about the image: '{question}'.\nAnswer: ",
             "From the image caption '{caption}' and object detections '{detections}', find the answer to: '{question}'. Respond in just one word.\nAnswer: ",
             "Challenge: With the caption '{caption}' and objects detected as '{detections}', determine the single-word answer to the question: '{question}'.\nAnswer: ",
             "Answer in a single word for the question: {question} using image caption: {caption} and object detections: {detections}.\nAnswer: "]

def make_prompt(in_context_example_indices, test_example_index, template):
    ### WRITE YOUR CODE HERE 
    icl_captions = dataset[in_context_example_indices]['Generated Caption']
    icl_detections = dataset[in_context_example_indices]['Generated Detections']
    icl_questions = dataset[in_context_example_indices]['Question']
    icl_answers = dataset[in_context_example_indices]['Answer']

    prompt = r""
    for caption, detection, question, answer in zip(icl_captions, icl_detections, icl_questions, icl_answers):
        # print(caption, detection, question, answer)
        tmpltd_txt = template
        tmpltd_txt = tmpltd_txt.replace("{caption}",caption)
        tmpltd_txt = tmpltd_txt.replace("{detections}",detection)
        tmpltd_txt = tmpltd_txt.replace("{question}",question)
        prompt += f"\n{tmpltd_txt}" + f"{answer}\n\n\n"
    # print(f"Prompt with ICLP{prompt}")

    test_caption = dataset[test_example_index]['Generated Caption']
    test_detection = dataset[test_example_index]['Generated Detections']
    test_question = dataset[test_example_index]['Question']
    test_answer = dataset[test_example_index]['Answer']

    tmpltd_txt = template
    tmpltd_txt = tmpltd_txt.replace("{caption}",test_caption)
    tmpltd_txt = tmpltd_txt.replace("{detections}",test_detection)
    tmpltd_txt = tmpltd_txt.replace("{question}",test_question)
    
    prompt += f"\n{tmpltd_txt}"
    # print(f"Final prompt:{prompt}")
    return prompt

# %%

for icl_examples in range(1, 6, 2):
    for indx, template in enumerate(templates):
        results = []
        # Record the start time
        start_time = time.time()
        for each_index in indices:
            in_context_example_indices = np.random.choice(filtered_indices, icl_examples)
            print(f"Using indices{in_context_example_indices}")
            template_prompt = make_prompt(in_context_example_indices, each_index, template)
            generated_txt = pipe(template_prompt, max_new_tokens=1)[0]['generated_text']
            results.append({"Model Output": generated_txt})
        df = pd.DataFrame(results)
        df.to_csv(f'../data/icl_template_{indx}_using_{icl_examples}_examples.csv', index=False)
        
        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

