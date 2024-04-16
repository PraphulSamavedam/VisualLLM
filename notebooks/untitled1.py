import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login
import random

import torch

torch.cuda.empty_cache()

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

in_context_example_indices = np.random.choice(numbers, 3)

def make_prompt(in_context_example_indices, test_example_index):
    ### WRITE YOUR CODE HERE 
    icl_dialogues = dataset[in_context_example_indices]
    icl_summaries = dataset['test'][in_context_example_indices]['summary']
    # print(f"Dialogues: {icl_dialogues} \nSummaries: {icl_summaries}")

    prompt = r""
    for dialogue, summary in zip(icl_dialogues, icl_summaries):
        prompt += f"\n{dialogue}\n" + f"{summary}\n\n\n"
    
    test_dialogue = dataset['test'][test_example_index]['dialogue']
    prompt += f"\n{test_dialogue}\n"

    return prompt