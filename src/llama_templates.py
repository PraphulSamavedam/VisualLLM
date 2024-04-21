"""
This file explores the performance of both quantized and unquantized LLama model 
through various templates for the first 1000 samples only.
"""

# !pip install datasets
# !pip install transformers
# !pip install huggingface_hub
# !pip install pandas

import os
import time
import pandas as pd
import torch
from datasets import Dataset
from transformers import pipeline
from huggingface_hub import login
from constants import detections_file_path, inferences_folder

# Folders
unquantized_folder = "blip_yolo_llama_unquantized_templates"
quantized_folder = "blip_yolo_llama_quantized_templates"

df = pd.read_csv(detections_file_path)

templates = ["Based on the image caption(provided by BLIP model) as '{caption}' and detections(provided by Yolo) as {detections}, answer in a single word the question based on the image details as question: '{question}'\nAnswer: ",
             "Using the image caption '{caption}' and detected objects '{detections}', answer the following question with a single word: '{question}'.\nAnswer: ",
             "Caption: '{caption}'. Detected Objects: '{detections}'. What is the one-word answer to this question about the image: '{question}'?\nAnswer: ",
             "Given the description '{caption}' and identified elements '{detections}', provide a one-word response to this inquiry about the image: '{question}'.\nAnswer: ",
             "From the image caption '{caption}' and object detections '{detections}', find the answer to: '{question}'. Respond in just one word.\nAnswer: ",
             "Challenge: With the caption '{caption}' and objects detected as '{detections}', determine the single-word answer to the question: '{question}'.\nAnswer: ",
             "Answer in a single word for the question: {question} using image caption: {caption} and object detections: {detections}.\nAnswer: "]

# Sets the Hugging Face token for running LLama
token = os.environ.get("LLAMA_TOKEN", "Missing LLAMA_TOKEN in environment variables")
login(token=token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unquantLlamaPipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", return_full_text=False,
                            device = device)
quantLlamaPipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", return_full_text=False,
                          device = device, torch_dtype= torch.bfloat16)

df = df[:1000]


for indx, template in enumerate(templates):
    if indx == 0:
        continue
    print(f"Running for {indx+1} template")

    df["Prompt"] = template
    df["Prompt"] = df.apply(lambda row: row['Prompt'].replace('{question}', row['Question']).replace('{detections}', row['Generated Detections']).replace('{caption}', row['Generated Caption']), axis=1)
    dataset = Dataset.from_pandas(df)

    ### Unquantized LLama run
    start_time = time.time()
    output = unquantLlamaPipe(dataset["Prompt"], max_new_tokens=1)
    df.loc[:, "Model Output"] = output
    file_name = f"{inferences_folder}/{unquantized_folder}/Template_{indx}_1k_unquantized.csv"
    df.to_csv(file_name, index=False)
    print(f"Successfully saved to {file_name}")

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    ### Quantized LLama run
    start_time = time.time()
    output = quantLlamaPipe(dataset["Prompt"], max_new_tokens=1)
    df.loc[:, "Model Output"] = output
    file_name = f"{inferences_folder}/{quantized_folder}/Template_{indx}_1k_quantized.csv"
    df.to_csv(file_name, index=False)
    print(f"Successfully saved to {file_name}")

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
