import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login

# Setup
validation_folder = "../data/validation"
annotations_file_path = f"{validation_folder}/annotations.json"
questions_file_path = f"{validation_folder}/questions.json"
login(token=token) ## This is bound to fail, add your token from chat and run

data_file = "../data/10k_mapped_question_answers.csv"
data = pd.read_csv(data_file)



import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")


for indx in range(data.shape[0]):
    question = data.loc[indx, "Question"]
    image_file = data.loc[indx, "Image file"]
    raw_image = Image.open(image_file).convert('RGB')
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    data.loc[indx, "Generate Answer"] = "Answer: " + processor.decode(out[0], skip_special_tokens=True)
    # print(f'{data.loc[indx, "Generate Answer"]}')

data.to_csv("base_blip.csv", index = False)
