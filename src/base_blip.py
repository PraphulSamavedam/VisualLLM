"""This file generates the performance of the BLIP for the question answering
based on the pre-trained checkpoint.
"""

import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from constants import mapped_qa_file_path, results_folder

# Setup
data = pd.read_csv(mapped_qa_file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# Process each question and image for answer.
for indx in range(data.shape[0]):
    question = data.loc[indx, "Question"]
    image_file = data.loc[indx, "Image file"]
    raw_image = Image.open(image_file).convert('RGB')
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    data.loc[indx, "Generate Answer"] = "Answer: " + processor.decode(out[0], skip_special_tokens=True)

# Store the baseline blip results into
data.to_csv(f"{results_folder}/base_blip.csv", index = False)
