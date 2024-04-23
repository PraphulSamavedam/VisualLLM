import torch 
from transformers import pipeline # Use a pipeline as a high-level helper
from constants import sampled_qa_file_path, captions_file_path
import pandas as pd

# Obtaining the image files list
print("Obtaining the image files list")
df = pd.read_csv(sampled_qa_file_path)

image_files = df["Image file"].to_list()

print("Generating captions using Blip Pipeline")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large",
                     device = device)
generated_captions = blip_pipe(image_files)

### Get the generated text along with image_id, image_file
captions = [[image_files[indx], x[0]['generated_text']] for indx, x in enumerate(generated_captions)]
df.loc[:, "Generated Caption"] = captions

# Store the blip captions of the image files
df[["Image file", "Generated Caption"]].to_csv(captions_file_path)
