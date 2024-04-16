# %%
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

# %%
from huggingface_hub import login
token = "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"

login(token=token) ## This is bound to fail, add your token from chat and run

# %%
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load LLaMA model for text-based question answering
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


# %%
IMAGE_DIR = "../data/validation/images"

# %%
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# %%
def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)

    # Inference
    results = yolo_model(img)

    # Plot and save results
    results.render()

    # Print results
    return results.pandas().xyxy[0].to_dict(orient="records")  # print pandas dataframe of detections

# %%
import os


count = 2
# Loop through each file in the directory
for filename in os.listdir(IMAGE_DIR):
    if(count==0):
        break
    count -= 1
    file_path = os.path.join(IMAGE_DIR, filename)

    # Check if it's a file and not a directory
    if os.path.isfile(file_path):
        print(f'Processing file: {filename}')
        detections = detect_objects(file_path)
        print("Completed detections")
        image = Image.open(file_path)
        image = blip_processor(images=image, return_tensors="pt").pixel_values
        print("Completed processing for blip")
        # Generate caption for the image
        generated_ids = blip_model.generate(image, max_length=50, num_beams=5, return_dict_in_generate=True).sequences
        caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        print("Completed image captioning")
        question = "Describe the image in detail with the given caption and object detections."
        prompt = f"Image Caption : {caption}\n\n Image object detections : {str(detections)}\n\n Question: {question}\nAnswer:"
        print("Completed prompt Requesting Llama")
        inputs = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        print("Generating Llama")
        outputs = llama_model.generate(**inputs, num_beams=2,  max_new_tokens=100)
        print("Decoding Llama output")
        answer = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Output: ")
        print(answer)


# # JSON file to save all detections
# json_filename = os.path.join(IMAGE_DIR, 'all_detections.json')
#
# # Write all detection results to a single JSON file
# with open(json_filename, 'w') as json_file:
#     json.dump(all_detections, json_file, indent=4)
#
# print(f'All results saved to {json_filename}')




