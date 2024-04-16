import pandas as pd
import torch
from PIL import Image

captions_file = "../data/10k_blip_captions.csv"
mapped_question_answers_file = "../data/10k_mapped_question_answers.csv"

captions_data = pd.read_csv(captions_file)
qa_data = pd.read_csv(mapped_question_answers_file)


qa_data.drop(columns=["Image file"], inplace=True)
data = pd.concat([qa_data, captions_data], axis=1)

# Create a lookup dictionary
lookup = data.set_index('Image ID')[['Image file', 'Generated Caption']].T.to_dict('list')

results = data.to_dict('records')

# Write data to a CSV file
df = data


print("Running YOLO")
# Generate detections using YOLO Pipeline
# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)

    # Inference
    results = yolo_model(img)

    # Plot and save results
    results.render()

    # Print results
    return results.pandas().xyxy[0].to_dict(orient="records")


# Create a new column for detections
df['Generated Detections'] = None


for index, row in df.iterrows():
    image_path = row['Image file']
    detections = detect_objects(image_path)
    df.at[index, 'Generated Detections'] = detections


print("Storing Detections")
# Write data to a CSV file
df.to_csv('10k_yolo_detections.csv', index=False)