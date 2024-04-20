import pandas as pd
import torch
from PIL import Image
from constants import mapped_qa_file_path, captions_file_path, detections_file_path

# Obtain the captions and questions, answers for the samples considered
captions_data = pd.read_csv(captions_file_path)
qa_data = pd.read_csv(mapped_qa_file_path)

# Created the concatenated data for analysis
qa_data.drop(columns=["Image file"], inplace=True)
df = pd.concat([qa_data, captions_data], axis=1)

# Create a lookup dictionary
lookup = df.set_index('Image ID')[['Image file', 'Generated Caption']].T.to_dict('list')
results = df.to_dict('records')

print("Running YOLO")
# Generate detections using YOLO Pipeline
# Load the YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def detect_objects(img_path:str):
    """
    This function opens the image provided as path and returns the 
    bounding boxes for the objects detected.
    """
    # Load image
    img = Image.open(img_path)

    # Inference
    inf_results = yolo_model(img)

    # Plot and save results
    inf_results.render()

    # Return results
    return inf_results.pandas().xyxy[0].to_dict(orient="records")


# Create a new column for detections
df['Generated Detections'] = None


for index, row in df.iterrows():
    image_path = row['Image file']
    detections = detect_objects(image_path)
    df.at[index, 'Generated Detections'] = detections


# Write data to a CSV file
print("Storing Detections")
df.to_csv(detections_file_path, index=False)
