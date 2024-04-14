import torch 
from transformers import pipeline  # Use a pipeline as a high-level helper
import os # For all os level functions
import json # For parsing the json files
import csv
from constants import validation_folder, validation_images_folder, samples
import pandas
import pyarrow as pa
import pyarrow.parquet as pq

# Define CSV file path
qa_path = 'mapped_question_answers.csv'
parquet_file_path = "data.parquet"


data = []
with open(qa_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

table = pq.read_table('data.parquet')
print(table)
