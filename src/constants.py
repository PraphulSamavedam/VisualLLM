### This file has constants used in the program
data_folder = "../data"
results_folder = "../results"
inferences_folder = "../inferences"
validation_folder = f"{data_folder}/validation"
train_folder = f"{data_folder}/train"
test_folder = f"{data_folder}/test"

training_images_folder = f"{train_folder}/images"
validation_images_folder = f"{validation_folder}/images"
test_images_folder = f"{test_folder}/images"

mapped_qa_file_path = f"{data_folder}/mapped_question_answers.csv"
sampled_qa_file_path = f"{data_folder}/10k_mapped_question_answers.csv"
captions_file_path = f"{data_folder}/10k_blip_captions.csv"
detections_file_path = f"{data_folder}/10k_yolo_detections.csv"

annotations_file_path = f"{validation_folder}/annotations.json"
questions_file_path = f"{validation_folder}/questions.json"

samples = None