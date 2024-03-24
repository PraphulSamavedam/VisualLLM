#!/bin/bash

# Setup the data once
if [ ! -d "data" ]; then
    mkdir "data"
    echo "data folder created successfully"
fi
#### Setup the train, test and validation folders
sub_folders=("train" "test" "validation")
for subfolder in "${subfolders[@]}"; do
  if [! -d "data/train"]; then 
     mkdir -p "data/$subfolder"
     echo "Created the data/$subfolder folder successfully"
  else
     echo "data/$subfolder folder already exists"
  fi
done

### Setup the images dataset if they don't exist
#### Train data setup
if [ ! -d "data/train/images" ]; then
   echo "Training images data doesn't exist. Fetching it and setting it up"
   cd "data/train"
   curl -O "http://images.cocodataset.org/zips/train2014.zip"
   # mv "train2014.zip" "data/train/train2014.zip"
   # cd "data/train"
   unzip "train2014.zip"
   mv "train2014" "images"
   rm "train2014.zip"
   cd "../../"
else
   echo "Training images data exists"
fi

#### Test data setup
if [ ! -d "data/test/images" ]; then
   echo "Testing images data doesn't exist. Fetching it and setting it up"
   cd "data/test"
   curl -O "http://images.cocodataset.org/zips/test2015.zip"
   unzip "test2015.zip"
   mv "test2015" "images"
   rm "test2015.zip"
   cd "../../"
else
   echo "Testing images data exists"
fi

#### Validation data setup
if [ ! -d "data/validation/images" ]; then
   echo "Validation images data doesn't exist. Fetching it and setting it up"
   cd "data/validation"
   curl -O "http://images.cocodataset.org/zips/val2014.zip"
   unzip "val2014.zip"
   mv "val2014" "images"
   rm "val2014.zip"
   cd "../../"
else
   echo "Validation images data exists"
fi

### Setup the annotators and questions as well
#### Sets up the training annotations and questions
if [ ! -f "data/train/annotations.json" ]; then
   echo "Missing training annotations"
   cd "data/train"
   curl -O "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
   unzip "v2_Annotations_Train_mscoco.zip"
   mv "v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json" "annotations.json"
   rm "v2_Annotations_Train_mscoco.zip"
   rmdir "v2_Annotations_Train_mscoco"
   cd "../../"
else
   echo "Training annotations exists"
fi

if [ ! -f "data/train/questions.json" ]; then
   echo "Missing training questions"
   cd "data/train"
   curl -O "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
   unzip "v2_Questions_Train_mscoco.zip"
   mv "v2_OpenEnded_mscoco_train2014_questions.json" "questions.json"
   rm "v2_Questions_Train_mscoco.zip"
   cd "../../"
else
   echo "Training questions exists"
fi

#### Settinng up testing questions is little difficult as it has questions of test and validation # Manually set this up
if [ ! -f "data/test/questions.json" ]; then
   echo "Missing testing questions. Proceed with manual setup it cannot be done."
   exit -100
else
   echo "Testing questions exists"
fi
### Below code doesn't work
#if [ ! -f "data/test/questions.json" ]; then
#   echo "Missing testing questions"
#   cd "data/test"
#   curl -O "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
#   unzip "v2_Questions_Test_mscoco.zip"
#   mv "v2_OpenEnded_mscoco_test2015_questions.json" "questions.json"
#   rm "v2_Questions_Test_mscoco.zip"
#   cd "../../"
#else
#   echo "Testing quesions exist"
#fi

#### Sets up the validation annotations and questions
if [ ! -f "data/validation/annotations.json" ]; then
   echo "Missing validation annotations"
   cd "data/validation"
   curl -O "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
   unzip "v2_Annotations_Val_mscoco.zip"
   mv "v2_mscoco_val2014_annotations.json" "annotations.json"
   rm "v2_Annotations_Val_mscoco.zip"
   cd "../../"
else
   echo "Validation annotations exists"
fi

if [ ! -f "data/validation/questions.json" ]; then
   echo "Missing validation questions"
   cd "data/validation"
   curl -O "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
   unzip "v2_Questions_Val_mscoco.zip"
   mv "v2_OpenEnded_mscoco_val2014_questions.json" "questions.json"
   rm "v2_Questions_Val_mscoco.zip"
   cd "../../"
else
   echo "Validation questions exists"
fi


