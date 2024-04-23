"""This file has the setup required as a single time setup for running the scripts.
Version: 1.0.0
"""

# !pip uninstall -y torch torchvision torchaudio 
# huggingface transformers

#### CPU setup
# !pip install huggingface transformers torch torchvision

#### GPU setup
# !nvcc --version
!pip install torch==2.2.1+cu121 torchaudio==2.2.1+cu121 torchdata==0.7.1 torchsummary==1.5.1 torchtext==0.17.1 torchvision==0.17.1+cu121 tornado==6.3.3 tqdm==4.66.2 -f https://download.pytorch.org/whl/cu121/
