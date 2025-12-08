import torch
print(torch.cuda.is_available())
print(torch.__version__)



from huggingface_hub import constants

import os
print("HF_HOME:", constants.HF_HOME)
print("TF_CACHE:", os.getenv("TRANSFORMERS_CACHE"))
