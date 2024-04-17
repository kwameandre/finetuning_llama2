# Set cache directory and load Huggingface api key
import os

username = os.getenv('USER')
directory_path = os.path.join('/scratch', username)
output_dir = os.path.join('/scratch', username ,'llama-output')

# Set Huggingface cache directory to be on scratch drive
if os.path.exists(directory_path):
    hf_cache_dir = os.path.join(directory_path, 'hf_cache')
    if not os.path.exists(hf_cache_dir):
        os.mkdir(hf_cache_dir)
    print(f"Okay, using {hf_cache_dir} for huggingface cache. Models will be stored there.")
    assert os.path.exists(hf_cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = f'/scratch/{username}/hf_cache/'
else:
    error_message = f"Are you sure you entered your username correctly? I couldn't find a directory {directory_path}."
    raise FileNotFoundError(error_message)

# Load Huggingface api key
api_key_loc = os.path.join('/home', username, '.apikeys', 'huggingface_api_key.txt')

if os.path.exists(api_key_loc):
    print('Huggingface API key loaded.')
    with open(api_key_loc, 'r') as api_key_file:
        huggingface_api_key = api_key_file.read().strip()  # Read and store the contents
else:
    error_message = f'Huggingface API key not found. You need to get an HF API key from the HF website and store it at {api_key_loc}.\n' \
                    'The API key will let you download models from Huggingface.'
    raise FileNotFoundError(error_message)
    
import torch
import wandb
from transformers.integrations import WandbCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, DataCollatorWithPadding, default_data_collator, Trainer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from configs import fsdp_config, train_config
from peft import get_peft_model, prepare_model_for_int8_training, PeftModelForCausalLM, LoraConfig, TaskType, prepare_model_for_int8_training, PeftModel
from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
)
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from datasets import Dataset, load_dataset
from pathlib import Path
import sys
import csv
import random
import json
from configs.datasets import samsum_dataset, alpaca_dataset, grammar_dataset
from ft_datasets.utils import Concatenator
import huggingface_hub
from huggingface_hub import notebook_login, Repository, HfApi, create_repo, delete_repo

tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        cache_dir=os.path.join('/scratch', username),
        load_in_8bit=True if train_config.quantization else None,
        use_auth_token=True,
)

tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)

plain_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    resume_download=True,
    quantization_config=BitsAndBytesConfig(
        float16_bits=8,  # Adjust as needed
        float32_bits=32,  # Adjust as needed
    ),
    device_map="auto" if train_config.quantization else None,
    cache_dir=os.path.join('/scratch', username),
    trust_remote_code=True,
    use_auth_token=True,
)

finetuned_model = PeftModelForCausalLM.from_pretrained(plain_model, output_dir)

#loop following code for continuous convo

#prompt for plain model
eval_prompt = " "

while(eval_prompt.lower() != "q" or eval_prompt.lower() != "quit"):
    eval_prompt = input("Enter a sample prompt, enter q or quit to stop: ")
    # Process the user input
    # For example, you can print the input or perform further processing as needed

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    print("Plain model output\n")

    plain_model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

    #loads finetuned model

    printf("finetuned model output\n")

    finetuned_model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))