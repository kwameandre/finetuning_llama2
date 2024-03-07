# command line args 
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

print("\nImporting packages\n")
    
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

print("\nLoading Model\n")
huggingface_hub.login(token = huggingface_api_key)
tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        cache_dir=os.path.join('/scratch', username),
        load_in_8bit=True if train_config.quantization else None,
        #token=huggingface_api_key,
        use_auth_token=True,
)

tokenizer.add_special_tokens(
    {
        "pad_token": "<PAD>",
    }
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    resume_download=True,
    quantization_config=BitsAndBytesConfig(
        float16_bits=8,  # Adjust as needed
        float32_bits=32,  # Adjust as needed
    ),
    device_map="auto" if train_config.quantization else None,
    cache_dir=os.path.join('/scratch', username),
    trust_remote_code=True,
    token=huggingface_api_key,
)
print("Model Loaded\n")

print("Loading dataset\n")
import argparse
from datasets import load_dataset

# Global variables
load = None

def main(load_data, wab):
    global load
    load = load_data.lower()
    
if load == "yes":
    #edit file path to your unique dataset
    dataset = load_dataset('csv', data_files='samsum-data/samsum-train.csv', split='train')
    valset = load_dataset('csv', data_files='samsum-data/samsum-validation.csv', split='train')
    testset = load_dataset('csv', data_files='samsum-data/samsum-test.csv', split='train')

else:
    # Load the dataset from a CSV file
    full_dataset = load_dataset('csv', data_files='combined_info.csv')

    # Get the number of examples in the dataset
    num_examples = len(full_dataset["train"])

    # Define the split ratios
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    # Calculate the number of examples for each split
    num_train_examples = int(num_examples * train_ratio)
    num_validation_examples = int(num_examples * validation_ratio)
    num_test_examples = int(num_examples * test_ratio)

    # Split the dataset
    splits = full_dataset["train"].train_test_split(
        test_size=num_test_examples,
        train_size=num_train_examples,
        shuffle=True
    )

    # Assign the splits to global variables
    dataset = splits["train"]
    valset = splits["test"]

    # If you want a separate test split, you can use the test split from the original split
    testset = full_dataset["train"].train_test_split(
        test_size=num_test_examples,
        train_size=num_train_examples
    )["test"]

# Execute main function
#main("yes")  # You can change "yes" to "no" if you want to load the data differently

# Edit the prompt to tell the model what to do including the variables from prompt.format
prompt = (
    f"Generate labels that best fit the following text with respect topic of :\n{{text}}\n---\nLabel:{{label}}\nLabels:\n"
)

#prompt for testing
test_prompt = (
    f"Tell me which labels best fit the following text:\n{{text}}\n---\nLabel:\n"
)

#edit the variables in prompt.format to match your data: essentially what you what the model to read
def apply_prompt_template(sample):
    return {
        "text": prompt.format(
            text = sample["text"],
            label = sample["label"],
        )
    }

#Only include what you want the model to see during testing
def apply_prompt_template_TEST(sample):
    return {
        "text": test_prompt.format(
            text = sample["text"],
        )
    }

# Data processing
data = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
val = valset.map(apply_prompt_template, remove_columns=list(valset.features))
test = testset.map(apply_prompt_template_TEST, remove_columns=list(testset.features))

# Tokenization
train_dataset = data.map(
    lambda sample: tokenizer(sample["text"]),
    batched=True,
    remove_columns=list(data.features),
).map(Concatenator(), batched=True)
val_dataset = val.map(
    lambda sample: tokenizer(sample["text"]),
    batched=True,
    remove_columns=list(val.features),
).map(Concatenator(), batched=True)
test_dataset = test.map(
    lambda sample: tokenizer(sample["text"]),
    batched=True,
    remove_columns=list(test.features),
).map(Concatenator(), batched=True)

print("Dataset Loaded\n")

#Edit eval_prompt to match your data
print("Testing model before training\n")
eval_prompt = """
Summarize this dialog:
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
---
Summary:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

print("Reducing parameters\n")
#reduces the parameters needed to train
model.train()
def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, #lower if run into issues
        lora_alpha=32,
        lora_dropout=0.05,
        bias= "none",
        target_modules = ["q_proj", "v_proj"]
    )
    
    kwargs = {
        'use_peft': True, 
        'peft_method': 'lora', 
        'quantization': True, 
        'use_fp16': True, 
        #edit when changing the model
        'model_name': os.path.join('/scratch', username, 'models--meta-llama--Llama-2-70b-hf/snapshots/90052941a64de02075ca800b09fcea1bdaacb939'), 
        'output_dir': os.path.join('/scratch', username)
    }
    
    update_config((train_config, fsdp_config), **kwargs)
    
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
print("Setting up LORA\n")
model, lora_config = create_peft_config(model)

torch.cuda.empty_cache()
from transformers import TrainerCallback
from contextlib import nullcontext
enable_profiler = False
#set up the configurations for training
config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 10,
    'gradient_accumulation_steps': 4,
    'per_device_train_batch_size': 2,
    'gradient_checkpointing': False,
}

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
    
    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler
            
        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()

print("LORA Completed\n")

#create a weights and biases account and input the api key in below
#change the names of the project
print("Creating weights and biases project\n")
wandb.init(project="tmp1", name="testingPyScript")
print("Project created\n")

from transformers.integrations import WandbCallback

torch.cuda.empty_cache()
# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16 = True,
    #logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    evaluation_strategy="steps",
    logging_steps=5,
    eval_steps=5,
    save_strategy="steps",
    optim="adamw_torch_fused",
    auto_find_batch_size = True, 
    max_steps=total_steps if enable_profiler else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'},
    remove_unused_columns=False,
    do_eval=True,
    save_steps=10,
    save_total_limit=5,
    load_best_model_at_end=True,
    push_to_hub=False,  # Set this to False to avoid conflicts with W&B logging
    logging_dir=wandb.run.dir,
    report_to="wandb",
)
with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        #TODO early stopping pat. 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), profiler_callback, WandbCallback()] if enable_profiler else [WandbCallback()],
    )
# Start training
print("Starting training\n")
trainer.train()
print("Training Complete\n")

print("Saving model\n")
#saves the model
model.save_pretrained(output_dir)

#Saves model configurations
model.config.to_json_file(os.path.join(output_dir, "config.json"))

#saves PEFT config
peft_config = model.peft_config
json_file_path = os.path.join(output_dir, "peft_config.json")

# Custom serialization function for LoraConfig objects
def lora_config_serializer(obj):
    if isinstance(obj, LoraConfig):
        # Return a dictionary representation of the LoraConfig object
        return obj.__dict__
    raise TypeError("Type not serializable")

# Write the dictionary to the JSON file using the custom serializer
with open(json_file_path, "w") as json_file:
    json.dump(peft_config, json_file, default=lora_config_serializer, indent=4)

print("Model saved\n")
  
#load model to huggingface hub
print("Loading model to huggingface hub")
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from pathlib import Path
#https://huggingface.co/docs/hub/repositories-getting-started
api = HfApi()

repo_id = "andrk9/testingPyScript"

try:
    create_repo(repo_id)
except RepositoryExistsError:
    # If the repository already exists, catch the exception and update the repository
    print(f"Repository '{repo_id}' already exists. Updating...")

api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--loadData', choices=['yes', 'no'], required=True, help='Whether to input test splits (yes/no)')
    #parser.add_argument('--wab', choices=['yes', 'no'], required=True, help='Load data to weights and biases')
    args = parser.parse_args()

    main(args.loadData)