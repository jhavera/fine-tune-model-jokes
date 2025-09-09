import os

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# --- This code block is for retrieving Hugging Face token when running in Google Colab --#
# from google.colab import userdata # Import userdata

# Use Hugging Face token from Colab Secrets
# HF_TOKEN = userdata.get("HF_TOKEN") # Load HF_TOKEN from Colab secrets
# if not HF_TOKEN:
#     raise RuntimeError(
#         "Hugging Face token not found. Set HF_TOKEN (or HUGGINGFACE_TOKEN) in your environment.\n"
#         "Make sure you've accepted the model's license on its model card page."
#     )

# --- This code block is for retrieving token when running on local host machine --#
HF_TOKEN_ENV = "HF_TOKEN"

def get_hf_token(required: bool = True) -> str:
    """
    Retrieve the Hugging Face token from the environment.

    Args:
        required: If True, raises an error when the token is missing.

    Returns:
        The token string (empty string if not required and unset).

    Raises:
        ValueError: If required is True and the token is not set.
    """
    token = os.getenv(HF_TOKEN_ENV)
    if required and not token:
        raise ValueError(
            f"Missing Hugging Face token. Please set the {HF_TOKEN_ENV} environment variable."
        )
    return token or ""


# Backwards-compatible module-level token for existing code paths.
HF_TOKEN = get_hf_token()

# ---- Load dataset from CSV and process it ----
def load_csv_dataset(file_path):
    dataset = load_dataset("csv", data_files=file_path)
    # The CSV has a 'joke' column, we will use it as 'text'
    def format_joke(example):
        example['text'] = example['joke']
        return example
    return dataset.map(format_joke)

# Use the correct file path
# dataset = load_csv_dataset("/content/data/dad_jokes.csv")
dataset = load_csv_dataset("data/dad_jokes.csv")
train_ds = dataset["train"]
eval_ds = dataset["train"].select(range(20)) # Use a small part for evaluation

# ---- Optional 4-bit quantization: enable only if CUDA is available ----
use_cuda = torch.cuda.is_available()
compute_dtype = getattr(torch, "float16")
bnb_config = None
device_map = "auto" if use_cuda else None # Set device_map to None if CUDA is not available

if use_cuda:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

# ---- Model and Tokenizer setup ----
model_id = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map=device_map,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---- LoRA and Trainer setup ----
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model = prepare_model_for_kbit_training(model)

# Move model to CUDA if available before initializing Trainer
# This explicit move might not be necessary with device_map=None when on CPU,
# but keeping it doesn't hurt and helps when CUDA is available.
if use_cuda:
    model = model.to("cuda")

output_dir = "./results"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1, # Reduced batch size
    gradient_accumulation_steps=4, # Increased accumulation steps to compensate
    gradient_checkpointing=True,
    optim="paged_adamw_32bit" if use_cuda else "adamw_torch",
    save_steps=0,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
)

# ---- Tokenize datasets and use standard Trainer ----
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
def _tokenize(batch):
    tokenized_batch = {}
    try:
        tokenized_batch = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
    except Exception as e:
        print(f"Error tokenizing batch: {batch}")
        print(f"Error message: {e}")
        # Return empty lists for the keys that would normally be in the tokenized batch
        tokenized_batch = {key: [] for key in ["input_ids", "attention_mask"]}
    return tokenized_batch


tokenized_train = train_ds.map(
    _tokenize,
    batched=True,
    remove_columns=train_ds.column_names,
)
tokenized_eval = eval_ds.map(
    _tokenize,
    batched=True,
    remove_columns=eval_ds.column_names,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

print("Fine-tuning complete!")
