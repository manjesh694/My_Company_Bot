import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer
)

# 1. Model Name
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------------
# CODE FROM YOUR BOOK (PAGE 83): 4-Bit Quantization Config
# -------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

print("Loading Model in 4-Bit (QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for 4-bit training
model = prepare_model_for_kbit_training(model)

# -------------------------------------------------------------
# CODE FROM YOUR BOOK (PAGE 83): LoRA Configuration
# -------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)

# Print percentage of trainable parameters (~0.12%)
model.print_trainable_parameters()

# -------------------------------------------------------------
# PREPARE DATASET & TRAINER
# -------------------------------------------------------------
training_data = [
    {"text": "Company Name: MyCompany. Role: Customer Support AI."},
    {"text": "Support Email: support@mycompany.com."},
]
dataset = Dataset.from_list(training_data)

def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./qlora_checkpoints",
    optim="paged_adamw_8bit",           # 8-bit Paged AdamW Optimizer
    learning_rate=2e-4,                 # Higher LR for LoRA (2e-4)
    lr_scheduler_type="cosine",
    warmup_steps=5,
    logging_steps=1,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    fp16=True,                          # QLoRA uses FP16/BF16 on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\n--- Starting QLoRA Training ---")
trainer.train()

# Save QLoRA Adapters
print("\nSaving QLoRA Model Adapters...")
model.save_pretrained("./qlora_my_company_model")
tokenizer.save_pretrained("./qlora_my_company_model")

print("✅ QLoRA Training Complete! Adapters saved to './qlora_my_company_model'.")