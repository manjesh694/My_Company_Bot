import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configuration (Using open-access model)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./sft_my_company_model"

# Detect CUDA GPU vs CPU
use_cuda = torch.cuda.is_available()
print(f"CUDA Available: {use_cuda}")

# 2. Tokenizer & Model Loading
print(f"Loading model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

if use_cuda:
    # Use 4-bit Quantization on GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
else:
    # CPU fallback
    print("GPU not detected. Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

# 3. LoRA (Fine-Tuning) Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    max_steps=50,  # Number of training steps
    fp16=use_cuda,
    bf16=False,
    save_strategy="steps",
    save_steps=25,
    optim="adamw_torch"
)

# 5. Dataset Loading
try:
    dataset = load_dataset("text", data_files={"train": "MyCompanyBot.txt"})
except Exception as e:
    print("MyCompanyBot.txt not found. Using sample dataset...")
    from datasets import Dataset
    sample_data = {"text": [
        "What is MyCompany? MyCompany provides AI solutions.",
        "Where is MyCompany located? Our headquarters are in New York.",
        "How to contact support? Email support@mycompany.com."
    ]}
    dataset = {"train": Dataset.from_dict(sample_data)}

# 6. Initialize Trainer (Fixed for newer TRL versions)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
)

# 7. Start Fine-Tuning
print("--- Starting Fine-Tuning ---")
trainer.train()

# 8. Save Fine-Tuned Model
print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("--- Training Complete Successfully! ---")