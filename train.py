import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)

# =============================================================
# 1. LOAD MODEL & TOKENIZER
# =============================================================
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Model and Tokenizer for Training...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cpu"
)

# =============================================================
# 2. PREPARE TRAINING DATASET
# =============================================================
training_data = [
    {"text": "Company Name: MyCompany. Purpose: Providing AI tools to clients."},
    {"text": "MyCompanyBot Support Email: support@mycompany.com."},
    {"text": "MyCompany refund policy: Refunds are processed within 5 business days."}
]

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list(training_data)

# Tokenize function
def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# =============================================================
# 3. TRAINING ARGUMENTS (TEXTBOOK OPTIMIZER & MIXED PRECISION)
# =============================================================
training_args = TrainingArguments(
    output_dir="./checkpoints",         # Save folder for checkpoints
    
    # --- Optimizer & Scheduler Settings (From Textbook Page 68) ---
    optim="adamw_torch",                 # AdamW Optimizer
    learning_rate=2e-5,                  # Peak Learning Rate
    lr_scheduler_type="cosine",          # Cosine decay schedule
    warmup_steps=1,                      # Warmup steps
    weight_decay=0.01,                   # Decoupled weight decay
    num_train_epochs=3,                  # Total training epochs
    per_device_train_batch_size=1,       # Batch size
    logging_steps=1,                     # Log every step
    save_strategy="epoch",               # Save model every epoch
    
    # --- Hardware & Precision Settings (From Textbook Page 71) ---
    use_cpu=True,                        # Set True for CPU / Set False if using GPU
    # fp16=True,                         # <-- Uncomment if using older GPU (T4, V100)
    # bf16=True,                         # <-- Uncomment if using newer GPU (RTX 30xx/40xx)
)

# =============================================================
# 4. INITIALIZE TRAINER & START TRAINING
# =============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\n--- Starting Training ---")
trainer.train()

# =============================================================
# 5. SAVE THE FINE-TUNED MODEL
# =============================================================
print("\nSaving trained model to './my_company_model'...")
model.save_pretrained("./my_company_model")
tokenizer.save_pretrained("./my_company_model")

print("\n✅ Training complete! Model saved to './my_company_model'.")