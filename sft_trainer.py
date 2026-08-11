import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 1. Load Model & Tokenizer
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Model and Tokenizer for SFT Training...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cpu"
)

# 2. Dataset formatted as Chat Messages (Page 197)
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are MyCompanyBot, an official support AI."},
            {"role": "user", "content": "What is the support email?"},
            {"role": "assistant", "content": "Our support email is support@mycompany.com."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are MyCompanyBot, an official support AI."},
            {"role": "user", "content": "Where is the main office located?"},
            {"role": "assistant", "content": "Our main office is located in New York."}
        ]
    }
]

dataset = Dataset.from_list(training_data)

# 3. SFT Configuration (Updated max_length)
sft_config = SFTConfig(
    output_dir="./sft_checkpoints",
    max_length=256,                      # <-- Updated from max_seq_length to max_length
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=1,
    use_cpu=True
)

# 4. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)

# 5. Start Fine-Tuning
print("\n--- Starting SFT Training ---")
trainer.train()

# 6. Save Fine-Tuned Model
print("\nSaving SFT trained model to './sft_my_company_model'...")
model.save_pretrained("./sft_my_company_model")
tokenizer.save_pretrained("./sft_my_company_model")

print("\n✅ SFT Training Complete! Model saved to './sft_my_company_model'.")