import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading CoT & RAG Bot...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cpu"
)

# -------------------------------------------------------------
# 1. RAG (Retrieval-Augmented Generation): Pass Document Context
# -------------------------------------------------------------
company_document = "MyCompany office is in New York. Office hours are 9:00 AM to 5:00 PM EST."

messages = [
    {
        "role": "system",
        "content": "Answer the question based ONLY on the provided context."
    },
    {
        "role": "user", 
        # -------------------------------------------------------------
        # 2. Zero-Shot CoT (Page 98): "Let's think step by step."
        # -------------------------------------------------------------
        "content": f"Context: {company_document}\n\nQuestion: How many hours is the office open each day?\n\nLet's think step by step."
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

output_tokens = model.generate(
    **inputs, 
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    output_tokens[0][inputs["input_ids"].shape[1]:], 
    skip_special_tokens=True
)

print("\n--- CoT Reasoning & Answer ---")
print(response)