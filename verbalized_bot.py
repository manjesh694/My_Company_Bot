import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Model
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading Verbalized Sampling Bot...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cpu"
)

# -------------------------------------------------------------
# CODE FROM YOUR BOOK (PAGE 168): Verbalized Sampling Function
# -------------------------------------------------------------
def ask_with_verbalized_sampling(customer_question):
    # Prompt the AI to output 3 candidate options with probabilities
    prompt = (
        f"Customer Question: '{customer_question}'\n\n"
        f"Generate 3 different helpful response options for this customer. "
        f"Assign a probability score (summing to 1.0) to each option based on how polite and helpful it is.\n"
        f"Format:\n"
        f"1. [Response 1] (Probability: 0.50)\n"
        f"2. [Response 2] (Probability: 0.30)\n"
        f"3. [Response 3] (Probability: 0.20)\n"
    )

    messages = [
        {"role": "system", "content": "You are a creative customer support AI for MyCompany."},
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")

    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=250,
        temperature=0.7,         # Higher temperature for creative options
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test it!
print("\n--- AI Generating 3 Options with Probabilities ---")
result = ask_with_verbalized_sampling("My order #555 has not arrived yet!")
print(result)