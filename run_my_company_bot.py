import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup paths and device
MODEL_PATH = "./sft_my_company_model"  # Path to your fine-tuned model checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {DEVICE}")

# 2. Load Tokenizer & Model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# 3. Interactive Bot Loop
def ask_bot(prompt: str):
    # Formulate system prompt
    full_prompt = f"System: You are an AI assistant for MyCompany.\nUser: {prompt}\nAnswer:"
    
    # Encode prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    # Generate Response
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=280,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    # Decode Output (stripping input prompt tokens)
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        output_tokens[0][input_len:], 
        skip_special_tokens=True
    )
    return response.strip()

# 4. Run Interactive Chat
if __name__ == "__main__":
    print("\n=== MyCompanyBot is Ready! Type 'exit' to quit. ===")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        bot_response = ask_bot(user_input)
        print(f"\n--- MyCompanyBot Answer ---\n{bot_response}")