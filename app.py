import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Configuration & Paths
MODEL_PATH = "./sft_my_company_model"  # Path to fine-tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MyCompanyBot on device: {DEVICE}...")

# 2. Load Model and Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    print("Fine-tuned model loaded successfully!")
except Exception as e:
    print(f"Loading fine-tuned model failed ({e}). Loading base model...")
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(DEVICE)

# 3. Response Generation Function
def respond(message, history):
    # Construct chat context
    system_prompt = "System: You are a helpful AI assistant for MyCompany.\n"
    
    # Format chat history into prompt
    full_prompt = system_prompt
    for user_msg, bot_msg in history:
        full_prompt += f"User: {user_msg}\nAnswer: {bot_msg}\n"
    
    full_prompt += f"User: {message}\nAnswer:"

    # Encode prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    # Generate response tokens
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=250,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    # Extract new text
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        output_tokens[0][input_len:], 
        skip_special_tokens=True
    ).strip()

    return response

# 4. Launch Gradio ChatGPT-Style Web UI
demo = gr.ChatInterface(
    fn=respond,
    title="🏢 MyCompany AI Assistant",
    description="Ask questions about MyCompany products, services, team members, and policies.",
    examples=[
        "Who is Manjesh?",
        "What does MyCompany do?",
        "How can I contact support?"
    ],
    theme="soft"
)

if __name__ == "__main__":
    print("\n=== Starting MyCompany Web UI ===")
    demo.launch(share=False)