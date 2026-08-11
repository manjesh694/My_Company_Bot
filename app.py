import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Paths & Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./sft_my_company_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MyCompanyBot on device: {DEVICE}...")

# 2. Load Base Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(DEVICE)

# 3. Load Fine-Tuned PEFT Adapter
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
    print("Fine-tuned PEFT adapter loaded successfully!")
except Exception as e:
    print(f"Could not load adapter ({e}). Using base model directly...")
    model = base_model

# 4. Response Function (Handles both dictionary & tuple history formats)
def respond(message, history):
    system_prompt = "System: You are an AI assistant for MyCompany.\n"
    full_prompt = system_prompt
    
    # Process conversation history
    if history:
        for item in history:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user":
                    full_prompt += f"User: {content}\n"
                elif role == "assistant":
                    full_prompt += f"Answer: {content}\n"
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                u_msg, b_msg = item
                full_prompt += f"User: {u_msg}\nAnswer: {b_msg}\n"
    
    full_prompt += f"User: {message}\nAnswer:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        output_tokens[0][input_len:], 
        skip_special_tokens=True
    ).strip()

    return response

# 5. Launch Gradio ChatGPT-Style Interface
demo = gr.ChatInterface(
    fn=respond,
    title="🏢 MyCompany AI Assistant",
    description="Ask questions about MyCompany products, services, team members, and policies.",
    examples=[
        "Who is Manjesh?",
        "What does MyCompany do?",
        "How can I contact support?"
        "how to long in for employee ID"
    ]
)

if __name__ == "__main__":
    print("\n=== Starting MyCompany Web UI ===")
    demo.launch(share=False)