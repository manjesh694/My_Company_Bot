import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Model (Using your fine-tuned local model)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading JSON Bot...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cpu"
)

# -------------------------------------------------------------
# CODE FROM YOUR BOOK (PAGES 95-97): Structured JSON Prompting
# -------------------------------------------------------------
messages = [
    {
        "role": "system",
        "content": (
            "You are an AI Email Classifier for MyCompany. "
            "Extract information from the email and respond ONLY with valid JSON.\n\n"
            "JSON Schema:\n"
            "{\n"
            '  "intent": "refund|complaint|question|praise",\n'
            '  "urgency": "low|medium|high",\n'
            '  "product_mentioned": "string or null",\n'
            '  "summary": "one sentence summary"\n'
            "}"
        )
    },
    {
        "role": "user",
        "content": "Email: Hello! I was billed twice for my AI software subscription today. I need a refund immediately!"
    }
]

# 2. Apply Chat Template & Encode
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# 3. Generate JSON Response
output_tokens = model.generate(
    **inputs, 
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    output_tokens[0][inputs["input_ids"].shape[1]:], 
    skip_special_tokens=True
)

print("\n--- 1. Raw AI JSON Output ---")
print(response)

# -------------------------------------------------------------
# 4. PARSE JSON IN PYTHON (So your computer program can read it!)
# -------------------------------------------------------------
try:
    data = json.loads(response.strip())
    print("\n--- 2. Parsed Python Object ---")
    print("Intent Detected  :", data.get("intent"))
    print("Urgency Level    :", data.get("urgency"))
    print("Product Mentioned:", data.get("product_mentioned"))
    print("Summary          :", data.get("summary"))
except Exception as e:
    print("\nCould not parse as JSON:", e)