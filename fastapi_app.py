import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# 1. Configuration & Paths
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./sft_my_company_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing FastAPI backend on device: {DEVICE}...")

# 2. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(DEVICE)

try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(DEVICE)
    print("Fine-tuned PEFT adapter loaded successfully!")
except Exception as e:
    print(f"Adapter load failed ({e}). Fallback to base model...")
    model = base_model

# 3. FastAPI App Setup
app = FastAPI(title="MyCompany AI Assistant API")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# 4. API Endpoints
@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """REST API endpoint for sending prompts to MyCompanyBot."""
    prompt = f"System: You are an AI assistant for MyCompany.\nUser: {request.message}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

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
    bot_response = tokenizer.decode(
        output_tokens[0][input_len:], 
        skip_special_tokens=True
    ).strip()

    return ChatResponse(response=bot_response)

# 5. Embedded HTML Frontend Interface
@app.get("/", response_class=HTMLResponse)
def serve_web_ui():
    """Serves the Web Chat Interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyCompany AI Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f6f8; display: flex; justify-content: center; padding: 20px; }
            .chat-container { width: 100%; max-width: 600px; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); padding: 20px; }
            .chat-box { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
            .user-msg { color: #007bff; text-align: right; margin: 8px 0; font-weight: bold; }
            .bot-msg { color: #28a745; text-align: left; margin: 8px 0; font-weight: bold; }
            .input-group { display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h2>🏢 MyCompany AI Assistant (FastAPI)</h2>
            <div id="chatBox" class="chat-box"></div>
            <div class="input-group">
                <input type="text" id="userInput" placeholder="Ask about MyCompany..." onkeypress="if(event.key==='Enter') sendMessage()" />
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('userInput');
                const chatBox = document.getElementById('chatBox');
                const msg = input.value.trim();
                if (!msg) return;

                chatBox.innerHTML += `<div class="user-msg">You: ${msg}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: msg })
                    });
                    const data = await res.json();
                    chatBox.innerHTML += `<div class="bot-msg">Bot: ${data.response}</div>`;
                } catch (e) {
                    chatBox.innerHTML += `<div class="bot-msg" style="color:red;">Error connecting to API.</div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)