# 🏢 MyCompany AI Assistant (`My_Company_Bot`)

An end-to-end Enterprise Domain-Adapted Large Language Model (LLM) assistant built with **Supervised Fine-Tuning (SFT)**, **QLoRA (4-bit Quantization)**, **Vector Memory**, **Gradio Web UI**, **FastAPI REST API**, and **Docker Deployment**.

---

## 🛠️ Tech Stack & Technologies
* **Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Frameworks:** PyTorch, Hugging Face Transformers, TRL (`SFTTrainer`), PEFT (QLoRA)
* **Web UI:** Gradio, HTML5/CSS3
* **Backend REST API:** FastAPI, Uvicorn, Pydantic
* **Vector Memory:** Custom Cosine Similarity Vector Store with Temporal Decay & Recency Weighting
* **Deployment:** Docker, Docker Compose, Git/GitHub

---

## 📁 Repository Structure
```text
My_Company_Bot/
│
├── fsdp_qlora_trainer.py   # QLoRA + SFT Fine-Tuning engine on domain dataset
├── run_my_company_bot.py   # Interactive CLI terminal chat interface
├── app.py                  # Gradio ChatGPT-style Web UI
├── fastapi_app.py          # FastAPI REST API Backend with embedded Web UI
├── memory_store.py         # Vector Memory Store with temporal decay & importance scoring
├── memory_agent.py         # MemGPT-style self-directed memory agent loop
├── MyCompanyBot.txt        # Enterprise domain dataset
├── Dockerfile              # Production Docker container image setup
├── docker-compose.yml      # Docker container orchestration file
├── requirements.txt        # Python dependencies list
└── README.md               # Project documentation