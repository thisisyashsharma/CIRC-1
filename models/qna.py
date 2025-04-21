from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# ✅ Load Hugging Face token securely from environment
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN environment variable not set. Please add it in Streamlit Cloud secrets.")

# ✅ Auto-detect device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)

@torch.no_grad()
def ask_question(caption, question):
    prompt = f"""Image Caption: {caption}

Question: {question}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
