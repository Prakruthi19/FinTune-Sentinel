
mport gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 1. Configuration
# Replace with your actual HF repo and GGUF filename
repo_id = "Prakruthi/FinTune-Sentinel-GGUF"
model_filename = "Llama-3.2-3B-Instruct.Q4_K_M.gguf" 

# 2. Download and Load Model (Runs on CPU)
print("Loading model from Hub...")
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
print("✅ Download complete!")
# Change this in app.py
print("🚀 Loading model...")
llm = Llama(
    model_path=model_path, 
    n_ctx=512,      # Reduce from 2048 → 512
    n_threads=1,    # Reduce from 2 → 1
    verbose=False,   # Hide logs
    n_gpu_layers=0
) # 2 threads for HF free tier

print("✅ Model loaded!")

def predict(message, history):
    # Professional prompt formatting
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Senior Financial Analyst.<|eot_id|><|start_header_id|>user<|end_header_id|>
{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"
    
    # Generate response
    response = llm(
        prompt, 
        max_tokens=256, 
        stop=["<|eot_id|>", "user", "system"], 
        echo=False
    )
    return response["choices"][0]["text"].strip()

# 3. Launch UI
gr.ChatInterface(
    predict,
    title="🛡️ FinTune-Sentinel (CPU Optimized)",
    description="Secure financial analysis running on edge-ready GGUF.",
).launch()
