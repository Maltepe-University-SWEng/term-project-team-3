# fikra_api_server.py – Serve your LoRA-fine-tuned Mistral via HTTP
# ---------------------------------------------------------------
# Quick-start:
#   pip install fastapi uvicorn[standard] transformers peft torch datasets
#   python fikra_api_server.py
# Then POST a JSON body {"prompt": "..."} to http://localhost:8000/generate
# ---------------------------------------------------------------
from starlette.middleware.cors import CORSMiddleware

MAX_NEW_TOKENS = 120
BATCH_SIZE = 1
CONTEXT_LENGTH = 256
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn

# ---------------------------------------------------------------
# Config – edit if your paths differ
# ---------------------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "./mistral_fikra_output"  # LoRA weights you saved after training

DEVICE = (
    "mps" if torch.backends.mps.is_available() else
    ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Running on {DEVICE}")

DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

# ---------------------------------------------------------------
# Load base + LoRA and merge for faster inference
# ---------------------------------------------------------------
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,  # Keep initial DTYPE for base model loading if possible
    trust_remote_code=True,
).to(DEVICE)  # Load base model to target device first

print("Attaching LoRA adapter …")
# Load PeftModel, ensuring it's on the same device as base_model initially
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, device_map={"": DEVICE})

print("Merging LoRA weights …")
# Move model to CPU for merging to save MPS/GPU memory
model = model.to("cpu")
model = model.merge_and_unload()  # makes a single set of weights; no PEFT overhead

# Move the merged model back to the target device and set dtype and eval mode
model.to(DEVICE, dtype=DTYPE).eval()

# Clean up base_model if it's no longer needed and taking up memory
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()
elif DEVICE == "cuda":
    torch.cuda.empty_cache()

# ---------------------------------------------------------------
# FastAPI definitions
# ---------------------------------------------------------------
app = FastAPI(title="Fikra-GPT API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 60
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True

class GenerationResponse(BaseModel):
    generated_text: str



@app.post("/generate", response_model=GenerationResponse)
@torch.inference_mode()
def generate(req: GenerationRequest):
    inputs = tokenizer(
        req.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256  # Reduced from 512
    ).to(DEVICE)

    out = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
        num_beams=1,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        repetition_penalty=1.0,  # Disable repetition penalty
        length_penalty=1.0  # Disable length penalty
    )
    return {"generated_text": tokenizer.decode(out[0], skip_special_tokens=True)}
# ---------------------------------------------------------------
# Entry-point for `python fikra_api_server.py`
# ---------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("fikra_api_server:app", host="0.0.0.0", port=8000, reload=False)


