# fikra_api_server.py – FastAPI server tuned for Apple‑Silicon MPS (v0.2.2)
# ------------------------------------------------------------------
# • Removed stray comment that caused a SyntaxError.
# • torch.compile now called without invalid backend arg.
# ------------------------------------------------------------------

import os
import asyncio
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

# ------------------------------------------------------------------
# 1. Environment & device
# ------------------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # Allow CPU fallback for unsupported MPS ops
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # Advise MPS memory limit (0.0 means no specific limit, let PyTorch manage)

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
print(f"[Server] Running on {DEVICE} (dtype={DTYPE})")

# ------------------------------------------------------------------
# 2. Paths – edit if needed
# ------------------------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR = "./mistral_fikra_output"  # LoRA weights you trained

# ------------------------------------------------------------------
# 3. Load & optimise model once at startup
# ------------------------------------------------------------------
print("[Server] Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

def load_model() -> torch.nn.Module:
    print("[Server] Loading base model …")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )
    print("[Server] Attaching LoRA adapter …")
    merged = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
    merged.to(DEVICE)

    if DEVICE == "mps":
        print("[Server] torch.compile on MPS … (first request will take ~15-30 s with max-autotune)")
        merged = torch.compile(merged, mode="max-autotune") # Use max-autotune for potentially better perf

    merged.eval()
    torch.set_grad_enabled(False) # Disable gradients for inference
    return merged

model = load_model()

# ------------------------------------------------------------------
# 4. FastAPI setup with streaming
# ------------------------------------------------------------------
app = FastAPI(title="Fikra‑GPT API", version="0.2.2")

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = 60
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    do_sample: bool | None = True

# Async generator that yields the whole completion once ready
async def generate_tokens(req: GenerationRequest) -> AsyncGenerator[bytes, None]:
    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)

    # Ensure input tensors are in the correct dtype, especially for MPS
    inputs = {key: val.to(dtype=DTYPE if val.is_floating_point() else val.dtype) for key, val in inputs.items()}

    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    # buffer to capture tokens from streamer callback
    buffer: list[str] = []
    streamer.accept_token = lambda token_id, token: buffer.append(token)

    gen_task = asyncio.to_thread(
        model.generate,
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=req.do_sample,
        use_cache=True,
        streamer=streamer,
    )

    await gen_task  # wait for generation to finish
    yield "".join(buffer).encode("utf-8")

@app.post("/generate", response_class=StreamingResponse)
async def generate_endpoint(req: GenerationRequest):
    if not req.prompt.strip():
        return JSONResponse({"error": "prompt must not be empty"}, status_code=400)

    return StreamingResponse(generate_tokens(req), media_type="text/plain")

# ------------------------------------------------------------------
# 5. Health‑check route
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "dtype": str(DTYPE)}

# ------------------------------------------------------------------
# 6. Entry‑point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fikra_api_server_2:app", # Corrected module name to match filename
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
