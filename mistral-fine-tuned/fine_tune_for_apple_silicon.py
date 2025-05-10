
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# --------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./mistral_fikra_output"
LOG_DIR = "./logs"
TRAIN_FILE = "fikralar.json"
MAX_LENGTH = 512  # shorter context → lower memory

# --------------------------------------------------------------
# 2. Device & optional Metal memory cap tweak
# --------------------------------------------------------------
if torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # remove hard cap (optional)
    DEVICE = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# --------------------------------------------------------------
# 3. Tokenizer
# --------------------------------------------------------------
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    print("Tokenizer lacks pad_token → use eos_token as pad_token")
    tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.pad_token_id is None:       # if new token added
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# --------------------------------------------------------------
# 4. Base model (fp16) + LoRA adapters
# --------------------------------------------------------------
print(f"Loading base model {MODEL_NAME} …")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Resize embedding layer if we added a new token
if len(tokenizer) > model.config.vocab_size:
    print(f"Resizing token embeddings → {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

# LoRA config – ranks + where to inject
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # typical KV‑proj injection for Mistral
)

print("Attaching LoRA adapters …")
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

print(f"Moving model to {DEVICE}")
model.to(DEVICE)

# --------------------------------------------------------------
# 5. Dataset
# --------------------------------------------------------------
print(f"Loading training data from {TRAIN_FILE} …")
raw_train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

TEXT_COLUMN_GUESSES = ["fikra", "text", "content"]

def preprocess_function(examples):
    # detect text column
    col = next((c for c in TEXT_COLUMN_GUESSES if c in examples), None)
    if col is None:
        raise ValueError(f"None of {TEXT_COLUMN_GUESSES} found in dataset columns: {list(examples.keys())}")

    tokenized = tokenizer(
        examples[col],
        truncation=True,
        padding="longest",
        max_length=MAX_LENGTH,
        return_attention_mask=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing …")
tokenized_train_dataset = raw_train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_train_dataset.column_names,
)
print("Example tokenized sample:", {k: v[:3] for k, v in tokenized_train_dataset[0].items()})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --------------------------------------------------------------
# 6. TrainingArguments
# --------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # effective batch = 4
    logging_dir=LOG_DIR,
    logging_steps=25,
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True if DEVICE.type == "cuda" else False,  # cuda only
    bf16=False,
    gradient_checkpointing=False,  # not needed with LoRA
)

# --------------------------------------------------------------
# 7. Trainer
# --------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --------------------------------------------------------------
# 8. Train
# --------------------------------------------------------------
print("Starting training …")
try:
    trainer.train()
    trainer.save_model()
    print(f"Model + LoRA adapters saved to {OUTPUT_DIR}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Training failed: {e}")

# --------------------------------------------------------------
# 9. Inference example (optional)
# --------------------------------------------------------------
if __name__ == "__main__":
    prompt = "Merhaba, sana bir fıkra anlatayım: "
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=60, temperature=0.7, top_p=0.95)
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))
