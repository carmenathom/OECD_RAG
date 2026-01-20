import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.9
DO_SAMPLE = True
REPETITION_PENALTY = 1.1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype = DTYPE,
    device_map = "auto"
)
model.eval()


def generate_answer(prompt: str) -> Dict:
    if not prompt.strip():
        raise ValueError("Prompt must be non-empty")

    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature = TEMPERATURE,
            top_p = TOP_P,
            do_sample = DO_SAMPLE,
            repetition_penalty = REPETITION_PENALTY,
            eos_token_id = tokenizer.eos_token_id
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens = True)

    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    return {"answer": text}
