import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from typing import Dict
from src.rag.config import DEVICE, DTYPE


MODEL_NAME = "ministral/Ministral-3b-instruct"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.9
DO_SAMPLE = True
REPETITION_PENALTY = 1.5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE
)
model.to(DEVICE)
model.eval()

def generate_answer(prompt: str) -> Dict:
    if not prompt.strip():
        raise ValueError("Prompt must be non-empty")

    inputs = tokenizer(prompt, return_tensors = "pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **tokenizer(prompt, return_tensors = "pt").to(DEVICE),
            max_new_tokens=256,               
            do_sample=False,                  
            no_repeat_ngram_size=4,           
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens = True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    return {"answer": text}
