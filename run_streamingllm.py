import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from apply_streamingllm import apply_streamingllm_to_model
from utils import get_wikitext_text, get_pg19_sample, compute_ppl_and_speed

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "EleutherAI/pythia-70m"

print("Loading Pythia-70M ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Applying StreamingLLM (start=4, recent=508) ...")
apply_streamingllm_to_model(model, start_size=4, recent_size=508)


print("\n=== WikiText-2 (StreamingLLM) ===")
text = get_wikitext_text()
ppl, ttft, tpot = compute_ppl_and_speed(model, tokenizer, text, device=device) 
print(f"PPL: {ppl:.2f}  |  TTFT: {ttft:.4f}s  |  Avg TPOT: {tpot:.4f}s")

print("\n=== PG-19 (StreamingLLM) ===")
text = get_pg19_sample()
ppl, ttft, tpot = compute_ppl_and_speed(model, tokenizer, text, device=device)
print(f"PPL: {ppl:.2f}  |  TTFT: {ttft:.4f}s  |  Avg TPOT: {tpot:.4f}s")