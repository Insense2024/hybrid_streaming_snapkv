import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_wikitext_text, get_pg19_sample, compute_ppl_and_speed
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "EleutherAI/pythia-70m"

print("Loading model (baseline)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("\n=== WikiText-2 (Full Cache) ===")
text = get_wikitext_text()
ppl, ttft, tpot = compute_ppl_and_speed(model, tokenizer, text, device=device)
print(f"PPL: {ppl:.2f}  |  TTFT: {ttft:.4f}s  |  Avg TPOT: {tpot:.4f}s")

print("\n=== PG-19 (Full Cache) ===")
text = get_pg19_sample()
ppl, ttft, tpot = compute_ppl_and_speed(model, tokenizer, text, device=device)
print(f"PPL: {ppl:.2f}  |  TTFT: {ttft:.4f}s  |  Avg TPOT: {tpot:.4f}s")