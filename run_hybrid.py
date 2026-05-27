import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid_streaming_snapkv import apply_hybrid_streaming_snapkv
from utils import (
    get_wikitext_text,
    get_pg19_sample,
    compute_ppl_and_speed,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "EleutherAI/pythia-70m"

print("Loading Pythia-70M ...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Applying Hybrid StreamingLLM + SnapKV ...")

apply_hybrid_streaming_snapkv(
    model,
    compression_ratio=0.5,
    sink_tokens=4,
    window_tokens=384,
    max_cache_tokens=512,
)

print("\n=== WikiText-2 (HybridKV) ===")

text = get_wikitext_text()

ppl, ttft, tpot = compute_ppl_and_speed(
    model,
    tokenizer,
    text,
    device=device,
)

print(
    f"PPL: {ppl:.2f}  |  "
    f"TTFT: {ttft:.4f}s  |  "
    f"Avg TPOT: {tpot:.4f}s"
)

print("\n=== PG-19 (HybridKV) ===")

text = get_pg19_sample()

ppl, ttft, tpot = compute_ppl_and_speed(
    model,
    tokenizer,
    text,
    device=device,
)

print(
    f"PPL: {ppl:.2f}  |  "
    f"TTFT: {ttft:.4f}s  |  "
    f"Avg TPOT: {tpot:.4f}s"
)