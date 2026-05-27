import torch
import time
from datasets import load_dataset

def get_wikitext_text(max_chars=50000):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    return text[:max_chars]

def get_pg19_sample(sample_idx=0, max_chars=50000):
    dataset = load_dataset("emozilla/pg19", split="test")  
    sample = dataset[sample_idx]["text"]
    return sample[:max_chars]

def compute_ppl_and_speed(model, tokenizer, text, prefill_len=512,
                          max_new_tokens=256, device="cuda", past_compressor=None):
    inputs = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    seq_len = inputs.size(0)
    if seq_len < prefill_len + max_new_tokens + 1:
        prefill_len = min(256, seq_len // 2)
        max_new_tokens = seq_len - prefill_len - 1
        if max_new_tokens <= 0:
            raise ValueError("文本太短，请增大 max_chars")

    prefill_ids = inputs[:prefill_len].unsqueeze(0)
    target_ids = inputs[prefill_len:prefill_len + max_new_tokens]

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model(prefill_ids, use_cache=True)
        past = outputs.past_key_values
        if past_compressor is not None:
            past = past_compressor(past)
    torch.cuda.synchronize()
    ttft = time.time() - start

    total_loss = 0.0
    tpot_sum = 0.0
    prev_token = prefill_ids[:, -1:]

    for i, target in enumerate(target_ids):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model(prev_token, past_key_values=past, use_cache=True)
        torch.cuda.synchronize()
        tpot_sum += time.time() - start

        past = outputs.past_key_values
        logits = outputs.logits[0, -1, :]
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), target.unsqueeze(0)
        )
        total_loss += loss.item()

        prev_token = target.unsqueeze(0).unsqueeze(0)

    avg_tpot = tpot_sum / max_new_tokens
    ppl = torch.exp(torch.tensor(total_loss / max_new_tokens)).item()
    return ppl, ttft, avg_tpot