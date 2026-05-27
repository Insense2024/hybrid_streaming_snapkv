"""
Apply StreamingLLM to a GPT‑NeoX model (Pythia) using the official StartRecentKVCache.
The attention forward is lightly wrapped to compress KV cache after each step.
"""
import torch
from streaming_llm.kv_cache import StartRecentKVCache

def apply_streamingllm_to_model(model, start_size=4, recent_size=508):
    # Create one KV cache manager per layer
    kv_caches = [
        StartRecentKVCache(start_size=start_size, recent_size=recent_size,
                           k_seq_dim=2, v_seq_dim=2)
        for _ in model.gpt_neox.layers
    ]

    for i, layer in enumerate(model.gpt_neox.layers):
        attn = layer.attention
        orig_forward = attn.forward
        kv_cache = kv_caches[i]

        def make_forward(old_forward, cache):
            def custom_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                layer_past=None,
                use_cache=False,
                output_attentions=False,
                **kwargs
            ):
                # Call original forward (let it handle everything internally)
                out = old_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    layer_past=layer_past,
                    use_cache=True,          # force cache
                    output_attentions=False,
                    **kwargs
                )
                # out is either (attn_output, present) or more
                attn_output = out[0]
                present = out[1] if len(out) > 1 else None

                if use_cache and present is not None:
                    # present is a tuple (key, value)
                    compressed_present = cache(present)   # official compression
                    if output_attentions:
                        return (attn_output, compressed_present) + out[2:]
                    return attn_output, compressed_present
                return out

            return custom_forward

        attn.forward = make_forward(orig_forward, kv_cache)

    return kv_caches   # not strictly needed, but you can keep it