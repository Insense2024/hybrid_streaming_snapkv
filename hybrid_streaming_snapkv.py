import torch
from kvpress import SnapKVPress


class HybridKVManager:
    def __init__(
        self,
        sink_tokens=4,
        window_tokens=384,
        max_cache_tokens=512,
    ):
        self.sink_tokens = sink_tokens
        self.window_tokens = window_tokens
        self.max_cache_tokens = max_cache_tokens

    def update(self, key, value, scores=None):
        """
        key/value:
            [b, h, seq, d]

        scores:
            [b, kv_heads, seq]
        """

        b, h, seq_len, d = key.shape

        if seq_len <= self.max_cache_tokens:
            return key, value

        # ------------------------------------------------
        # sink tokens
        # ------------------------------------------------

        sink_idx = list(range(self.sink_tokens))

        # ------------------------------------------------
        # recent window
        # ------------------------------------------------

        recent_start = max(
            self.sink_tokens,
            seq_len - self.window_tokens
        )

        recent_idx = list(range(recent_start, seq_len))

        # ------------------------------------------------
        # important tokens from SnapKV
        # ------------------------------------------------

        important_idx = []

        if scores is not None:

            mean_scores = scores.mean(dim=(0, 1))

            budget = (
                self.max_cache_tokens
                - len(sink_idx)
                - len(recent_idx)
            )

            if budget > 0:

                topk = torch.topk(
                    mean_scores,
                    k=min(budget, mean_scores.numel())
                ).indices.tolist()

                important_idx = topk

        # ------------------------------------------------
        # merge
        # ------------------------------------------------

        keep_idx = sorted(
            list(
                set(
                    sink_idx
                    + recent_idx
                    + important_idx
                )
            )
        )

        keep = torch.tensor(
            keep_idx,
            device=key.device,
            dtype=torch.long
        )

        keep_exp = (
            keep[None, None, :, None]
            .expand(b, h, len(keep_idx), d)
        )

        key = torch.gather(key, 2, keep_exp)
        value = torch.gather(value, 2, keep_exp)

        return key, value


def apply_hybrid_streaming_snapkv(
    model,
    compression_ratio=0.5,
    sink_tokens=4,
    window_tokens=384,
    max_cache_tokens=512,
):

    press = SnapKVPress(
        compression_ratio=compression_ratio
    )

    managers = [
        HybridKVManager(
            sink_tokens=sink_tokens,
            window_tokens=window_tokens,
            max_cache_tokens=max_cache_tokens,
        )
        for _ in model.gpt_neox.layers
    ]

    for i, layer in enumerate(model.gpt_neox.layers):

        attn = layer.attention
        orig_forward = attn.forward
        manager = managers[i]

        def make_forward(old_forward, mgr):

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

                out = old_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    layer_past=layer_past,
                    use_cache=True,
                    output_attentions=True,
                    **kwargs
                )

                attn_output = out[0]
                attentions = out[1]
                present = out[2]

                if use_cache and present is not None:

                    key, value = present

                    try:

                        scores = press.score(
                            module=attn,
                            hidden_states=hidden_states,
                            keys=key,
                            values=value,
                            attentions=attentions,
                            kwargs=kwargs,
                        )

                    except Exception:
                        scores = None

                    key, value = mgr.update(
                        key,
                        value,
                        scores=scores
                    )

                    present = (key, value)

                return attn_output, present

            return custom_forward

        attn.forward = make_forward(orig_forward, manager)

    return model