import torch


def get_prerope_query_states(module, hidden_states):
    """
    Get query states before RoPE.
    Compatible with GPTNeoX/Pythia attention.
    """

    qkv = module.query_key_value(hidden_states)

    bsz, seq_len, _ = qkv.shape

    num_heads = module.num_attention_heads
    head_dim = module.head_size

    qkv = qkv.view(
        bsz,
        seq_len,
        num_heads,
        3 * head_dim
    )

    query_states = qkv[..., :head_dim]

    query_states = query_states.permute(
        0,
        2,
        1,
        3
    ).contiguous()

    return query_states