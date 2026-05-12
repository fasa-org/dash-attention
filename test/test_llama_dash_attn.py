import pytest
import torch
import transformers


def test_llama_dash_attn_smoke():
    from dash_attn.models.llama import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        chunk_size=16,
        estimate_diagonal=False,
        sigma=1.0e6,
        scaling_factor=1.0,
    )

    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (1, 64), device="cuda")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False, return_active_blocks=True)
    torch.cuda.synchronize()

    assert outputs.logits.shape == (1, 64, config.vocab_size)
    active_blocks = model.get_active_blocks()
    assert active_blocks is not None
    assert len(active_blocks) == config.num_hidden_layers
    assert active_blocks[0].shape == (1, config.num_key_value_heads, 64)
    assert torch.isfinite(outputs.logits).all()

if __name__ == "__main__":
    test_llama_dash_attn_smoke()
