<div align="center">
<h1>DashAttention</h1>

<p><strong>Differentiable and Adaptive Sparse Hierarchical Attention</strong></p>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/fasa-org/dash-attention" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/fasa-org/dashattention" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/DashAttention-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2605.18753" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-2605.18753-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>



## Installation

For the usage of DashAttention kernels and running the example, please run the following script:
```
pip install -e .
```

For benchmark environment setup, please refer to each corresponding folder.

## Usage

The dash attention's interface can be used as follows:

```python
queries = torch.randn(batch, query_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
keys = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
values = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype).contiguous()
head_cls = torch.randn(kv_heads, head_dim, device=device, dtype=dtype).contiguous()

model = dash_attn(
    chunk_size=chunk_size,
    enable_gqa=True,
    estimate_diagonal=True,
    return_active_blocks=True,
)

out, active_blocks = model(queries, keys, values, head_cls)
```

We also provide an example on how to use DashAttention in Llama-architecture models in [here](./example/run_niah.py).
```
python ./example/run_niah.py
```

## Documentation

DashAttention implements the attention mechanism introduced in [DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention](https://arxiv.org/abs/2605.18753). The method replaces fixed-budget top-k block routing with an adaptive, differentiable sparse router, then refines the selected regions with token-level softmax attention.

### How it works

The implementation follows the three-stage hierarchy described in the paper:

1. **Local chunk summarization**: `dash_attn.prefill.summarize_chunk` and `dash_attn.decoding.summarize_chunk` build one learned key summary per KV chunk.
2. **Entmax block routing**: `score_blocks` computes sparse chunk supports and routing priors from query-to-summary scores.
3. **Prior-induced sparse softmax**: `full_attn` applies token-level attention only over routed chunks, using the Stage 1 prior to preserve differentiability through the hierarchy.

The public kernel wrapper is [`dash_attn.dash_attn_interface.dash_attn`](./dash_attn/dash_attn_interface.py). It supports both prefill and decoding: prefill summarizes the current sequence and stores complete chunk summaries, while decoding reuses the chunk-summary cache and appends newly completed chunks.

### Core API

```python
from dash_attn import dash_attn

attn = dash_attn(
    chunk_size=64,
    enable_gqa=True,
    estimate_diagonal=True,
    scaling_factor=1.0,
    return_active_blocks=False,
)
```

Important arguments:

| Argument | Description |
|:-|:-|
| `chunk_size` | Number of tokens per routed KV chunk. |
| `enable_gqa` | Enables grouped-query attention support when query heads outnumber KV heads. |
| `estimate_diagonal` | Includes special handling for the current or near-diagonal chunk. |
| `scaling_factor` | Scales routing logits before sparse block selection; this is the main knob for sparsity. |
| `return_active_blocks` | Returns the number of active routed blocks per token for sparsity analysis. |
| `max_chunks` | Preallocated chunk-summary cache capacity used during decoding. |
| `sigma` | Controls the strength of the Stage 1 routing prior used by Stage 2. |

Inputs are expected in `[batch, heads, seq_len, head_dim]` layout for `queries`, `keys`, and `values`; `head_cls` has shape `[kv_heads, head_dim]`.

### Llama integration

DashAttention includes a Llama-compatible modeling implementation in [`dash_attn.models.llama`](./dash_attn/models/llama). `LlamaConfig` defaults to `attn_implementation="dash_attn"` and adds DashAttention-specific fields such as `chunk_size`, `estimate_diagonal`, `sigma`, and `scaling_factor`.

```python
from dash_attn.models.llama import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
    "fasa-org/MiniCPM-4-8B-DashAttention",
    attn_implementation="dash_attn",
    torch_dtype="auto",
)
```

To inspect routing behavior, call the model with `return_active_blocks=True`, then read `model.get_active_blocks()`.

### Examples and tests

- [`example/run_niah.py`](./example/run_niah.py) runs a needle-in-a-haystack style generation example and reports measured sparsity.
- [`test/test_smoke.py`](./test/test_smoke.py) checks the standalone DashAttention kernel wrapper.
- [`test/test_llama_dash_attn.py`](./test/test_llama_dash_attn.py) checks the Llama integration and active-block reporting.

Run the test suite with:

```bash
pytest
```

The current kernels require CUDA-capable hardware.

## Models

We release our 8B models for reproducibility. 

| Model | Link |
|:-:|:-:|
| 8B-FullAttn | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-FullAttn) |
| 8B-InfLLMv2 | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-InfLLMv2) |
| 8B-NSA | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-NSA) |
| 8B-DashAttention | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-DashAttention) |

## Benchmarks

- Performance: Please refer to [README](./benchmarks/performance/README.md).

## License

This project is released under the [BSD-3-Clause License](./LICENSE).

## Acknowledgement

This repository is developed with the aid of [RULER](https://github.com/NVIDIA/RULER), [OLMES](https://github.com/allenai/olmes), [InfLLMv2](https://github.com/OpenBMB/infllmv2_cuda_impl), and [NSA-triton](https://github.com/XunhaoLai/native-sparse-attention-triton).

## Citation 

```latex
@article{dash-attention,
  title={DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention},
  author={Huang, Yuxiang and Gon{\c{c}}alves, Nuno M. T. and Alvetreti, Federico and Li, Lei and Han, Xu and Ponti, Edoardo M. and Martins, Andr{\'e} F. T. and Treviso, Marcos V.},
  journal={arXiv preprint arXiv:2605.18753},
  year={2026}
}
```
