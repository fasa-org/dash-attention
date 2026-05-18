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
  <a href="https://arxiv.org/abs/TODO" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-TODO-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>
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

## Models

We release our 8B models for reproducibility. 

| Model | Link |
|:-:|:-:|
| 8B-FullAttn | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-FullAttn) |
| 8B-InfLLMv2 | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-InfLLMv2) |
| 8B-NSA | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-NSA) |
| 8B-DashAttention | [Hugging Face](https://huggingface.co/fasa-org/MiniCPM-4-8B-DashAttention) |

## Benchmarks

- Efficiency: Please refer to [README](./benchmarks/efficiency/README.md).

- Perfermance: Please refer to [README](./benchmarks/performance/README.md).

## Acknowledgement

This repository is developed with the aid of [RULER](https://github.com/NVIDIA/RULER), [OLMES](https://github.com/allenai/olmes), [InfLLMv2](https://github.com/OpenBMB/infllmv2_cuda_impl), and [NSA-triton](https://github.com/XunhaoLai/native-sparse-attention-triton).

## Citation 

```latex
@article{dash-attention,
  title={DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention},
  author={Huang, Yuxiang and Gon{\c{c}}alves, Nuno M. T. and Alvetreti, Federico and Li, Lei and Han, Xu and Ponti, Edoardo M. and Martins, Andr{\'e} F. T. and Treviso, Marcos V.},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```
