<h1>DashAttention</h1>


**Differentiable and Adaptive Sparse Hierarchical Attention**
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/fasa-org/dash-attention" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/TODO" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/NOSA-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/TODO" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-TODO-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>



# Installation

TBD

# Usage

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
# Efficiency

TBD

# Performance

TBD


# Citation 

```latex
@article{dash-attention,
  title={DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention},
  author={Huang, Yuxiang and Gon{\c{c}}alves, Nuno M. T. and Alvetreti, Federico and Li, Lei and Han, Xu and Ponti, Edoardo M. and Martins, Andr{\'e} F. T. and Treviso, Marcos V.},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```