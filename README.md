# Dash Attention Training Modules Implementations in Megatron

- kernels: `./infllmv2_entmax` contains stage 1, transform score (adding attention sinks and sliding windows), and stage 2 kernels.

- When initalizing the model, the attention module is an instance of `attention.py`'s class `SelfAttention`. The core attention refers to `dash_attention.py`'s `DashAttnDotProductAttention`. 

- The entrance of DashAttention Module (parameters): `attention.py:367`

- The detailed implementation of DashAttention's Training: `dash_attention.py:181`



