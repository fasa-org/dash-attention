# Reproducing Performance Benchmark Results

- For performance benchmarks, we use Triton-based kernels that are mathematically equivalent to the Hugging Face (HF) modeling implementations. Please note that while the released models are ready for out-of-the-box usage, running these experiments requires patching the implementation.

- The model implementation for benchmarking is included in tools. Here are the instructions on how to patch them.
    - First, download the models from Hugging Face.
    - For InfLLMv2, NSA, DashAttention models, copy `tools/config.json` to the model directory.
        - For InfLLMv2, also copy `tools/infllmv2/*` to the model directory.
        - For NSA, also copy `tools/nsa/*` to the model directory.
        - For DashAttention, also copy `tools/dashattn/*` to the model directory.
    - Then you are all set.
        - Tune the `scaling_factor` at `tools/infllmv2/modeling_llama_long_infllmv2:818` to adjust the sparsity of DashAttention.
        - Tune the `topk` to adjust the sparsity of InfLLMv2 and NSA.
    
- RULER: Please refer to `RULER/README.md`
- HELMET: Please refer to `HELMET/README.md`