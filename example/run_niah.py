import argparse
import torch
from transformers import AutoTokenizer
from dash_attn.models.llama import LlamaForCausalLM 


DEFAULT_MODEL_PATH = "/home/test/test01/hyx/MiniCPM-4-1B-SFT-DashAttn"
BOLD = "\033[1m"
RESET = "\033[0m"


def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def make_input(digits: str, before_repeats: int, after_repeats: int) -> str:
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * before_repeats
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * after_repeats
    query = "Now, give me the exact number of the pass key. The pass key is "
    return head + before + needle + after + query


def count_chat_tokens(tokenizer, prompt: str) -> int:
    return int(encode_chat_prompt(tokenizer, prompt).input_ids.shape[-1])


def build_niah_s_1_prompt(tokenizer, digits: str, target_tokens: int) -> tuple[str, str, str]:
    filler = "The sky is blue. The tree is green. The flower is red. The sun is yellow. "
    base_prompt = make_input(digits, before_repeats=0, after_repeats=0)
    base_tokens = count_chat_tokens(tokenizer, base_prompt)
    filler_tokens = max(count_chat_tokens(tokenizer, base_prompt + filler) - base_tokens, 1)
    total_repeats = max((target_tokens - base_tokens) // filler_tokens, 0)

    before_repeats = total_repeats // 3
    after_repeats = total_repeats - before_repeats
    prompt = make_input(digits, before_repeats=before_repeats, after_repeats=after_repeats)

    # Add one sentence at a time until the fully chat-templated prompt is just
    # under the requested token budget.
    while count_chat_tokens(tokenizer, make_input(digits, before_repeats, after_repeats + 1)) <= target_tokens:
        after_repeats += 1
        prompt = make_input(digits, before_repeats=before_repeats, after_repeats=after_repeats)

    query = "Now, give me the exact number of the pass key. The pass key is"
    return prompt, digits, query


def encode_chat_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template is None:
        return tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 16k NIAH-style generate example with DashAttn model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--target-tokens", type=int, default=16384)
    parser.add_argument("--digits", default="4835926")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="dash_attn", choices=["dash_attn", "sdpa"])
    parser.add_argument("--scaling-factor", type=float, default=1.0)
    parser.add_argument("--sparsity-exclude-prefix-tokens", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    prompt, answer, question = build_niah_s_1_prompt(tokenizer, args.digits, args.target_tokens)
    inputs = encode_chat_prompt(tokenizer, prompt)
    input_ids = inputs.input_ids.to(args.device)
    attention_mask = inputs.attention_mask.to(args.device)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
        scaling_factor=args.scaling_factor,
    ).to(args.device)
    model.eval()

    print(bold(f"Prompt tokens: {input_ids.shape[-1]}"))
    print(bold(f"Question: {question}"))
    print(bold(f"Expected answer: {answer}"))

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_active_blocks=args.attn_implementation == "dash_attn",
        )

    active_blocks = model.get_active_blocks()
    if active_blocks is not None:
        sequence_tokens = active_blocks[0].shape[-1]
        exclude_tokens = min(args.sparsity_exclude_prefix_tokens, sequence_tokens)
        measured_blocks = [blocks[..., exclude_tokens:] for blocks in active_blocks]
        token_count = measured_blocks[0].shape[-1]
        if token_count > 0:
            avg_active_blocks = torch.stack([blocks.float().mean() for blocks in measured_blocks]).mean().item()
            dense_equivalent_tokens = avg_active_blocks * model.config.chunk_size
            sparsity = 1.0 - dense_equivalent_tokens / sequence_tokens
            print(bold(f"Sparsity: {100 * sparsity:.4f}%"))

    generated = output_ids[0, input_ids.shape[-1] :]
    print(bold("Model answer:"))
    print(bold(tokenizer.decode(generated, skip_special_tokens=True).strip()))


if __name__ == "__main__":
    main()
