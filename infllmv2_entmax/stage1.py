import torch
from torch.autograd import Function


from .stage1_fwd import compressed_attention_fwd
from .stage1_bwd import compressed_attention_bwd


class _CompressedAttentionFn(Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        kernel_size: int,
        kernel_stride: int,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q: int,
        max_seqlen_k: int,
        sm_scale: float,
    ):
        attn_score = compressed_attention_fwd(
            q,
            k,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        )

        ctx.save_for_backward(q, k, cu_seqlens_q, cu_seqlens_k)
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        ctx.sm_scale = sm_scale

        return attn_score

    @staticmethod
    def backward(ctx, grad_attn):

        q, k, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors

        grad_attn = grad_attn.contiguous()

        dq, dk = compressed_attention_bwd(
            grad_attn,
            q,
            k,
            ctx.kernel_size,
            ctx.kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.sm_scale,
        )

        return (
            dq,   # q
            dk,   # k
            None, # kernel_size
            None, # kernel_stride
            None, # cu_seqlens_q
            None, # cu_seqlens_k
            None, # max_seqlen_q
            None, # max_seqlen_k
            None, # sm_scale
        )

def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    return _CompressedAttentionFn.apply(
        q,
        k,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )

def reference_attn_score_aligned(
    q, k,
    kernel_size, kernel_stride,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_k,
    sm_scale,
):
    total_q, Hq, D = q.shape
    total_k, Hk, _ = k.shape
    B = cu_seqlens_q.numel() - 1
    share = Hq // Hk
    qk_scale = sm_scale * 1.44269504

    # ✅ 对齐 forward：输出是 [Hq, total_q, max_seqlen_k]
    out = torch.full(
        (Hq, total_q, max_seqlen_k),
        float("-inf"),
        device=q.device,
        dtype=torch.bfloat16,
    )

    for b in range(B):
        qs = int(cu_seqlens_q[b]); qe = int(cu_seqlens_q[b + 1])
        ks = int(cu_seqlens_k[b]); ke = int(cu_seqlens_k[b + 1])
        q_len = qe - qs
        k_len = ke - ks

        # forward 的 s_ptr shape=(q_len, k_len)，只会写到 [:, 0:k_len]
        # 因此 reference 也必须写到 out[..., 0:k_len]
        if k_len == 0 or q_len == 0:
            continue
        if k_len > max_seqlen_k:
            raise ValueError(f"k_len ({k_len}) > max_seqlen_k ({max_seqlen_k})")

        q_b = q[qs:qe].float()      # [q_len, Hq, D]
        k_b = k[ks:ke].float()      # [k_len, Hk, D]

        # 和 forward 对齐的“局部位置”定义：
        # q_pos 是 batch 内的 token index [0..q_len-1]
        # key_pos 是对应压缩 key index 映射回“原始位置”的坐标
        q_pos = torch.arange(q_len, device=q.device)  # [q_len]
        key_pos = torch.arange(k_len, device=q.device) * kernel_stride + (kernel_size - 1)  # [k_len]

        # forward: q_start_in_seq = ... + (kernel_size-1)
        # => batch 内 q_pos < (kernel_size-1) 的行完全不写（保持 -inf）
        valid_q = q_pos >= (kernel_size - 1)

        # forward 的 mask：off_q >= key_pos 才可见
        causal = (q_pos[:, None] >= key_pos[None, :])  # [q_len, k_len]

        for hq in range(Hq):
            hk = hq // share
            scores = (q_b[:, hq, :] @ k_b[:, hk, :].T) * qk_scale  # [q_len, k_len]

            full_scores = torch.full(
                (q_len, k_len),
                float("-inf"),
                device=q.device,
                dtype=torch.float32,
            )
            # 只在 valid_q 的行写入；不可见位置保持 -inf
            full_scores[valid_q] = torch.where(causal[valid_q], scores[valid_q], float("-inf"))

            # ✅ 关键：写入 K 轴从 0 开始，而不是 ks:ke
            out[hq, qs:qe, 0:k_len] = full_scores.to(torch.bfloat16)

    return out

def _build_varlen(B, q_lens, k_lens, device):
    cu_q = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_k = torch.zeros(B + 1, dtype=torch.int32, device=device)

    for i in range(B):
        cu_q[i + 1] = cu_q[i] + q_lens[i]
        cu_k[i + 1] = cu_k[i] + k_lens[i]

    return cu_q, cu_k


def _max_err(a, b):
    a[a<-100000] = 0
    b[b<-100000] = 0
    return (a.float() - b.float()).abs().max().item()


def test_compressed_attention():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    B = 3
    Hq = 16
    Hk = 2
    D = 64

    kernel_size = 32
    kernel_stride = 16
    sm_scale = 1.0 / (D ** 0.5)

    q_lens = [91, 203, 73]
    k_lens = [33, 67, 28]

    cu_q, cu_k = _build_varlen(B, q_lens, k_lens, DEVICE)

    total_q = int(cu_q[-1])
    total_k = int(cu_k[-1])
    max_seqlen_q = max(q_lens)
    max_seqlen_k = max(k_lens)

    q = torch.randn(total_q, Hq, D, device=DEVICE, dtype=DTYPE, requires_grad=True) 
    k = torch.randn(total_k, Hk, D, device=DEVICE, dtype=DTYPE, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)


    with torch.no_grad():
        out_triton = compressed_attention(q, k, kernel_size, kernel_stride, cu_q, cu_k, max_seqlen_q, max_seqlen_k, sm_scale)

        out_ref = reference_attn_score_aligned(
            q_ref, k_ref,
            kernel_size, kernel_stride,
            cu_q, cu_k,
            max_seqlen_k,
            sm_scale
        )

    fwd_err = _max_err(out_triton, out_ref)
    print("Forward max error:", fwd_err)

    grad = torch.randn_like(out_triton)

    # ---- triton autograd ----
    out = compressed_attention(q, k, kernel_size, kernel_stride, cu_q, cu_k, max_seqlen_q, max_seqlen_k, sm_scale)
    loss = (out * grad).sum()
    loss.backward()

    dq = q.grad.detach().clone()
    dk = k.grad.detach().clone()

    # ---- reference autograd ----
    out_ref = reference_attn_score_aligned(
        q_ref, k_ref,
        kernel_size, kernel_stride,
        cu_q, cu_k,
        max_seqlen_k,
        sm_scale
    )
    loss_ref = (out_ref * grad).sum()
    loss_ref.backward()

    dq_ref = q_ref.grad
    dk_ref = k_ref.grad

    dq_err = _max_err(dq, dq_ref)
    dk_err = _max_err(dk, dk_ref)

    print("dQ max error:", dq_err)
    print("dK max error:", dk_err)


    assert fwd_err < 5e-2, f"Forward mismatch: {fwd_err}"
    assert dq_err < 5e-2, f"dQ mismatch: {dq_err}"
    assert dk_err < 0.1, f"dK mismatch: {dk_err}"

    print("All tests passed.")


if __name__ == "__main__":
    test_compressed_attention()