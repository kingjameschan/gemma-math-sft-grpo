import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

import flash_attn
print(f"flash-attn: {flash_attn.__version__}")

# Test flash attention works
from flash_attn import flash_attn_func
batch, heads, seqlen, dim = 2, 8, 256, 64
q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
k = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
v = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, k, v)
print(f"flash_attn_func output shape: {out.shape} -- OK!")

# Test varlen (needed for packing)
from flash_attn import flash_attn_varlen_func
total_tokens = 512
q2 = torch.randn(total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
k2 = torch.randn(total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
v2 = torch.randn(total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
cu_seqlens = torch.tensor([0, 200, 512], dtype=torch.int32, device="cuda")
out2 = flash_attn_varlen_func(q2, k2, v2, cu_seqlens, cu_seqlens, 200, 200)
print(f"flash_attn_varlen_func output shape: {out2.shape} -- OK! (packing ready)")
