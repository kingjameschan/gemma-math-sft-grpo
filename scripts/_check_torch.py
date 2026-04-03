import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
try:
    import flash_attn
    print(f"flash-attn: {flash_attn.__version__}")
except ImportError:
    print("flash-attn: not installed")
