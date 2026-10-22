import torch
import torchvision
import numpy as np

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("CUDA version used by PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
    print("Compute capability:", torch.cuda.get_device_capability(0))

    # Test an actual GPU computation
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU computation successful!")
    print("Output shape:", y.shape)
    print("Output device:", y.device)
    print("Allocated GPU memory:", round(torch.cuda.memory_allocated() / 1024**2, 2), "MB")
else:
    print("CUDA is NOT available. Check your PyTorch installation and NVIDIA driver.")