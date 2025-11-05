import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU name:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

