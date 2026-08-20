import torch

print("Torch Version:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
