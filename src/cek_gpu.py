import torch

if torch.cuda.is_available():
    print("GPU tersedia")
    print("Nama GPU:", torch.cuda.get_device_name(0))
    print("Jumlah GPU:", torch.cuda.device_count())
else:
    print("GPU tidak tersedia")