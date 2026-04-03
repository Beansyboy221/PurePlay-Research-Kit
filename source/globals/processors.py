import torch
import os

TORCH_DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DEVICE = torch.device(TORCH_DEVICE_TYPE)
CPU_WORKERS = min(max(os.cpu_count() // 2, 2), 8) if TORCH_DEVICE_TYPE == 'cuda' \
    else min(os.cpu_count() // 2, 4)