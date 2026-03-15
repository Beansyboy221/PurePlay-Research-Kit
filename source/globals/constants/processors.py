import torch
import os

#region Processors
TORCH_DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DEVICE = torch.device(TORCH_DEVICE_TYPE)
CPU_WORKERS = max(os.cpu_count()//2, 2) if TORCH_DEVICE_TYPE == 'cuda' \
    else os.cpu_count()//2
#endregion