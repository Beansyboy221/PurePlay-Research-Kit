import torch

from misc import logging_utils

logger = logging_utils.get_logger()

def optimize_cuda_for_hardware() -> None:
    '''Applies CUDA optimizations based on the detected GPU architecture.'''
    if not torch.cuda.is_available():
        logger.warning('CUDA not available. Running on CPU.')
        return

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)
    
    has_tensor_cores = major >= 7
    has_tf32 = major >= 8
    has_bf16 = major >= 8

    logger.info(f'Attempting to optimize CUDA for hardware...')
    logger.info(f'CUDA device: {device_name} (sm_{major}{minor})')
    logger.info(f'Tensor Cores: {has_tensor_cores} | TF32/BF16: {has_tf32}')

    precision = 'medium' if has_tf32 else 'highest'
    torch.set_float32_matmul_precision(precision)
    logger.info(f'float32 matmul precision set to: {precision}')

    if has_bf16:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        logger.info('BF16 reduced precision reduction enabled.')

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    logger.info('cuDNN benchmark enabled (best for fixed input sizes).')