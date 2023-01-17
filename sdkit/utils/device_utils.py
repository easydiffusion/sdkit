import torch
import psutil
from gc import collect, get_objects

from sdkit import Context

def gc(context: Context):
    collect()
    if context.device == 'cpu':
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def get_device_usage(device, log_info=False):
    cpu_used = psutil.cpu_percent()
    ram_used, ram_total = psutil.virtual_memory().used, psutil.virtual_memory().total
    vram_free, vram_total = torch.cuda.mem_get_info(device) if device != 'cpu' else (0, 0)
    vram_used = (vram_total - vram_free)

    ram_used /= 1024**3
    ram_total /= 1024**3
    vram_used /= 1024**3
    vram_total /= 1024**3

    if log_info:
        from sdkit.utils import log
        msg = f'CPU utilization: {cpu_used:.1f}%, System RAM used: {ram_used:.1f} of {ram_total:.1f} GiB'
        if device != 'cpu': msg += f', GPU RAM used ({device}): {vram_used:.1f} of {vram_total:.1f} GiB'
        log.info(msg)

    return cpu_used, ram_used, ram_total, vram_used, vram_total
