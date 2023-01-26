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

def print_largest_tensors_in_memory(device, num=10):
    objs = []
    total_mem = 0
    device = torch.device(device) if isinstance(device, str) else device
    l = get_objects()
    for obj in l:
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))) and obj.device == device:
                size = obj.nelement() * obj.element_size() / 1024**2
                x = str(obj)
                objs.append([size, obj.shape, f'{x[:100]}..{x[-100:-1]}'])
                del x
                total_mem += size
        except:
            pass

    objs.sort(key=lambda x: x[0], reverse=True)
    n = len(objs)
    objs = objs[:min(num, n)]

    print(f'== {num} largest tensors on {device} ==')
    for i, o in enumerate(objs):
        print(f'{i+1}. Size: {o[0]:.1f} MiB, Shape: {o[1]}, Data: {o[2]}')
    print('--')
    print(f'Total memory occupied on {device} by {n} tensors: {total_mem:.1f} MiB')
