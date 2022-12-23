import torch
from gc import collect

from sdkit import Context

def gc(context: Context):
    collect()
    if context.device == 'cpu':
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
