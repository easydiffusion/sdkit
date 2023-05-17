import torch
from sdkit.modules.codeformer.utils.registry import ARCH_REGISTRY
from sdkit.modules.codeformer.archs.codeformer_arch import CodeFormer

from sdkit import Context


def load_model(context: Context, **kwargs):
    # Register CodeFormer
    ARCH_REGISTRY.register(CodeFormer)
    

def unload_model(context: Context, **kwargs):
    pass

