import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from sdkit.modules.codeformer.utils.realesrgan_utils import RealESRGANer
from sdkit.modules.codeformer.utils.misc import get_device, gpu_is_available
from sdkit.modules.codeformer.utils.download_util import load_file_from_url
from sdkit.modules.codeformer.archs.codeformer_arch import CodeFormer
from sdkit.modules.codeformer.utils.registry import ARCH_REGISTRY

from sdkit import Context
import torch


def load_model(context: Context, **kwargs):
    # Register CodeFormer
    ARCH_REGISTRY.register(CodeFormer)
    
    # Load pre-trained model URL
    pretrain_model_url = {
        'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
        'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
    }

    # Extract the model directory path for CodeFormer
    codeformer_path = context.model_paths['codeformer']
    model_dir = os.path.dirname(os.path.dirname(codeformer_path))
    
    # Ensure the sub-folders exist under 'codeformer' directory
    for model_type in ['parsing', 'detection']:
        model_folder_path = os.path.join(model_dir, 'codeformer', model_type)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
    
    # download weights if not exists
    for model_type in pretrain_model_url:
        if model_type == 'parsing' or model_type == 'detection':
            model_folder_path = os.path.join(model_dir, 'codeformer', model_type)
        else:
            model_folder_path = os.path.join(model_dir, model_type)
            
        if not os.path.exists(os.path.join(model_folder_path, f'{model_type}.pth')):
            load_file_from_url(url=pretrain_model_url[model_type], model_dir=model_folder_path, progress=True, file_name=None)

    def set_realesrgan():
        half = True if gpu_is_available() else False
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path=os.path.join(model_dir, "realesrgan/RealESRGAN_x2plus.pth"),
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=half,
        )
        return upsampler

    # Extract the model directory path for CodeFormer
    codeformer_path = context.model_paths['codeformer']
    model_dir = os.path.dirname(os.path.dirname(codeformer_path))

    upsampler = set_realesrgan()
    device = get_device()
    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt_path = os.path.join(model_dir, "codeformer/codeformer.pth")
    checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

    return (upsampler, codeformer_net)
    

def unload_model(context: Context, **kwargs):
    pass
