import torch
from torch import Tensor

from .custom.unipc_sampler import UniPCSampler

from sdkit import Context

samplers = {
    'uni_pc': UniPCSampler,
}

def sample(context: Context, sampler_name:str=None, noise: Tensor=None, batch_size: int=1, shape: tuple=(), steps: int=50, cond: Tensor=None, uncond: Tensor=None, guidance_scale: float=0.8, callback=None, **kwargs):
    model = context.models['stable-diffusion']

    common_params = {
        'S': steps,
        'batch_size': batch_size,
        'shape': shape,
        'conditioning': cond,
        'verbose': False,
        'unconditional_guidance_scale': guidance_scale,
        'unconditional_conditioning': uncond,
        'eta': 0.,
        'img_callback': callback,
    }

    #more possible options and their default value:

    #callback=None,
    #normals_sequence=None,
    #quantize_x0=False,
    #mask=None,
    #x0=None,
    #temperature=1.,
    #noise_dropout=0.,
    #score_corrector=None,
    #corrector_kwargs=None,
    #log_every_t=100,
    #unconditional_guidance_scale=1.,
    #unconditional_conditioning=None,

    samples, _ = _sample_txt2img(model, sampler_name, noise, steps, batch_size, common_params.copy(), **kwargs)
    return samples

def _sample_txt2img(model, sampler_name, noise, steps, batch_size, params, **kwargs):
    sampler = samplers[sampler_name](model)
    params.update({
        'x_T': noise,
    })

    return sampler.sample(**params)

