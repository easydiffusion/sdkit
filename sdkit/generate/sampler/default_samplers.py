import torch
from torch import Tensor

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from sdkit import Context

samplers = {
    'ddim': DDIMSampler,
    'plms': PLMSSampler,
    'dpm_solver_stability': DPMSolverSampler,
}

def sample(context: Context, sampler_name:str=None, noise: Tensor=None, batch_size: int=1, shape: tuple=(), steps: int=50, cond: Tensor=None, uncond: Tensor=None, guidance_scale: float=0.8, callback=None, **kwargs):
    model = context.models['stable-diffusion']
    sample_fn = _sample_txt2img if 'init_image_latent' not in kwargs else _sample_img2img

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

    samples, _ = sample_fn(model, sampler_name, noise, steps, batch_size, common_params.copy(), **kwargs)
    return samples

def _sample_txt2img(model, sampler_name, noise, steps, batch_size, params, **kwargs):
    sampler = samplers[sampler_name](model)
    params.update({
        'x_T': noise,
    })

    return sampler.sample(**params)

def _sample_img2img(model, sampler_name, noise, steps, batch_size, params, **kwargs):
    sampler = DDIMSampler(model)

    actual_inference_steps = int(steps * kwargs['prompt_strength'])
    init_image_latent = kwargs['init_image_latent']
    mask = kwargs.get('mask')

    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0., verbose=False)
    z_enc = sampler.stochastic_encode(init_image_latent, torch.tensor([actual_inference_steps] * batch_size).to(model.device), noise=noise)

    sampler.make_schedule = (lambda **kwargs: kwargs) # we've already called this, don't call this again from within the sampler
    sampler.ddim_timesteps = sampler.ddim_timesteps[:actual_inference_steps]

    params.update({
        'S': actual_inference_steps,
        'x_T': z_enc,
        'x0': init_image_latent if mask is not None else None,
        'mask': mask,
    })

    return sampler.sample(**params)
