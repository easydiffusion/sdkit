from torch import Tensor

from sdkit import Context

from .unipc_sampler import UniPCSampler

# unipc is highly customizable

# variant: bh1, bh2, vary_coeff
# time_skip: logSNR, time_uniform, time_quadratic
# order: 1, 2, 3
# lower_order_final: True, False
# thresholding: True, False

# TODO: find best suggestions for sample params
samplers = {
    "unipc_snr": {
        "variant": "bh1",
        "time_skip": "logSNR",
        "order": 3,
        "lower_order_final": True,
        "thresholding": False,
    },
    "unipc_tu": {
        "variant": "bh2",
        "time_skip": "time_uniform",
        "order": 2,
        "lower_order_final": True,
        "thresholding": False,
    },
    "unipc_tq": {
        "variant": "bh1",
        "time_skip": "time_quadratic",
        "order": 3,
        "lower_order_final": True,
        "thresholding": False,
    },
    "unipc_snr_2": {
        "variant": "vary_coeff",
        "time_skip": "logSNR",
        "order": 1,
        "lower_order_final": True,
        "thresholding": False,
    },
    "unipc_tu_2": {
        "variant": "bh1",
        "time_skip": "time_uniform",
        "order": 3,
        "lower_order_final": True,
        "thresholding": False,
    },
}


def sample(
    context: Context,
    sampler_name: str = None,
    noise: Tensor = None,
    batch_size: int = 1,
    shape: tuple = (),
    steps: int = 50,
    cond: Tensor = None,
    uncond: Tensor = None,
    guidance_scale: float = 0.8,
    callback=None,
    **kwargs
):
    model = context.models["stable-diffusion"]

    common_params = {
        "S": steps,
        "batch_size": batch_size,
        "shape": shape,
        "conditioning": cond,
        "verbose": False,
        "unconditional_guidance_scale": guidance_scale,
        "unconditional_conditioning": uncond,
        "eta": 0.0,
        "img_callback": callback,
    }
    common_params.update(samplers[sampler_name])
    if "sampler_params" in kwargs:
        common_params.update(kwargs["sampler_params"])

    # more possible options and their default value:

    # callback=None,
    # normals_sequence=None,
    # quantize_x0=False,
    # mask=None,
    # x0=None,
    # temperature=1.,
    # noise_dropout=0.,
    # score_corrector=None,
    # corrector_kwargs=None,
    # log_every_t=100,
    # unconditional_guidance_scale=1.,
    # unconditional_conditioning=None,

    samples, _ = _sample_txt2img(model, sampler_name, noise, steps, batch_size, common_params.copy(), **kwargs)
    return samples


def _sample_txt2img(model, sampler_name, noise, steps, batch_size, params, **kwargs):
    sampler = UniPCSampler(model)
    params.update(
        {
            "x_T": noise,
        }
    )

    return sampler.sample(**params)
