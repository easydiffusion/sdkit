"""
A simplified version of the original txt2img.py script that is included with Stable Diffusion 2.0.

Useful for testing responses and memory usage against the original script.
"""

from contextlib import nullcontext
from itertools import islice

import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm, trange

torch.set_grad_enabled(False)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    seed_everything(42)

    config = OmegaConf.load("path/to/models/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, f"path/to/models/stable-diffusion/sd-v1-4.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = PLMSSampler(model)

    batch_size = 1
    prompt = "astronaut"
    assert prompt is not None
    data = [batch_size * [prompt]]

    sample_count = 0

    start_code = None

    for i in range(4):
        precision_scope = autocast
        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
            for n in trange(1, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [4, 2048 // 8, 2048 // 8]
                    try:
                        samples, _ = sampler.sample(
                            S=1,
                            conditioning=c,
                            batch_size=1,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=7.5,
                            unconditional_conditioning=uc,
                            eta=0.0,
                            x_T=start_code,
                        )

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            base_count += 1
                            sample_count += 1
                    except Exception as e:
                        print(e)


main()
