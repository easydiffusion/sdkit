from contextlib import nullcontext

import torch
from pytorch_lightning import seed_everything
from tqdm import trange
from typing import Any, Optional, List, Union
from PIL import Image

from sdkit import Context
from sdkit.utils import (
    apply_color_profile,
    base64_str_to_img,
    gc,
    get_image_latent_and_mask,
    latent_samples_to_images,
    resize_img,
    log,
    black_to_transparent,
    get_image,
)

from .prompt_parser import get_cond_and_uncond
from .sampler import make_samples

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def generate_images(
    context: Context,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    init_image=None,
    init_image_mask=None,
    control_image=None,
    control_alpha=1.0,
    prompt_strength: float = 0.8,
    preserve_init_image_color_profile=False,
    strict_mask_border=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    hypernetwork_strength: float = 0,
    tiling=None,
    lora_alpha: Union[float, List[float]] = 0,
    sampler_params={},
    callback=None,
):
    req_args = locals()

    try:
        images = []

        if "stable-diffusion" not in context.models:
            raise RuntimeError(
                "The model for Stable Diffusion has not been loaded yet! If you've tried to load it, please check the logs above this message for errors (while loading the model)."
            )

        model = context.models["stable-diffusion"]

        if context.test_diffusers:
            return make_with_diffusers(
                context,
                prompt,
                negative_prompt,
                seed,
                width,
                height,
                num_outputs,
                num_inference_steps,
                guidance_scale,
                init_image,
                init_image_mask,
                control_image,
                control_alpha,
                prompt_strength,
                # preserve_init_image_color_profile,
                sampler_name,
                # hypernetwork_strength,
                lora_alpha,
                tiling,
                strict_mask_border,
                # sampler_params,
                callback,
            )

        if "hypernetwork" in context.models:
            context.models["hypernetwork"]["hypernetwork_strength"] = hypernetwork_strength

        seed_everything(seed)
        precision_scope = torch.autocast if context.half_precision else nullcontext

        with precision_scope(context.torch_device.type):
            cond, uncond = get_cond_and_uncond(prompt, negative_prompt, num_outputs, model)

        generate_fn = txt2img if init_image is None else img2img
        common_sampler_params = {
            "context": context,
            "sampler_name": sampler_name,
            "seed": seed,
            "batch_size": num_outputs,
            "shape": [4, height // 8, width // 8],
            "cond": cond,
            "uncond": uncond,
            "guidance_scale": guidance_scale,
            "sampler_params": sampler_params,
            "callback": callback,
        }

        with torch.no_grad(), precision_scope(context.torch_device.type):
            for _ in trange(1, desc="Sampling"):
                images += generate_fn(common_sampler_params.copy(), **req_args)
                gc(context)

        return images
    finally:
        context.init_image_latent, context.init_image_mask_tensor = None, None


def txt2img(params: dict, context: Context, num_inference_steps, **kwargs):
    params.update(
        {
            "steps": num_inference_steps,
        }
    )

    samples = make_samples(**params)
    return latent_samples_to_images(context, samples)


def img2img(
    params: dict,
    context: Context,
    num_inference_steps,
    num_outputs,
    width,
    height,
    init_image,
    init_image_mask,
    prompt_strength,
    preserve_init_image_color_profile,
    strict_mask_border=False,
    **kwargs,
):
    init_image = get_image(init_image)
    init_image_mask = get_image(init_image_mask)

    if not hasattr(context, "init_image_latent") or context.init_image_latent is None:
        context.init_image_latent, context.init_image_mask_tensor = get_image_latent_and_mask(
            context, init_image, init_image_mask, width, height, num_outputs
        )

    params.update(
        {
            "steps": num_inference_steps,
            "init_image_latent": context.init_image_latent,
            "mask": context.init_image_mask_tensor,
            "prompt_strength": prompt_strength,
        }
    )

    samples = make_samples(**params)
    images = latent_samples_to_images(context, samples)

    if preserve_init_image_color_profile:
        for i, img in enumerate(images):
            images[i] = apply_color_profile(init_image, img)

    if init_image_mask and strict_mask_border:
        images = blend_mask(images, init_image, init_image_mask, width, height)

    return images


def make_with_diffusers(
    context: Context,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    width: int = 512,
    height: int = 512,
    num_outputs: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    init_image=None,
    init_image_mask=None,
    control_image=None,
    control_alpha=1.0,
    prompt_strength: float = 0.8,
    # preserve_init_image_color_profile=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    # hypernetwork_strength: float = 0,
    lora_alpha: Union[float, List[float]] = 0,
    # sampler_params={},
    tiling=None,
    strict_mask_border=False,
    callback=None,
):
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionControlNetPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
    )
    from diffusers.models.lora import LoRACompatibleConv

    from sdkit.models.model_loader.lora import apply_lora_model
    from sdkit.generate.sampler import diffusers_samplers
    import numpy as np

    prompt = prompt.lower()
    negative_prompt = negative_prompt.lower()

    model = context.models["stable-diffusion"]
    default_pipe = model["default"]
    if context.torch_device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator(context.torch_device).manual_seed(seed)

    is_sd_xl = isinstance(
        default_pipe,
        (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline),
    )
    sd_config = model["config"]
    context_dim = sd_config.model.params.get("unet_config", {}).get("params", {}).get("context_dim", None)
    if is_sd_xl:
        context_dim = 2048

    cmd = {
        "guidance_scale": guidance_scale,
        "generator": generator,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": num_outputs,
    }
    if init_image:
        init_image = get_image(init_image)
        cmd["image"] = resize_img(init_image.convert("RGB"), width, height, clamp_to_8=True)
        cmd["strength"] = prompt_strength
    if init_image_mask:
        init_image_mask = get_image(init_image_mask)
        cmd["mask_image"] = resize_img(init_image_mask.convert("RGB"), width, height, clamp_to_8=True)

    if init_image:
        operation_to_apply = "inpainting" if init_image_mask else "img2img"
    else:
        operation_to_apply = "txt2img"

    if operation_to_apply not in model:
        if "inpainting" in model and len(model) == 1:
            raise RuntimeError(
                f"This model does not support {operation_to_apply}! This model requires an initial image and mask."
            )

        raise NotImplementedError(
            f"This model does not support {operation_to_apply}! Supported operations: {model.keys()}"
        )

    if control_image is None or "controlnet" not in context.models:
        operation_to_apply = model[operation_to_apply]
    else:
        controlnet = context.models["controlnet"]

        if isinstance(control_image, list):
            assert isinstance(controlnet, list)
            assert len(control_image) == len(controlnet)

            control_alpha = control_alpha if isinstance(control_alpha, list) else [1.0] * len(control_image)
            assert len(control_alpha) == len(control_image)

            for cn in controlnet:
                assert_controlnet_model(cn, context_dim)

            cmd["controlnet_conditioning_scale"] = control_alpha
            control_image = [get_image(img) for img in control_image]
            control_image = [resize_img(img.convert("RGB"), width, height, clamp_to_8=True) for img in control_image]
        else:
            control_image = get_image(control_image)
            control_image = resize_img(control_image.convert("RGB"), width, height, clamp_to_8=True)
            assert_controlnet_model(controlnet, context_dim)

            if control_alpha is None:
                control_alpha = 1.0

            assert not isinstance(control_alpha, list)
            control_alpha = float(control_alpha)
            cmd["controlnet_conditioning_scale"] = control_alpha

        if operation_to_apply == "txt2img":
            cmd["image"] = control_image
        else:
            cmd["control_image"] = control_image

        if is_sd_xl:
            controlnet_op = {
                "txt2img": StableDiffusionXLControlNetPipeline,
                "img2img": StableDiffusionXLControlNetImg2ImgPipeline,
                "inpainting": StableDiffusionXLControlNetInpaintPipeline,
            }
            operation_to_apply_cls = controlnet_op[operation_to_apply]
        else:
            controlnet_op = {
                "txt2img": StableDiffusionControlNetPipeline,
                "img2img": StableDiffusionControlNetImg2ImgPipeline,
                "inpainting": StableDiffusionControlNetInpaintPipeline,
            }
            operation_to_apply_cls = controlnet_op[operation_to_apply]

        operation_to_apply = operation_to_apply_cls(controlnet=controlnet, **default_pipe.components)

        if hasattr(operation_to_apply, "watermark"):
            operation_to_apply.watermark = None

    if sampler_name.startswith("unipc_tu"):
        sampler_name = "unipc_tu_2" if num_inference_steps < 10 else "unipc_tu"

    operation_to_apply.scheduler = diffusers_samplers.make_sampler(sampler_name, model["default_scheduler_config"])
    if operation_to_apply.scheduler is None:
        raise NotImplementedError(f"The sampler '{sampler_name}' is not supported (yet)!")
    log.info(f"Using sampler: {operation_to_apply.scheduler} because of {sampler_name}")

    if isinstance(
        operation_to_apply,
        (StableDiffusionInpaintPipelineLegacy, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline),
    ):
        del cmd["width"]
        del cmd["height"]
    elif isinstance(operation_to_apply, StableDiffusionInpaintPipeline):
        del cmd["strength"]

    cmd["callback"] = lambda i, t, x_samples: callback(x_samples, i, operation_to_apply) if callback else None
    cmd["callback_steps"] = 1

    # apply the LoRA (if necessary)
    if context.models.get("lora"):
        log.info("Applying LoRA...")
        lora_count = len(context.models["lora"])
        lora_alpha = lora_alpha if isinstance(lora_alpha, list) else [lora_alpha] * lora_count
        lora_alpha = np.array(lora_alpha)
        if hasattr(context, "_last_lora_alpha"):
            apply_lora_model(context, -context._last_lora_alpha)  # undo the last LoRA apply

        apply_lora_model(context, lora_alpha)
        context._last_lora_alpha = lora_alpha

    # --------------------------------------------------------------------------------------------------
    # -- https://github.com/huggingface/diffusers/issues/2633
    log.info("Applying tiling settings")
    if tiling == "xy":
        modex = "circular"
        modey = "circular"
    elif tiling == "x":
        modex = "circular"
        modey = "constant"
    elif tiling == "y":
        modex = "constant"
        modey = "circular"
    else:
        modex = "constant"
        modey = "constant"

    def asymmetricConv2DConvForward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        F = torch.nn.functional
        self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
        self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
        working = F.pad(input, self.paddingX, mode=modex)
        working = F.pad(working, self.paddingY, mode=modey)
        return F.conv2d(working, weight, bias, self.stride, torch.nn.modules.utils._pair(0), self.dilation, self.groups)

    def lora_conv_forward(self, hidden_states, scale=1.0):
        return super(self.__class__, self).forward(hidden_states)

    targets = [
        operation_to_apply.vae,
        operation_to_apply.text_encoder,
        operation_to_apply.unet,
    ]
    if is_sd_xl:
        targets.append(operation_to_apply.text_encoder_2)

    conv_layers = []
    targets = [t for t in targets if t]
    for target in targets:
        for module in target.modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)

    for cl in conv_layers:
        if isinstance(cl, LoRACompatibleConv) and cl.lora_layer is None:
            cl.lora_layer = lambda *x: 0
            if not hasattr(cl, "_forward_bkp"):
                cl._forward_bkp = cl.forward
                cl._forward_tiling = lora_conv_forward.__get__(cl)

            cl.forward = cl._forward_bkp if tiling is None else cl._forward_tiling

        if not hasattr(cl, "_conv_forward_bkp"):
            cl._conv_forward_bkp = cl._conv_forward

        _conv_forward_tiling = asymmetricConv2DConvForward.__get__(cl, torch.nn.Conv2d)

        cl._conv_forward = cl._conv_forward_bkp if tiling is None else _conv_forward_tiling

    # --------------------------------------------------------------------------------------------------
    log.info("Parsing the prompt...")

    # make the prompt embeds
    compel = model["compel"]
    log.info("compel is ready")

    if is_sd_xl:
        if operation_to_apply.text_encoder:
            cmd["prompt_embeds"], cmd["pooled_prompt_embeds"] = compel(prompt)
            log.info("Made prompt embeds")

            cmd["negative_prompt_embeds"], cmd["negative_pooled_prompt_embeds"] = compel(negative_prompt)
            log.info("Made negative prompt embeds")

            cmd["prompt_embeds"], cmd["negative_prompt_embeds"] = compel.pad_conditioning_tensors_to_same_length(
                [cmd["prompt_embeds"], cmd["negative_prompt_embeds"]]
            )
        elif init_image is None or init_image_mask is not None:
            raise Exception(
                "The SD-XL Refiner model only supports img2img! Please set an initial image, or remove the inpainting mask!"
            )
        else:  # SDXL refiner doesn't work with prompt embeds yet
            cmd["prompt"] = prompt
            cmd["negative_prompt"] = negative_prompt
    else:
        cmd["prompt_embeds"] = compel(prompt)
        log.info("Made prompt embeds")

        cmd["negative_prompt_embeds"] = compel(negative_prompt)
        log.info("Made negative prompt embeds")

        cmd["prompt_embeds"], cmd["negative_prompt_embeds"] = compel.pad_conditioning_tensors_to_same_length(
            [cmd["prompt_embeds"], cmd["negative_prompt_embeds"]]
        )

    log.info("Done parsing the prompt")
    # --------------------------------------------------------------------------------------------------

    # create TensorRT buffers, if necessary
    if hasattr(operation_to_apply.unet, "_allocate_trt_buffers"):
        dtype = torch.float16 if context.half_precision else torch.float32
        operation_to_apply.unet._allocate_trt_buffers(
            operation_to_apply, context.torch_device, dtype, num_outputs, width, height
        )

    # apply
    log.info(f"applying: {operation_to_apply}")
    log.info(f"Running on diffusers: {cmd}")

    enable_vae_tiling = default_pipe.vae.use_tiling
    if tiling:
        log.info(f"Disabling VAE tiling because seamless tiling is enabled: {tiling}")
        default_pipe.vae.use_tiling = False  # disable VAE tiling before use, otherwise seamless tiling fails

    try:
        images = operation_to_apply(**cmd).images
    finally:
        default_pipe.vae.use_tiling = enable_vae_tiling

    if is_sd_xl and context.half_precision:  # cleanup - workaround since SDXL upcasts the vae
        operation_to_apply.vae = operation_to_apply.vae.to(dtype=torch.float16)

    if init_image_mask and strict_mask_border:
        images = blend_mask(images, init_image, init_image_mask, width, height)

    return images


def assert_controlnet_model(controlnet, sd_context_dim):
    cn_dim = controlnet.mid_block.attentions[0].transformer_blocks[0].attn2.to_k.weight.shape[1]
    if cn_dim != sd_context_dim:
        raise RuntimeError(
            f"Sorry, you're trying to use a {get_sd_type_from_dim(cn_dim)} controlnet model with a {get_sd_type_from_dim(sd_context_dim)} Stable Diffusion model. They're not compatible, please use a compatible model!"
        )


def get_sd_type_from_dim(dim: int) -> str:
    dims = {768: "SD 1", 1024: "SD 2", 2048: "SDXL"}
    return dims.get(dim, "Unknown")


def blend_mask(images, init_image, init_image_mask, width, height):
    """
    Blend initial and final images using mask.
    Otherwise inpainting suffers from gradual degradation each execution, progressively losing detail and becoming
    blurrier even for fully preserved parts of the image which should remain unchanged. This loss is especially
    dramatic for larger images like 1024x1024. Now, this pixel space compositing approach isn't a panacea, as you can
    often see a faint discontinuity around the masked area unless you use the feathered brush when drawing the
    mask (regardless of whether color profile preservation is checked), but it at least guarantees that unchanged
    pixels remain unchanged so that you can reliably perform inpainting a dozen times to various parts. There remains
    data loss somewhere deeper along the pipeline (maybe the VAE decode and reencode is lossy, maybe the denoising
    is not properly paying attention to the mask, maybe slight noise is being added where it shouldn't be...), but
    this mitigates the issue until the root problem is identified.
    """

    if init_image_mask != None:
        init_image_mask = init_image_mask.convert("RGB")
        init_image_mask = black_to_transparent(init_image_mask)

        for i, img in enumerate(images):
            if init_image.size != img.size:
                init_image = resize_img(init_image, img.width, img.height)
            if init_image_mask.size != img.size:
                init_image_mask = resize_img(init_image_mask, img.width, img.height)

            images[i] = Image.composite(img, init_image, init_image_mask)
            images[i] = images[i].convert("RGB")

    return images
