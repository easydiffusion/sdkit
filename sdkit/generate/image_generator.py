from contextlib import nullcontext

import torch
from pytorch_lightning import seed_everything
from tqdm import trange
from typing import Optional, List, Union
from PIL import Image

from sdkit import Context
from sdkit.utils import (
    apply_color_profile,
    base64_str_to_img,
    gc,
    get_embedding_token,
    get_image_latent_and_mask,
    latent_samples_to_images,
    resize_img,
    log,
    load_tensor_file,
    black_to_transparent,
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
    control_alpha=None,
    prompt_strength: float = 0.8,
    preserve_init_image_color_profile=False,
    strict_mask_border=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    hypernetwork_strength: float = 0,
    tiling="none",
    lora_alpha: Union[float, List[float]] = 0,
    sampler_params={},
    callback=None,
):
    req_args = locals()

    try:
        images = []

        seed_everything(seed)
        precision_scope = torch.autocast if context.half_precision else nullcontext

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

        with precision_scope("cuda"):
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

        with torch.no_grad(), precision_scope("cuda"):
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
    control_alpha=None,
    prompt_strength: float = 0.8,
    # preserve_init_image_color_profile=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    # hypernetwork_strength: float = 0,
    lora_alpha: Union[float, List[float]] = 0,
    # sampler_params={},
    tiling="none",
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
    )

    from sdkit.models.model_loader.lora import apply_lora_model
    from sdkit.generate.sampler import diffusers_samplers
    import numpy as np

    prompt = prompt.lower()
    negative_prompt = negative_prompt.lower()

    model = context.models["stable-diffusion"]
    default_pipe = model["default"]
    if context.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator(context.device).manual_seed(seed)

    is_sd_xl = isinstance(
        default_pipe,
        (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline),
    )

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
        cmd["image"] = resize_img(init_image.convert("RGB"), width, height, clamp_to_64=True)
        cmd["strength"] = prompt_strength
    if init_image_mask:
        init_image_mask = get_image(init_image_mask)
        cmd["mask_image"] = resize_img(init_image_mask.convert("RGB"), width, height, clamp_to_64=True)

    if init_image:
        operation_to_apply = "inpainting" if init_image_mask else "img2img"
    else:
        operation_to_apply = "txt2img"

    if operation_to_apply not in model:
        if "inpainting" in model and len(model) == 1:
            raise RuntimeError(
                f"This model does not support {operation_to_apply}! This model requires an initial image and mask."
            )

        if control_image and isinstance(model["default"], StableDiffusionXLImg2ImgPipeline):
            raise RuntimeError(
                "ControlNet only supports text-to-image with SD-XL right now. Please remove the initial image and try again!"
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

            cmd["controlnet_conditioning_scale"] = control_alpha

        if operation_to_apply == "txt2img":
            cmd["image"] = control_image
        else:
            cmd["control_image"] = control_image

        if is_sd_xl:
            if operation_to_apply != "txt2img":
                raise Exception(
                    "ControlNet only supports text-to-image with SD-XL right now. Please remove the initial image and try again!"
                )

            operation_to_apply_cls = StableDiffusionXLControlNetPipeline
        else:
            controlnet_op = {
                "txt2img": StableDiffusionControlNetPipeline,
                "img2img": StableDiffusionControlNetImg2ImgPipeline,
                "inpainting": StableDiffusionControlNetInpaintPipeline,
            }
            operation_to_apply_cls = controlnet_op[operation_to_apply]

        operation_to_apply = operation_to_apply_cls(controlnet=controlnet, **default_pipe.components)

    if sampler_name.startswith("unipc_tu"):
        sampler_name = "unipc_tu_2" if num_inference_steps < 10 else "unipc_tu"

    operation_to_apply.scheduler = diffusers_samplers.make_sampler(sampler_name, model["default_scheduler"])
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

    if context.embeddings_path != None:
        load_embeddings(context, prompt, negative_prompt, operation_to_apply)

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
        cl._conv_forward = asymmetricConv2DConvForward.__get__(cl, torch.nn.Conv2d)

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
        operation_to_apply.unet._allocate_trt_buffers(operation_to_apply, context.device, dtype, width, height)

    # apply
    log.info(f"applying: {operation_to_apply}")
    log.info(f"Running on diffusers: {cmd}")

    images = operation_to_apply(**cmd).images

    if is_sd_xl and context.half_precision:  # cleanup - workaround since SDXL upcasts the vae
        operation_to_apply.vae = operation_to_apply.vae.to(dtype=torch.float16)

    if init_image_mask and strict_mask_border:
        images = blend_mask(images, init_image, init_image_mask, width, height)

    return images


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

    import numpy as np

    if init_image_mask != None:
        # Check if it has alpha channel, else make black transparent
        channel_count = np.array(init_image_mask).shape[2]
        if channel_count < 4:
            init_image_mask = black_to_transparent(init_image_mask)

        # Extract the mask from the alpha channel.
        composite_mask = init_image_mask.getchannel(3)
        composite_mask = resize_img(composite_mask, width, height)
        for i, img in enumerate(images):
            images[i] = Image.composite(img, init_image, composite_mask)
            images[i] = images[i].convert("RGB")

    return images


def load_embeddings(context, prompt, negative_prompt, default_pipe):
    import traceback

    pt_files = list(context.embeddings_path.rglob("*.pt"))
    bin_files = list(context.embeddings_path.rglob("*.bin"))
    st_files = list(context.embeddings_path.rglob("*.safetensors"))

    log.info("Applying Embeddings...")

    for filename in pt_files + bin_files + st_files:
        skip_embedding = False
        embeds_name = get_embedding_token(filename.name).lower()
        if (
            embeds_name not in prompt and embeds_name not in negative_prompt
        ) or embeds_name in context._loaded_embeddings:
            continue
        log.info(f"### Load: embedding {filename} ###")

        embedding = load_tensor_file(filename)
        dump_embedding_info(embedding)

        model_dim = default_pipe.text_encoder.get_input_embeddings().weight.data[0].shape[0]

        if "emb_params" in embedding.keys():
            if model_dim != embedding["emb_params"].size(dim=-1):
                skip_embedding = True
        elif "<concept>" in embedding.keys():
            if model_dim != embedding["<concept>"].size(dim=-1):
                skip_embedding = True
        elif "string_to_param" in embedding.keys():
            for trained_token in embedding["string_to_param"]:
                embeds = embedding["string_to_param"][trained_token]
                if model_dim != embeds.size(dim=-1):
                    skip_embedding = True
                    continue
        else:
            log.info(f"Embedding {filename} has an unknown internal structure. Trying to load it anyways.")

        if skip_embedding:
            log.info(
                f"Skipping embedding {filename}, due to incompatible embedding size, e.g. because this a StableDiffusion 2 embedding used with a StableDiffusion 1 model, or vice versa."
            )
        else:
            try:
                default_pipe.load_textual_inversion(filename, embeds_name)
            except:
                log.error(f"Embedding {filename} can't be loaded. Proceeding without it!")
                log.error(traceback.format_exc())
            else:
                context._loaded_embeddings.add(embeds_name)


def dump_embedding_info(embedding):
    for key in dict(embedding).keys():
        if key == "string_to_token":
            for s in dict(embedding[key]).keys():
                log.info(f"  - {key}: {s}")
        elif key == "string_to_param":
            for s in dict(embedding[key]).keys():
                log.info(f"  - {key}: {s}")
        elif key == "name" or key == "sd_checkpoint" or key == "sd_checkpoint_name" or key == "step":
            log.info(f"  - {key}: '{embedding[key]}'")
        else:
            log.info(f"  # {key}")


def get_image(img):
    if not isinstance(img, str):
        return img

    if img.startswith("data:image"):
        return base64_str_to_img(img)

    import os

    if os.path.exists(img):
        from PIL import Image

        return Image.open(img)
