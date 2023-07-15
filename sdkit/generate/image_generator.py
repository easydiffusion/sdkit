from contextlib import nullcontext

import torch
from pytorch_lightning import seed_everything
from tqdm import trange
from typing import Optional, List, Union

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
    prompt_strength: float = 0.8,
    preserve_init_image_color_profile=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    hypernetwork_strength: float = 0,
    tiling="none",
    lora_alpha: Union[float, List[float]] = [],
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
                prompt_strength,
                # preserve_init_image_color_profile,
                sampler_name,
                # hypernetwork_strength,
                lora_alpha,
                tiling,
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
    prompt_strength: float = 0.8,
    # preserve_init_image_color_profile=False,
    sampler_name: str = "euler_a",  # "ddim", "plms", "heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms",
    # "dpm_solver_stability", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde", "dpm_fast"
    # "dpm_adaptive"
    # hypernetwork_strength: float = 0,
    lora_alpha: Union[float, List[float]] = [],
    # sampler_params={},
    tiling="none",
    callback=None,
):
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
    )

    from sdkit.models.model_loader.lora import apply_lora_model
    import numpy as np

    prompt = prompt.lower()
    negative_prompt = negative_prompt.lower()

    model = context.models["stable-diffusion"]
    if context.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator(context.device).manual_seed(seed)

    cmd = {
        "guidance_scale": guidance_scale,
        "generator": generator,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": num_outputs,
    }
    if init_image:
        cmd["image"] = get_image(init_image).convert("RGB")
        cmd["image"] = resize_img(cmd["image"], width, height, clamp_to_64=True)
        cmd["strength"] = prompt_strength
    if init_image_mask:
        cmd["mask_image"] = get_image(init_image_mask).convert("RGB")
        cmd["mask_image"] = resize_img(cmd["mask_image"], width, height, clamp_to_64=True)

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

    operation_to_apply = model[operation_to_apply]
    if context.samplers.get(sampler_name) is None:
        raise NotImplementedError(f"The sampler '{sampler_name}' is not supported (yet)!")

    operation_to_apply.scheduler = context.samplers[sampler_name]
    log.info(f"Using sampler: {operation_to_apply.scheduler} because of {sampler_name}")

    if isinstance(operation_to_apply, StableDiffusionInpaintPipelineLegacy) or isinstance(
        operation_to_apply, StableDiffusionImg2ImgPipeline
    ):
        del cmd["width"]
        del cmd["height"]
    elif isinstance(operation_to_apply, StableDiffusionInpaintPipeline):
        del cmd["strength"]

    cmd["callback"] = lambda i, t, x_samples: callback(x_samples, i, operation_to_apply) if callback else None

    # apply the LoRA (if necessary)
    if context.models.get("lora"):
        log.info("Applying LoRA...")
        lora_alpha = lora_alpha if isinstance(lora_alpha, list) else [lora_alpha]
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
    conv_layers = []
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

    return operation_to_apply(**cmd).images


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
