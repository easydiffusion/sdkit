import os
import tempfile
import traceback
from pathlib import Path
from urllib.parse import urlparse
import math

import ldm.modules.attention
import ldm.modules.diffusionmodules.model
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.nn.functional import silu
from transformers import logging as tr_logging

from sdkit import Context
from sdkit.utils import download_file, hash_file_quick, load_tensor_file, log, save_tensor_file, is_cpu_device

tr_logging.set_verbosity_error()  # suppress unnecessary logging


def load_model(
    context: Context,
    scan_model=True,
    check_for_config_with_same_name=True,
    clip_skip=False,
    convert_to_tensorrt=False,
    trt_build_config={"batch_size_range": (1, 1), "dimensions_range": [(768, 1024)]},
    **kwargs,
):
    from sdkit.models import scan_model as scan_model_fn

    from . import optimizations

    if hasattr(context, "orig_half_precision"):
        context.half_precision = context.orig_half_precision
        del context.orig_half_precision

    model_path = context.model_paths.get("stable-diffusion")
    config_file_path = get_model_config_file(context, check_for_config_with_same_name)

    if scan_model:
        scan_result = scan_model_fn(model_path)
        if scan_result.issues_count > 0 or scan_result.infected_files > 0:
            raise Exception(f"Model scan failed! Potentially infected model: {model_path}")

    if context.test_diffusers:
        from sdkit.models import get_model_info_from_db

        sd = load_tensor_file(model_path)
        sd = sd["state_dict"] if "state_dict" in sd else sd

        if config_file_path is None:
            if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias" in sd:  # SDXL Base
                info = get_model_info_from_db(model_type="stable-diffusion", model_id="sd-xl-base-1.0")
                config_file_path = resolve_model_config_file_path(info, model_path)
            elif "conditioner.embedders.0.model.transformer.resblocks.9.mlp.c_proj.bias" in sd:  # SDXL Refiner
                info = get_model_info_from_db(model_type="stable-diffusion", model_id="sd-xl-refiner-1.0")
                config_file_path = resolve_model_config_file_path(info, model_path)
            elif (
                "model.diffusion_model.input_blocks.0.0.weight" in sd
                and sd["model.diffusion_model.input_blocks.0.0.weight"].shape[1] == 9
            ):  # inpainting
                if "cond_stage_model.model.transformer.resblocks.7.mlp.c_proj.bias" in sd:
                    info = get_model_info_from_db(model_type="stable-diffusion", model_id="2.0-512-inpainting-ema")
                    config_file_path = resolve_model_config_file_path(info, model_path)
                else:
                    info = get_model_info_from_db(model_type="stable-diffusion", model_id="1.5-inpainting")
                    config_file_path = resolve_model_config_file_path(info, model_path)
            elif "cond_stage_model.model.transformer.resblocks.14.mlp.c_proj.weight" in sd:  # SD 2
                info = get_model_info_from_db(model_type="stable-diffusion", model_id="2.0-512-base-ema")
                config_file_path = resolve_model_config_file_path(info, model_path)
                # first use the 2.0 config, then test whether to use v-prediction or not
                # after the unet is ready

        if config_file_path is None:
            # try using an SD 1.4 config
            sd_v1_4_info = get_model_info_from_db(model_type="stable-diffusion", model_id="1.4")
            config_file_path = resolve_model_config_file_path(sd_v1_4_info, model_path)

        return load_diffusers_model(
            context, sd, model_path, config_file_path, clip_skip, convert_to_tensorrt, trt_build_config
        )

    # load the model file
    sd = load_tensor_file(model_path)
    sd = sd["state_dict"] if "state_dict" in sd else sd

    if is_lora(sd):
        raise Exception("The model file doesn't contain a model's checkpoint. Instead, it seems to be a LORA file.")

    # try to guess the config, if no config file was given
    # check if a key specific to SD 2.0 is missing
    if config_file_path is None and "cond_stage_model.model.ln_final.bias" not in sd.keys():
        # try using an SD 1.4 config
        from sdkit.models import get_model_info_from_db

        sd_v1_4_info = get_model_info_from_db(model_type="stable-diffusion", model_id="1.4")
        config_file_path = resolve_model_config_file_path(sd_v1_4_info, model_path)

    # load the config
    if config_file_path is None:
        raise Exception(
            'Unknown model! No config file path specified in context.model_configs for the "stable-diffusion" model!'
        )

    log.info(f"using config: {config_file_path}")
    config = OmegaConf.load(config_file_path)
    config.model.params.unet_config.params.use_fp16 = context.half_precision

    extra_config = config.get("extra", {})
    attn_precision = extra_config.get("attn_precision", "fp16" if context.half_precision else "fp32")
    log.info(f"using attn_precision: {attn_precision}")

    # instantiate the model
    model = instantiate_from_config(config.model)
    _, _ = model.load_state_dict(sd, strict=False)

    model = model.half() if context.half_precision else model.float()

    optimizations.send_to_device(context, model)
    model.eval()
    del sd

    # optimize CrossAttention.forward() for faster performance, and lower VRAM usage
    ldm.modules.attention.CrossAttention.forward = optimizations.make_attn_forward(
        context, attn_precision=attn_precision
    )
    ldm.modules.diffusionmodules.model.nonlinearity = silu

    test_and_fix_precision(context, model, config, attn_precision)

    # save the model vae into a temp folder (used for restoring the default VAE, if a custom VAE is unloaded)
    save_tensor_file(
        model.first_stage_model.state_dict(), os.path.join(tempfile.gettempdir(), "sd-base-vae.safetensors")
    )

    # optimizations.print_model_size_breakdown(model)

    return model


def unload_model(context: Context, **kwargs):
    context.module_in_gpu = None  # don't keep a dangling reference, prevents gc


def load_diffusers_model(
    context: Context, state_dict, model_path, config_file_path, clip_skip, convert_to_tensorrt, trt_build_config
):
    import torch
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    )
    from diffusers.models.attention_processor import Attention
    from compel import Compel, DiffusersTextualInversionManager, ReturnedEmbeddingsType as Skip
    import platform

    from sdkit.generate.sampler import diffusers_samplers
    from sdkit.utils import gc, has_amd_gpu
    import torch.nn.functional as F

    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    log.info("loading on diffusers")

    log.info(f"using config: {config_file_path}")
    config = OmegaConf.load(config_file_path)
    if hasattr(config.model.params, "unet_config"):  # TODO is this config change really necessary?
        config.model.params.unet_config.params.use_fp16 = context.half_precision

    is_sd_xl = hasattr(config.model.params, "network_config")
    if "model.diffusion_model.input_blocks.0.0.weight" in state_dict:
        is_inpainting = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape[1] == 9
    else:
        is_inpainting = False

    # fix for NAI keys
    check_and_fix_nai_keys(state_dict)

    extra_config = config.get("extra", {})
    attn_precision = extra_config.get("attn_precision", "fp16" if context.half_precision else "fp32")
    log.info(f"using attn_precision: {attn_precision}")

    model_load_params = {
        "extract_ema": False,
        "scheduler_type": "ddim",
        "from_safetensors": model_path.endswith(".safetensors"),
        "upcast_attention": (attn_precision == "fp32"),
        "device": "cpu",
        "load_safety_checker": False,
        "original_config_file": config_file_path,
    }

    if is_inpainting:
        model_load_params["pipeline_class"] = StableDiffusionInpaintPipeline
    elif is_sd_xl:
        if config.model.params.network_config.params.context_dim == 2048:
            model_load_params["pipeline_class"] = StableDiffusionXLPipeline
        else:
            model_load_params["pipeline_class"] = StableDiffusionXLImg2ImgPipeline

    # optimize for TRT or DirectML (AMD on Windows)
    model_component, _ = os.path.splitext(model_path)
    model_trt_path = model_component + ".trt"
    unet_onnx_path = model_component + ".unet.onnx"

    use_directml = platform.system() == "Windows" and has_amd_gpu()
    try:
        from importlib.metadata import version

        version("onnxruntime-directml")  # check if this is installed
    except:
        use_directml = False

    if is_cpu_device(context.torch_device):
        convert_to_tensorrt = False

    # remove SDPA if torch 2.0 and need to convert to ONNX
    needs_onnx = convert_to_tensorrt or (
        use_directml and (not os.path.exists(unet_onnx_path) or os.stat(unet_onnx_path).st_size == 0)
    )
    swap_sdpa = needs_onnx and hasattr(F, "scaled_dot_product_attention")
    old_sdpa = getattr(F, "scaled_dot_product_attention", None) if swap_sdpa else None
    if swap_sdpa:
        delattr(F, "scaled_dot_product_attention")

    # txt2img
    default_pipe = download_from_original_stable_diffusion_ckpt(state_dict, **model_load_params)

    if swap_sdpa and old_sdpa:
        setattr(F, "scaled_dot_product_attention", old_sdpa)

    default_pipe.requires_safety_checker = False
    default_pipe.safety_checker = None

    model_type = "SD1"
    if is_sd_xl:
        model_type = "SDXL"
    else:
        context_dim = default_pipe.text_encoder.get_input_embeddings().weight.data[0].shape[0]
        if context_dim == 768:
            model_type = "SD1"
        elif context_dim == 1024:
            model_type = "SD2"

    if is_sd_xl:
        # until the image artifacts go away: https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/discussions/31
        default_pipe.watermark.apply_watermark = lambda images: images

    if use_directml and (not os.path.exists(unet_onnx_path) or os.stat(unet_onnx_path).st_size == 0):
        from sdkit.utils import gc, convert_pipeline_unet_to_onnx

        log.info("Converting UNet to ONNX to run on AMD on Windows..")
        convert_pipeline_unet_to_onnx(default_pipe, unet_onnx_path, device="cpu", fp16=False)  # on cpu, so fp32
        log.info("Converted UNet to ONNX to run on AMD on Windows!")
    elif convert_to_tensorrt:
        from sdkit.utils import gc, convert_pipeline_to_tensorrt

        default_pipe = default_pipe.to(context.torch_device, torch.float16 if context.half_precision else torch.float32)

        batch_size_range = trt_build_config["batch_size_range"]
        dimensions_range = trt_build_config["dimensions_range"]

        log.info("Converting model to TensorRT for acceleration..")
        convert_pipeline_to_tensorrt(
            default_pipe, model_trt_path, batch_size_range, dimensions_range, fp16=context.half_precision
        )
        log.info("Converted model to TensorRT for acceleration!")

        default_pipe = default_pipe.to("cpu", torch.float32)

    # keep the VAE for future use (maybe use copy.deepcopy() instead of a file)
    save_tensor_file(default_pipe.vae.state_dict(), os.path.join(tempfile.gettempdir(), "sd-base-vae.safetensors"))

    # memory optimizations

    if context.vram_usage_level == "low" and not is_cpu_device(context.torch_device):
        if context.half_precision:
            default_pipe = default_pipe.to("cpu", torch.float16, silence_dtype_warnings=True)
        default_pipe.enable_sequential_cpu_offload(device=context.torch_device)
    else:
        if context.half_precision:
            default_pipe = default_pipe.to(context.torch_device, torch.float16)
        else:
            default_pipe = default_pipe.to(context.torch_device)

    if context.vram_usage_level == "high":
        default_pipe.enable_attention_slicing(4)
    else:
        default_pipe.enable_attention_slicing(1)

    try:
        import xformers

        default_pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    if hasattr(default_pipe, "enable_vae_slicing"):
        default_pipe.enable_vae_slicing()
    if hasattr(default_pipe, "enable_vae_tiling"):
        default_pipe.enable_vae_tiling()

    # /memory optimizations

    scheduler_config = dict(default_pipe.scheduler.config)

    # if SD 2, test whether to use 'v' prediction mode
    if model_type == "SD2" and not is_inpainting:
        # idea based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/d04e3e921e8ee71442a1f4a1d6e91c05b8238007

        dtype = torch.float16 if context.half_precision else torch.float32

        text_hidden_size = default_pipe.text_encoder.config.hidden_size
        in_channels = default_pipe.unet.config.in_channels

        test_embeds = torch.ones((1, 2, text_hidden_size), device=context.torch_device, dtype=dtype) * 0.5
        test_x = torch.ones((1, in_channels, 8, 8), device=context.torch_device, dtype=dtype) * 0.5
        t = torch.asarray([999], device=context.torch_device, dtype=dtype)

        noise_pred = default_pipe.unet(test_x, t, encoder_hidden_states=test_embeds, return_dict=False)[0]
        out = (noise_pred - test_x).mean().item()

        if math.isnan(out):
            log.info("Probably black images, trying fp32 attention precision")
            for m in default_pipe.unet.modules():
                if not isinstance(m, Attention):
                    continue
                m.upcast_attention = True

            noise_pred = default_pipe.unet(test_x, t, encoder_hidden_states=test_embeds, return_dict=False)[0]
            out = (noise_pred - test_x).mean().item()

        v_prediction_type = out < -1

        scheduler_config["prediction_type"] = "v_prediction" if v_prediction_type else "epsilon"
        log.info(f"Using {scheduler_config['prediction_type']} parameterization")

    # make the compel prompt parser object
    textual_inversion_manager = DiffusersTextualInversionManager(default_pipe)
    if is_sd_xl:
        skip = Skip.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        compel = Compel(
            tokenizer=[default_pipe.tokenizer, default_pipe.tokenizer_2],
            text_encoder=[default_pipe.text_encoder, default_pipe.text_encoder_2],
            truncate_long_prompts=False,
            returned_embeddings_type=skip,
            device=context.torch_device,
            # textual_inversion_manager=textual_inversion_manager, # SD XL doesn't support embeddings (yet)
            requires_pooled=[False, True],
        )
    else:
        skip = Skip.PENULTIMATE_HIDDEN_STATES_NORMALIZED if clip_skip else Skip.LAST_HIDDEN_STATES_NORMALIZED
        compel = Compel(
            tokenizer=default_pipe.tokenizer,
            text_encoder=default_pipe.text_encoder,
            truncate_long_prompts=False,
            returned_embeddings_type=skip,
            device=context.torch_device,
            textual_inversion_manager=textual_inversion_manager,
        )

    # load the TensorRT or DirectML unet, if present
    if use_directml and os.path.exists(unet_onnx_path) and os.stat(unet_onnx_path).st_size > 0:
        from .accelerators import apply_directml_unet

        apply_directml_unet(default_pipe, unet_onnx_path)
        log.info("Using DirectML accelerated UNet")
    elif convert_to_tensorrt and os.path.exists(model_trt_path):
        from .accelerators import apply_tensorrt

        apply_tensorrt(default_pipe, model_trt_path)

    model = {
        "config": config,
        "default": default_pipe,
        "compel": compel,
        "default_scheduler_config": scheduler_config,
        "type": model_type,
        "params": {
            "clip_skip": clip_skip,
            "convert_to_tensorrt": convert_to_tensorrt,
            "trt_build_config": trt_build_config,
        },
    }

    if hasattr(config, "model") and hasattr(config.model, "target") and "LatentInpaintDiffusion" in config.model.target:
        log.info("Loaded on diffusers")
        model["inpainting"] = StableDiffusionInpaintPipeline(**default_pipe.components)

        return model

    pipe_txt2img = default_pipe

    # img2img
    if isinstance(default_pipe, StableDiffusionXLPipeline):
        pipe_img2img = StableDiffusionXLImg2ImgPipeline(**default_pipe.components)
    elif isinstance(default_pipe, StableDiffusionXLImg2ImgPipeline):
        pipe_img2img = default_pipe
        pipe_txt2img = StableDiffusionXLPipeline(**default_pipe.components)
    else:
        pipe_img2img = StableDiffusionImg2ImgPipeline(**default_pipe.components)

    if is_sd_xl:
        # until the image artifacts go away: https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9/discussions/31
        pipe_img2img.watermark.apply_watermark = lambda images: images

    # inpainting
    if isinstance(default_pipe, (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline)):
        pipe_inpainting = StableDiffusionXLInpaintPipeline(**default_pipe.components)
        pipe_inpainting.watermark.apply_watermark = lambda images: images
    else:
        inpaint_legacy_components = {}
        for key in ("vae", "text_encoder", "tokenizer", "unet", "scheduler", "safety_checker", "feature_extractor"):
            inpaint_legacy_components[key] = getattr(default_pipe, key)

        pipe_inpainting = StableDiffusionInpaintPipelineLegacy(**inpaint_legacy_components)

    model["txt2img"] = pipe_txt2img
    model["img2img"] = pipe_img2img
    model["inpainting"] = pipe_inpainting
    model["default_scheduler_config"] = scheduler_config

    if isinstance(default_pipe, StableDiffusionXLImg2ImgPipeline):
        del model["txt2img"]
        del model["inpainting"]

    gc(context)

    log.info("Loaded on diffusers")

    context._loaded_embeddings = set(())

    return model


def test_and_fix_precision(context, model, config, attn_precision):
    prev_model = context.models.get("stable-diffusion")
    prev_lora_alpha = getattr(context, "_last_lora_alpha", [0])
    context.models["stable-diffusion"] = model
    if hasattr(context, "_last_lora_alpha"):
        import numpy as np

        lora_models = context.models.get("lora", [0])
        context._last_lora_alpha = np.array([0] * len(lora_models))

    # test precision
    try:
        import torch
        from sdkit.generate import generate_images
        from diffusers.models.attention_processor import Attention

        from . import optimizations

        if context.test_diffusers and "txt2img" not in model:
            raise Exception("Skipping precision test for a non-txt2img model. This is not a problem.")

        images = generate_images(
            context, prompt="Horse", width=64, height=64, num_inference_steps=1, sampler_name="euler_a"
        )
        is_black_image = not images[0].getbbox()
        if is_black_image and attn_precision == "fp16":
            attn_precision = "fp32"
            log.info(f"trying attn_precision: {attn_precision}")
            if context.test_diffusers:
                pipe = model["default"]
                for m in pipe.unet.modules():
                    if not isinstance(m, Attention):
                        continue
                    m.upcast_attention = True
            else:
                ldm.modules.attention.CrossAttention.forward = optimizations.make_attn_forward(
                    context, attn_precision=attn_precision
                )
            images = generate_images(
                context, prompt="Horse", width=64, height=64, num_inference_steps=1, sampler_name="euler_a"
            )
            is_black_image = not images[0].getbbox()

        if is_black_image and attn_precision == "fp32" and context.half_precision:
            log.info("trying full precision")
            context.orig_half_precision = context.half_precision
            context.half_precision = False

            if context.test_diffusers:
                pipe = model["default"]
                pipe = pipe.to(torch_dtype=torch.float32)
            else:
                config.model.params.unet_config.params.use_fp16 = False
                model = model.float()

    except:
        log.error(traceback.format_exc())

    context.models["stable-diffusion"] = prev_model
    if hasattr(context, "_last_lora_alpha"):
        context._last_lora_alpha = prev_lora_alpha


def get_model_config_file(context: Context, check_for_config_with_same_name):
    from sdkit.models import get_model_info_from_db

    if context.model_configs.get("stable-diffusion") is not None:
        return context.model_configs["stable-diffusion"]

    model_path = context.model_paths["stable-diffusion"]

    if check_for_config_with_same_name:
        model_name_path = os.path.splitext(model_path)[0]
        model_config_path = f"{model_name_path}.yaml"
        if os.path.exists(model_config_path):
            return model_config_path

    quick_hash = hash_file_quick(model_path)
    model_info = get_model_info_from_db(quick_hash=quick_hash)

    return resolve_model_config_file_path(model_info, model_path)


def resolve_model_config_file_path(model_info, model_path):
    if model_info is None:
        return
    config_url = model_info.get("config_url")
    if config_url is None:
        return

    if config_url.startswith("http"):
        config_file_name = os.path.basename(urlparse(config_url).path)
        model_dir_name = os.path.dirname(model_path)
        config_file_path = os.path.join(model_dir_name, config_file_name)

        if not os.path.exists(config_file_path):
            download_file(config_url, config_file_path)
    else:
        from sdkit.models import models_db

        models_db_path = Path(models_db.__file__).parent
        config_file_path = models_db_path / config_url

    return config_file_path


def is_lora(sd):
    heads = list(set([s[:5] for s in sd.keys()]))
    # for h in heads:
    #    log.info(f"Header '{h}'")
    return "lora_" in heads and "first" not in heads and "cond_" not in heads


def check_and_fix_nai_keys(state_dict):
    NAI_KEYS = (
        "cond_stage_model.transformer.embeddings.",
        "cond_stage_model.transformer.encoder.",
        "cond_stage_model.transformer.final_layer_norm.",
    )
    nai_keys = [k for k in state_dict if k.startswith(NAI_KEYS)]
    for old_key in nai_keys:
        new_key = old_key.replace(".transformer.", ".transformer.text_model.")
        state_dict[new_key] = state_dict[old_key]
        del state_dict[old_key]
