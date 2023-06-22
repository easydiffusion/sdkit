import os
import tempfile
import traceback
from pathlib import Path
from urllib.parse import urlparse

import ldm.modules.attention
import ldm.modules.diffusionmodules.model
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.nn.functional import silu
from transformers import logging as tr_logging

from sdkit import Context
from sdkit.utils import (
    download_file,
    hash_file_quick,
    load_tensor_file,
    log,
    save_tensor_file,
)

tr_logging.set_verbosity_error()  # suppress unnecessary logging


def load_model(context: Context, scan_model=True, check_for_config_with_same_name=True, **kwargs):
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
        if config_file_path is None:
            # try using an SD 1.4 config
            from sdkit.models import get_model_info_from_db

            sd_v1_4_info = get_model_info_from_db(model_type="stable-diffusion", model_id="1.4")
            config_file_path = resolve_model_config_file_path(sd_v1_4_info, model_path)

        return load_diffusers_model(context, model_path, config_file_path)

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


def load_diffusers_model(context: Context, model_path, config_file_path):
    import torch
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
    )
    from compel import Compel

    from sdkit.generate.sampler import diffusers_samplers
    from sdkit.utils import gc

    from .convert_from_ckpt import download_from_original_stable_diffusion_ckpt

    from .optimizations import optimized_get_attention_scores, get_optimized_attentionblock_forward

    from diffusers.models.attention_processor import Attention
    from diffusers.models.attention import AttentionBlock

    Attention.get_attention_scores = optimized_get_attention_scores
    AttentionBlock.forward = get_optimized_attentionblock_forward

    log.info("loading on diffusers")

    log.info(f"using config: {config_file_path}")
    config = OmegaConf.load(config_file_path)
    config.model.params.unet_config.params.use_fp16 = context.half_precision

    extra_config = config.get("extra", {})
    attn_precision = extra_config.get("attn_precision", "fp16" if context.half_precision else "fp32")
    log.info(f"using attn_precision: {attn_precision}")

    # txt2img
    default_pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path=model_path,
        original_config_file=config_file_path,
        extract_ema=False,
        scheduler_type="ddim",
        from_safetensors=model_path.endswith(".safetensors"),
        upcast_attention=(attn_precision == "fp32"),
        is_img2img=False,
        device="cpu",
    )

    default_pipe.requires_safety_checker = False
    default_pipe.safety_checker = None

    save_tensor_file(default_pipe.vae.state_dict(), os.path.join(tempfile.gettempdir(), "sd-base-vae.safetensors"))

    if context.vram_usage_level == "low" and "cuda" in context.device:
        if context.half_precision:
            default_pipe = default_pipe.to("cpu", torch.float16, silence_dtype_warnings=True)
        default_pipe.enable_sequential_cpu_offload(gpu_id=torch.device(context.device).index)
    else:
        if context.half_precision:
            default_pipe = default_pipe.to(context.device, torch.float16)
        else:
            default_pipe = default_pipe.to(context.device)

    if context.vram_usage_level == "high":
        default_pipe.enable_attention_slicing(4)
    else:
        default_pipe.enable_attention_slicing(1)

    try:
        import xformers

        default_pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    if torch.__version__.startswith("2.") and hasattr(default_pipe, "enable_vae_slicing"):
        default_pipe.enable_vae_slicing()

    # make the compel prompt parser object
    compel = Compel(
        tokenizer=default_pipe.tokenizer,
        text_encoder=default_pipe.text_encoder,
        truncate_long_prompts=False,
        use_penultimate_clip_layer=context.clip_skip,
        device=context.device,
    )

    # make samplers
    diffusers_samplers.make_samplers(context, default_pipe.scheduler)

    if isinstance(default_pipe, StableDiffusionInpaintPipeline):
        log.info("Loaded on diffusers")
        return {
            "config": config,
            "default": default_pipe,
            "inpainting": default_pipe,
            "compel": compel,
        }

    pipe_txt2img = default_pipe

    # img2img
    pipe_img2img = StableDiffusionImg2ImgPipeline(
        vae=pipe_txt2img.vae,
        text_encoder=pipe_txt2img.text_encoder,
        tokenizer=pipe_txt2img.tokenizer,
        unet=pipe_txt2img.unet,
        scheduler=pipe_txt2img.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    # inpainting
    # TODO - use legacy only if not an Inpainting Model. confirm this.
    pipe_inpainting = StableDiffusionInpaintPipelineLegacy(
        vae=pipe_txt2img.vae,
        text_encoder=pipe_txt2img.text_encoder,
        tokenizer=pipe_txt2img.tokenizer,
        unet=pipe_txt2img.unet,
        scheduler=pipe_txt2img.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    gc(context)

    if context.embeddings_path != None:
        load_embeddings(context.embeddings_path, default_pipe)

    log.info("Loaded on diffusers")

    return {
        "config": config,
        "default": default_pipe,
        "txt2img": pipe_txt2img,
        "img2img": pipe_img2img,
        "inpainting": pipe_inpainting,
        "compel": compel,
    }


def load_embeddings(path, default_pipe):
    pt_files = list(path.rglob("*.pt"))
    bin_files = list(path.rglob("*.bin"))
    st_files = list(path.rglob("*.safetensors"))

    for filename in pt_files + bin_files + st_files:
        log.info(f"### Load embedding {filename} ###")
        skip_embedding = False
        embeds_name =  filename.name.split(".")[0]

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
        
        if "name" in embedding.keys():
            embeds_name = embedding["name"]

        if skip_embedding:
            log.info(f"Skipping embedding {filename}, due to incompatible embedding size, e.g. because this a StableDiffusion 2 embedding used with a StableDiffusion 1 model, or vice versa.")
        else:
            try:
                default_pipe.load_textual_inversion(filename, embeds_name)
            except:
                log.error(f"Embedding {filename} can't be loaded. Proceeding without it!")
                log.error(traceback.format_exc())


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


def test_and_fix_precision(context, model, config, attn_precision):
    prev_model = context.models.get("stable-diffusion")
    context.models["stable-diffusion"] = model

    # test precision
    try:
        from sdkit.generate import generate_images

        from . import optimizations

        images = generate_images(
            context, prompt="Horse", width=64, height=64, num_inference_steps=1, sampler_name="plms"
        )
        is_black_image = not images[0].getbbox()
        if is_black_image and attn_precision == "fp16":
            attn_precision = "fp32"
            log.info(f"trying attn_precision: {attn_precision}")
            ldm.modules.attention.CrossAttention.forward = optimizations.make_attn_forward(
                context, attn_precision=attn_precision
            )
            images = generate_images(context, prompt="Horse", width=64, height=64, num_inference_steps=1)
            is_black_image = not images[0].getbbox()

        if is_black_image and attn_precision == "fp32" and context.half_precision:
            log.info("trying full precision")
            context.orig_half_precision = context.half_precision
            context.half_precision = False
            config.model.params.unet_config.params.use_fp16 = False
            model = model.float()
    except:
        log.error(traceback.format_exc())

    context.models["stable-diffusion"] = prev_model


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
