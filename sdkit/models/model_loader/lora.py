import traceback

import torch
import torch.nn as nn

from sdkit import Context
from sdkit.utils import load_tensor_file, log, get_nested_attr

from dataclasses import dataclass


@dataclass
class LoraBlock:
    block_name: str
    module: nn.Module = None
    up: torch.Tensor = None
    down: torch.Tensor = None
    alpha: float = 1.0

    @property
    def rank(self):
        return self.up.shape[1]

    def apply(self, alpha):
        try:
            weight = self._get_weight()

            # mix of ideas from diffusers and automatic1111
            up = self.up.to(weight.device, dtype=torch.float32)
            down = self.down.to(weight.device, dtype=torch.float32)
            rank = up.shape[1]

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                up = up.squeeze(2).squeeze(2)
                down = down.squeeze(2).squeeze(2)
                y = torch.mm(up, down).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                y = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                y = torch.mm(up, down)

            y *= self.alpha * alpha / rank

            weight.data += y
        except Exception as e:
            log.error(f"Unable to apply {alpha} to {self.block_name}")
            raise e

        return y

    def _get_weight(self):
        if hasattr(self.module, "_hf_hook"):
            weight = self.module._hf_hook.weights_map["weight"]
        else:
            weight = self.module.weight

        return weight.data


def load_model(context: Context, **kwargs):
    lora_model_path = context.model_paths.get("lora")
    lora_model_paths = lora_model_path if isinstance(lora_model_path, list) else [lora_model_path]

    pipe = context.models["stable-diffusion"]["default"]
    sd_config = context.models["stable-diffusion"]["config"]

    loras = [load_tensor_file(path) for path in lora_model_paths]
    loras = [load_lora(pipe, lora, sd_config) for lora in loras]

    return loras


def load_lora(pipe, lora, sd_config):
    context_dim = sd_config.model.params.get("unet_config", {}).get("params", {}).get("context_dim", None)
    if sd_config.model.params.get("network_config", {}).get("params", {}).get("context_dim", None):
        context_dim = 2048

    lora_blocks = {}
    lora = {_name(key): val for key, val in lora.items()}

    if "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.down" in lora:  # check for SD compatibility
        lora_dim = lora["text_encoder.text_model.encoder.layers.0.self_attn.q_proj.down"].shape[1]
        if lora_dim != context_dim:
            raise RuntimeError(
                f"Sorry, you're trying to use a {get_sd_type_from_dim(lora_dim)} LoRA model with a {get_sd_type_from_dim(context_dim)} Stable Diffusion model. They're not compatible, please use a compatible model!"
            )

    is_lycoris = any("lora.mid" in key for key in lora.keys())

    for key, val in lora.items():
        block_name = ".".join(key.split(".")[:-1])
        if block_name in lora_blocks:
            block = lora_blocks[block_name]
        else:
            block = LoraBlock(block_name)

            try:
                block.module = get_nested_attr(pipe, block_name)
            except Exception as e:
                if is_lycoris:
                    log.warn(f"Skipping layer {key}, since we don't fully support LyCORIS models yet!")
                    continue
                raise e  # otherwise die

            lora_blocks[block_name] = block

        attr = key.split(".")[-1]
        setattr(block, attr, val)

    if is_lycoris:
        log.warn(
            "LyCORIS (LoCon/LoHA) models are not fully supported yet! They will work partially, but the images may not be perfect."
        )

    return lora_blocks


def move_model_to_cpu(context: Context):
    pass


def unload_model(context: Context, **kwargs):
    if hasattr(context, "_last_lora_alpha"):
        removal_alpha = -1 * context._last_lora_alpha
        del context._last_lora_alpha
        apply_lora_model(context, removal_alpha)


def apply_lora_model(context, alphas):
    if not context.test_diffusers:
        return

    log.info(f"Applying lora, alphas: {alphas}")

    try:
        loras = context.models["lora"]
        if len(loras) != len(alphas):
            traceback.print_stack()
            raise RuntimeError(f"{len(loras)} != {len(alphas)}")

        for lora, alpha in zip(loras, alphas):
            if abs(alpha) < 0.0001:  # alpha is too small, not applying
                continue
            for block in lora.values():
                block.apply(alpha)
    except Exception as e:
        log.error(traceback.format_exc())
        raise e


def _name(key):
    diffusers_name = key.replace("lora_unet_", "unet.").replace("lora_te_", "text_encoder.")
    diffusers_name = diffusers_name.replace("_", ".").replace("text.encoder", "text_encoder")
    diffusers_name = diffusers_name.replace("lora.down", "down")
    diffusers_name = diffusers_name.replace("lora.up", "up")
    diffusers_name = diffusers_name.replace("down.weight", "down").replace("up.weight", "up")
    if diffusers_name.startswith("unet."):
        diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
        diffusers_name = diffusers_name.replace("mid.block", "mid_block")
        diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
        diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
        diffusers_name = diffusers_name.replace("to.q", "to_q")
        diffusers_name = diffusers_name.replace("to.k", "to_k")
        diffusers_name = diffusers_name.replace("to.v", "to_v")
        diffusers_name = diffusers_name.replace("to.out.0", "to_out.0")
        diffusers_name = diffusers_name.replace("proj.in", "proj_in")
        diffusers_name = diffusers_name.replace("proj.out", "proj_out")
        diffusers_name = diffusers_name.replace("conv.shortcut", "conv_shortcut")
        diffusers_name = diffusers_name.replace("time.emb.proj", "time_emb_proj")
        diffusers_name = diffusers_name.replace("conv.in", "conv_in")
        diffusers_name = diffusers_name.replace("conv.out", "conv_out")
        diffusers_name = diffusers_name.replace("conv.norm.out", "conv_norm_out")
        diffusers_name = diffusers_name.replace("time.proj", "time_proj")
        diffusers_name = diffusers_name.replace("time.embedding", "time_embedding")
        diffusers_name = diffusers_name.replace("add.embedding", "add_embedding")
        diffusers_name = diffusers_name.replace("class.embedding", "class_embedding")
        diffusers_name = diffusers_name.replace("encoder.hid.proj", "encoder_hid_proj")
    elif diffusers_name.startswith("text_encoder."):
        diffusers_name = diffusers_name.replace("text.model", "text_model")
        diffusers_name = diffusers_name.replace("self.attn", "self_attn")
        diffusers_name = diffusers_name.replace("q.proj", "q_proj")
        diffusers_name = diffusers_name.replace("k.proj", "k_proj")
        diffusers_name = diffusers_name.replace("v.proj", "v_proj")
        diffusers_name = diffusers_name.replace("out.proj", "out_proj")
    return diffusers_name


def get_sd_type_from_dim(dim: int) -> str:
    dims = {768: "SD 1", 1024: "SD 2", 2048: "SDXL"}
    return dims.get(dim, "Unknown")
