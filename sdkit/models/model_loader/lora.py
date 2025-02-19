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

            x = self.alpha * alpha / rank
            if isinstance(x, torch.Tensor):
                x = x.to(y.device, dtype=y.dtype)
            y *= x

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

    model = context.models["stable-diffusion"]
    pipe = model["default"]
    sd_type = model["type"]

    loras = [(load_tensor_file(path), path) for path in lora_model_paths]
    loras = [load_lora(pipe, lora, sd_type, path) for lora, path in loras]

    return loras


def get_lora_type(lora):
    if (
        "text_encoder_2.text_model.encoder.layers.0.mlp.fc1.alpha" in lora
        or "unet.up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_q.down" in lora
    ):
        return "SDXL"

    if "text_encoder.text_model.encoder.layers.0.self_attn.q_proj.down" in lora:
        lora_dim = lora["text_encoder.text_model.encoder.layers.0.self_attn.q_proj.down"].shape[1]
        if lora_dim == 768:
            return "SD1"
        elif lora_dim == 1024:
            return "SD2"

    return "SD1"


def load_lora(pipe, lora, sd_type, lora_path):
    lora_blocks = {}
    lora = {_name(key): val for key, val in lora.items()}

    lora_type = get_lora_type(lora)
    if lora_type != sd_type:
        raise RuntimeError(
            f"Sorry, could not load {lora_path}. You're trying to use a {lora_type} LoRA model with a {sd_type} Stable Diffusion model. They're not compatible, please use a compatible model!"
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


def _name(key, unet_layers_per_block=2):
    diffusers_name = key
    diffusers_name = diffusers_name.replace("_", ".")
    diffusers_name = diffusers_name.replace("lora.unet.", "unet.")
    diffusers_name = diffusers_name.replace("lora.te.", "text_encoder.")
    diffusers_name = diffusers_name.replace("lora.te1.", "text_encoder.")
    diffusers_name = diffusers_name.replace("lora.te2.", "text_encoder_2.")
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
        # SDXL stuff
        diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
        diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
        diffusers_name = diffusers_name.replace("emb.layers.1", "time_emb_proj")
        diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

        diffusers_name = diffusers_name.replace("input.blocks.0.0.", "conv_in.")
        diffusers_name = diffusers_name.replace("out.2.", "conv_out.")

        # based on https://github.com/kohya-ss/sd-webui-additional-networks/blob/main/scripts/lora_compvis.py#L194
        parts = diffusers_name.split(".")
        if "input.blocks" in diffusers_name:
            if "op" in parts:
                idx = parts.index("op")
                layer_id = int(parts[idx - 2])
                block_id = (layer_id - 3) // (unet_layers_per_block + 1)
                diffusers_name = diffusers_name.replace(
                    f"input.blocks.{layer_id}.0.op", f"down_blocks.{block_id}.downsamplers.0.conv"
                )
            else:
                idx = parts.index("blocks")
                layer_id, net_type = int(parts[idx + 1]), int(parts[idx + 2])
                net_type_str = "attentions" if net_type == 1 else "resnets"
                block_id = (layer_id - 1) // (unet_layers_per_block + 1)
                layer_in_block_id = (layer_id - 1) % (unet_layers_per_block + 1)
                diffusers_name = diffusers_name.replace(
                    f"input.blocks.{layer_id}.{net_type}", f"down_blocks.{block_id}.{net_type_str}.{layer_in_block_id}"
                )
        elif "middle.block" in diffusers_name:
            idx = parts.index("block")
            net_type = int(parts[idx + 1])
            net_type_str = "resnets" if net_type % 2 == 0 else "attentions"
            block_id = net_type // 2
            diffusers_name = diffusers_name.replace(f"middle.block.{net_type}", f"mid_block.{net_type_str}.{block_id}")
        elif "output.blocks" in diffusers_name:
            if "conv" in parts:
                idx = parts.index("conv")
                layer_id, t = int(parts[idx - 2]), int(parts[idx - 1])
                block_id = (layer_id - 2) // (unet_layers_per_block + 1)
                diffusers_name = diffusers_name.replace(
                    f"output.blocks.{layer_id}.{t}.conv", f"up_blocks.{block_id}.upsamplers.0.conv"
                )
            else:
                idx = parts.index("blocks")
                layer_id, net_type = int(parts[idx + 1]), int(parts[idx + 2])
                net_type_str = "attentions" if net_type == 1 else "resnets"
                block_id = layer_id // (unet_layers_per_block + 1)
                layer_in_block_id = layer_id % (unet_layers_per_block + 1)
                diffusers_name = diffusers_name.replace(
                    f"output.blocks.{layer_id}.{net_type}", f"up_blocks.{block_id}.{net_type_str}.{layer_in_block_id}"
                )
        elif "time.embed" in diffusers_name:
            idx = parts.index("embed")
            layer_id = int(parts[idx + 1])
            block_id = int((layer_id + 2) / 2)
            diffusers_name = diffusers_name.replace(f"time.embed.{layer_id}", f"time_embedding.linear_{block_id}")

    elif diffusers_name.startswith("text_encoder"):
        diffusers_name = diffusers_name.replace("text.model", "text_model")
        diffusers_name = diffusers_name.replace("self.attn", "self_attn")
        diffusers_name = diffusers_name.replace("q.proj", "q_proj")
        diffusers_name = diffusers_name.replace("k.proj", "k_proj")
        diffusers_name = diffusers_name.replace("v.proj", "v_proj")
        diffusers_name = diffusers_name.replace("out.proj", "out_proj")
    return diffusers_name
