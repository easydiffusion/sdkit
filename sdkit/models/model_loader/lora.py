import torch
import traceback

from sdkit import Context
from sdkit.utils import load_tensor_file, log


def load_model(context: Context, **kwargs):
    lora_model_path = context.model_paths.get("lora")

    apply_lora_model(context, lora_model_path, direction=1)

    return lora_model_path


def move_model_to_cpu(context: Context):
    pass


def unload_model(context: Context, **kwargs):
    lora_model_path = context.models["lora"]
    apply_lora_model(context, lora_model_path, direction=-1)


def apply_lora_model(context, lora_model_path, direction):
    if not context.test_diffusers:
        return

    try:
        model = context.models["stable-diffusion"]
        default_pipe = model["default"]

        apply_lora(context, default_pipe, lora_path=lora_model_path, direction=direction)
    except:
        log.error(traceback.format_exc())
        log.error(f"Could not load LoRA: {lora_model_path}")


# Temporarily dumped from https://github.com/huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py
# Need to move this function into the `convert_from_ckpt.py` module (in diffusers), and use that instead.
def apply_lora(
    context,
    pipeline,
    lora_path,
    alpha=0.75,
    direction=1,
    lora_prefix_text_encoder="lora_te",
    lora_prefix_unet="lora_unet",
):
    state_dict = load_tensor_file(lora_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(lora_prefix_text_encoder + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            # "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight"
            # layer_infos = "down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v".split("_")

            layer_infos = key.split(".")[0].split(lora_prefix_unet + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32).to(context.device)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32).to(context.device)
            curr_layer.weight.data += direction * alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32).to(context.device)
            weight_down = state_dict[pair_keys[1]].to(torch.float32).to(context.device)
            curr_layer.weight.data += direction * alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
