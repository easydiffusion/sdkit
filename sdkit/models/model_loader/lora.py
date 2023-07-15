import traceback

import torch

from sdkit import Context
from sdkit.utils import load_tensor_file, log

LORA_MULTIPLIER = 2.0
TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"


def load_model(context: Context, **kwargs):
    lora_model_path = context.model_paths.get("lora")
    lora_model_paths = lora_model_path if isinstance(lora_model_path, list) else [lora_model_path]

    loras = [load_tensor_file(path) for path in lora_model_paths]
    loras = [_convert_kohya_lora_to_diffusers(lora) for lora in loras]

    return loras


def move_model_to_cpu(context: Context):
    pass


def unload_model(context: Context, **kwargs):
    if hasattr(context, "_last_lora_alpha"):
        removal_alpha = -1 * context._last_lora_alpha
        del context._last_lora_alpha
        apply_lora_model(context, alpha=removal_alpha)


def apply_lora_model(context, alpha):
    if not context.test_diffusers:
        return

    try:
        model = context.models["stable-diffusion"]
        default_pipe = model["default"]

        _apply_lora(context, default_pipe, alpha)
    except:
        log.error(traceback.format_exc())
        log.error("Could not apply LoRA!")


# Inspired from https://github.com/huggingface/diffusers/blob/main/scripts/convert_lora_safetensor_to_diffusers.py
def _apply_lora(context, pipeline, alphas):
    log.info(f"Applying lora, alphas: {alphas}")

    loras = context.models["lora"]
    assert len(loras) == len(alphas)

    precision = torch.float16 if context.half_precision else torch.float32

    for lora, alpha in zip(loras, alphas):
        _apply_single_lora(pipeline, lora, alpha, precision)


def _apply_single_lora(pipeline, lora, alpha, precision):
    state_dict = lora
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        module_chain = key.replace("attn1.processor", "attn1").replace("attn2.processor", "attn2")
        module_chain = module_chain.replace("_lora.down.weight", "").replace("_lora.up.weight", "")

        if module_chain.startswith("unet"):
            module_chain = module_chain.replace("to_out", "to_out.0")
        elif module_chain.startswith("text_encoder"):
            module_chain = module_chain.replace("to_k", "k_proj")
            module_chain = module_chain.replace("to_q", "q_proj")
            module_chain = module_chain.replace("to_v", "v_proj")
            module_chain = module_chain.replace("to_out", "out_proj")

        module_chain = module_chain.split(".")

        curr_layer = getattr(pipeline, module_chain.pop(0))

        # find the target layer
        temp_name = module_chain.pop(0)
        while len(module_chain) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(module_chain) > 0:
                    temp_name = module_chain.pop(0)
                elif len(module_chain) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "." + module_chain.pop(0)
                else:
                    temp_name = module_chain.pop(0)

        pair_keys = []
        if "lora.down" in key:
            pair_keys.append(key.replace("lora.down", "lora.up"))
            pair_keys.append(key)
            pair_keys.append(key.replace("lora.down.weight", "lora.alpha"))
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora.up", "lora.down"))
            pair_keys.append(key.replace("lora.up.weight", "lora.alpha"))

        if hasattr(curr_layer, "_hf_hook"):
            weight = curr_layer._hf_hook.weights_map["weight"]
        else:
            weight = curr_layer.weight

        # update weight
        # based on a mix of ideas from diffusers and automatic1111
        up = state_dict[pair_keys[0]].to(weight.device, dtype=torch.float32)
        down = state_dict[pair_keys[1]].to(weight.device, dtype=torch.float32)
        local_alpha = state_dict.get(pair_keys[2], 1.0)
        rank = up.shape[1]

        weight.data = weight.data.to(torch.float32)

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            up = up.squeeze(2).squeeze(2)
            down = down.squeeze(2).squeeze(2)
            y = torch.mm(up, down).unsqueeze(2).unsqueeze(3)
        elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
            y = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
        else:
            y = torch.mm(up, down)

        y *= alpha * LORA_MULTIPLIER * local_alpha / rank

        weight.data += y
        weight.data = weight.data.to(precision)

        # update visited list
        for item in pair_keys:
            visited.append(item)


# copied from diffusers/loaders.py, to test changes before proposing an upstream fix PR
def _convert_kohya_lora_to_diffusers(state_dict):
    unet_state_dict = {}
    te_state_dict = {}

    for key, value in state_dict.items():
        if "lora_down" in key:
            lora_name = key.split(".")[0]
            lora_name_up = lora_name + ".lora_up.weight"
            lora_name_alpha = lora_name + ".alpha"
            alpha = 1
            if lora_name_alpha in state_dict:
                alpha = state_dict[lora_name_alpha].item()

            if lora_name.startswith("lora_unet_"):
                diffusers_name = key.replace("lora_unet_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
                diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
                diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                if "transformer_blocks" in diffusers_name:
                    if "attn1" in diffusers_name or "attn2" in diffusers_name:
                        diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                        diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                        unet_state_dict[diffusers_name] = value
                        unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]
                        unet_state_dict[diffusers_name.replace(".down.weight", ".alpha")] = alpha
            elif lora_name.startswith("lora_te_"):
                diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name] = value
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]
                    te_state_dict[diffusers_name.replace(".down.weight", ".alpha")] = alpha

    unet_state_dict = {f"{UNET_NAME}.{module_name}": params for module_name, params in unet_state_dict.items()}
    te_state_dict = {f"{TEXT_ENCODER_NAME}.{module_name}": params for module_name, params in te_state_dict.items()}
    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict
