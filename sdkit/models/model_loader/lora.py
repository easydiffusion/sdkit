import traceback

import torch

from sdkit import Context
from sdkit.utils import load_tensor_file, log


def load_model(context: Context, **kwargs):
    model = context.models["stable-diffusion"]
    default_pipe = model["default"]

    lora_model_path = context.model_paths.get("lora")
    lora_model_paths = lora_model_path if isinstance(lora_model_path, list) else [lora_model_path]

    loras = [load_tensor_file(path) for path in lora_model_paths]
    loras = [default_pipe._convert_kohya_lora_to_diffusers(lora) for lora in loras]

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
    state_dict, network_alpha = lora
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
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora.up", "lora.down"))

        if hasattr(curr_layer, "_hf_hook"):
            weight = curr_layer._hf_hook.weights_map["weight"]
        else:
            weight = curr_layer.weight

        # update weight
        rank = state_dict[pair_keys[0]].shape[-1]
        net_alpha = network_alpha / rank

        weight.data = weight.data.to(torch.float32)

        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32).to(weight.device)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32).to(weight.device)
            y = alpha * net_alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32).to(weight.device)
            weight_down = state_dict[pair_keys[1]].to(torch.float32).to(weight.device)
            y = alpha * net_alpha * torch.mm(weight_up, weight_down)

        weight.data += y
        weight.data = weight.data.to(precision)

        # update visited list
        for item in pair_keys:
            visited.append(item)
