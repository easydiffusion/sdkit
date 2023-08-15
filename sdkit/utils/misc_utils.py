from sdkit import Context


def make_sd_context(model_path: str = "models/stable-diffusion/sd-v1-4.ckpt", vram_usage_level: str = "balanced"):
    from sdkit.models import load_model

    context = Context()
    context.test_diffusers = True
    context.model_paths["stable-diffusion"] = model_path
    context.vram_usage_level = vram_usage_level

    load_model(context, "stable-diffusion")

    return context


def get_nested_attr(o, key):
    "Returns a nested attribute, accessed via dot-separators. E.g. `text_model.encoder.foo` returns the nested attribute `foo`"
    keys = key.split(".")
    curr_layer = getattr(o, keys.pop(0))

    if len(keys) == 0:
        return curr_layer

    temp_name = keys.pop(0)
    while len(keys) > -1:
        try:
            curr_layer = curr_layer.__getattr__(temp_name)
            if len(keys) > 0:
                temp_name = keys.pop(0)
            elif len(keys) == 0:
                break
        except Exception:
            if len(keys) == 0:
                raise Exception(f"Could not find {key} in the given object!")

            if len(temp_name) > 0:
                temp_name += "." + keys.pop(0)
            else:
                temp_name = keys.pop(0)

    return curr_layer
