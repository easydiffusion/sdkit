from . import stable_diffusion, gfpgan, realesrgan

models = {
    'stable-diffusion': stable_diffusion.models,
    'gfpgan': gfpgan.models,
    'realesrgan': realesrgan.models,
}

index = None

def get_model_info_from_db(quick_hash=None, model_type=None, model_id=None):
    if quick_hash:
        if index is None:
            rebuild_index()

        return index.get(quick_hash)
    elif model_id and model_type:
        m = models.get(model_type, {})
        return m.get(model_id)

def rebuild_index():
    global index

    index = {}
    for _, m in models.items():
        module_index = {info.get('quick_hash'): info for _, info in m.items()}
        index.update(module_index)
