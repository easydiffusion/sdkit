# loosely inspired by https://github.com/lodimasq/batch-checkpoint-merger/blob/master/batch_checkpoint_merger/main.py#L71

from sdkit.utils import load_tensor_file, log, save_tensor_file


def merge_models(model0_path: str, model1_path: str, ratio: float, out_path: str, use_fp16=True):
    """
    Merges (using weighted sum) and writes to the `out_path`.

    * model0, model1 - the first and second model files to be merged
    * ratio - the ratio of the second model. 1 means only the second model will be used.
              0 means only the first model will be used.
    """

    log.info(f"[cyan]Merge models:[/cyan] Merging {model0_path} and {model1_path}, ratio {ratio}")
    merged = merge_two_models(model0_path, model1_path, ratio, use_fp16)

    log.info(f"[cyan]Merge models:[/cyan] ... saving as {out_path}")

    if out_path.lower().endswith(".safetensors"):
        # elldrethSOg4060Mix_v10.ckpt contains a state_dict key among all the tensors, but safetensors
        # assumes that all entries are tensors, not dicts => remove the key
        if "state_dict" in merged:
            del merged["state_dict"]
        save_tensor_file(merged, out_path)
    else:
        save_tensor_file({"state_dict": merged}, out_path)


# do this pair-wise, to avoid having to load all the models into memory
def merge_two_models(model0, model1, alpha, use_fp16=True):
    """
    Returns a tensor containing the merged model. Uses weighted-sum.

    * model0, model1 - the first and second model files to be merged
    * alpha - a float between [0, 1]. 0 means only model0 will be used, 1 means only model1.

    If model0 is a tensor, then model0 will be over-written with the merged data,
    and the same model0 reference will be returned.
    """

    model0_file = load_tensor_file(model0) if isinstance(model0, str) else model0
    model1_file = load_tensor_file(model1) if isinstance(model1, str) else model1

    model0 = model0_file["state_dict"] if "state_dict" in model0_file else model0_file
    model1 = model1_file["state_dict"] if "state_dict" in model1_file else model1_file

    # common weights
    for key in model0.keys():
        if "model" in key and key in model1:
            model0[key] = (1 - alpha) * model0[key] + alpha * model1[key]

    for key in model1.keys():
        if "model" in key and key not in model0:
            model0[key] = model1[key]

    if use_fp16:
        for key, val in model0.items():
            if "model" in key:
                model0[key] = val.half()

    # unload model1 from memory
    del model1

    return model0
