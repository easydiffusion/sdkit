from sdkit.train import merge_models

merge_models(
    model0_path="D:\\path\\to\\model_a.ckpt",
    model1_path="D:\\path\\to\\model_b.ckpt",
    ratio=0.3,
    out_path="D:\\path\\to\\merged_model.safetensors",
    use_fp16=True,
)
