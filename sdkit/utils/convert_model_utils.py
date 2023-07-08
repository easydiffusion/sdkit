import os
import shutil
from packaging import version
import warnings


def convert_pipeline_unet_to_onnx(pipeline, save_path, opset=17, fp16: bool = False):
    import torch
    import onnx
    from torch.jit import TracerWarning

    warnings.filterwarnings(
        "ignore",
        category=TracerWarning,
        message="Converting a tensor to a Python boolean might cause the trace to be incorrect",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="The shape inference of prim::Constant type is missing",
    )

    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size

    orig_device = pipeline.device
    orig_dtype = pipeline.unet.dtype

    _dtype = torch.float16 if fp16 else torch.float32
    _device = pipeline.device if fp16 else "cpu"
    pipeline = pipeline.to(_device, torch_dtype=_dtype)

    tmp_dir = save_path + "_"  # collect the individual weights here
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    tmp_model_path = os.path.join(tmp_dir, "model.onnx")

    model_name, _ = os.path.splitext(save_path)
    model_name = os.path.basename(model_name)

    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=_device, dtype=_dtype),
            torch.randn(2).to(device=_device, dtype=_dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=_device, dtype=_dtype),
            False,
        ),
        output_path=tmp_model_path,
        ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
        },
        opset=opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )

    unet = onnx.load(tmp_model_path)
    shutil.rmtree(tmp_dir)
    # collate external tensor files into one
    onnx.save_model(
        unet,
        save_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=model_name + ".onnx_weights.pb",
        convert_attribute=False,
    )

    pipeline = pipeline.to(orig_device, torch_dtype=orig_dtype)


def onnx_export(
    model,
    model_args: tuple,
    output_path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    import torch

    kwargs = {
        "input_names": ordered_input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "do_constant_folding": True,
        "use_external_data_format": use_external_data_format,
        "enable_onnx_checker": True,
        "opset_version": opset,
    }

    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    is_torch_higher_than_1_11 = version.parse(version.parse(torch.__version__).base_version) > version.parse("1.11")
    if is_torch_higher_than_1_11:
        del kwargs["use_external_data_format"]
        del kwargs["enable_onnx_checker"]

    torch.onnx.export(model, model_args, output_path, **kwargs)


def convert_onnx_unet_to_tensorrt(pipeline, onnx_path, save_path):
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    batch_size = 1
    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size

    TRT_BUILDER = trt.Builder(TRT_LOGGER)
    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    parse_success = onnx_parser.parse_from_file(onnx_path)

    for idx in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(idx))
    if not parse_success:
        raise RuntimeError("ONNX model parsing failed")

    config = TRT_BUILDER.create_builder_config()
    profile = TRT_BUILDER.create_optimization_profile()

    min_shape = {
        "sample": (batch_size, unet_in_channels, unet_sample_size, unet_sample_size),
        "encoder_hidden_states": (batch_size, num_tokens, text_hidden_size),
        "timestep": (batch_size,),
    }
    max_shape = {
        "sample": (batch_size * 2, unet_in_channels, unet_sample_size, unet_sample_size),
        "encoder_hidden_states": (batch_size * 2, num_tokens, text_hidden_size),
        "timestep": (batch_size * 2,),
    }

    for name in min_shape.keys():
        profile.set_shape(name, min_shape[name], min_shape[name], max_shape[name])

    config.add_optimization_profile(profile)

    # config.max_workspace_size = 4096 * (1 << 20)
    config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = TRT_BUILDER.build_serialized_network(network, config)

    ## save TRT engine
    with open(save_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TRT Engine saved to {save_path}")


def convert_pipeline_unet_to_tensorrt(pipeline, save_path, opset=17, fp16: bool = False):
    onnx_path = save_path + ".onnx"

    if not os.path.exists(onnx_path) or os.stat(onnx_path).st_size == 0:
        print("Making intermediate ONNX..")
        convert_pipeline_unet_to_onnx(pipeline, onnx_path, opset, fp16=False)

    if not os.path.exists(save_path) or os.stat(save_path).st_size == 0:
        print("Converting intermediate ONNX to TensorRT..")
        convert_onnx_unet_to_tensorrt(pipeline, onnx_path, save_path)
        # os.remove(onnx_path)
