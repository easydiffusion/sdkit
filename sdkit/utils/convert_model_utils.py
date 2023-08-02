import os
import shutil
from packaging import version
import warnings

from sdkit.utils import log


def convert_pipeline_unet_to_onnx(pipeline, save_path, opset=17, device=None, fp16: bool = False):
    if os.path.exists(save_path) and os.stat(save_path).st_size > 0:
        return

    import torch

    log.info("Making intermediate Unet ONNX..")

    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size

    model_args = (
        torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size),
        torch.randn(2),
        torch.randn(2, num_tokens, text_hidden_size),
        False,
    )
    input_names = ["sample", "timestep", "encoder_hidden_states", "return_dict"]
    dynamic_axes = {
        "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        "timestep": {0: "batch"},
        "encoder_hidden_states": {0: "batch", 1: "sequence"},
    }
    _convert_pipeline_model_to_onnx(
        pipeline,
        pipeline.unet,
        model_args,
        input_names,
        dynamic_axes,
        use_external_data_format=True,
        save_path=save_path,
        opset=opset,
        device=device,
        fp16=fp16,
    )


def convert_pipeline_vae_to_onnx(pipeline, save_path, opset=17, device=None, fp16: bool = False):
    if os.path.exists(save_path) and os.stat(save_path).st_size > 0:
        return

    import torch

    log.info("Making intermediate VAE ONNX..")

    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size

    model_args = (
        torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size),
        False,
    )
    input_names = ["sample"]
    dynamic_axes = {
        "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
    }
    _convert_pipeline_model_to_onnx(
        pipeline,
        pipeline.vae.decoder,
        model_args,
        input_names,
        dynamic_axes,
        use_external_data_format=False,
        save_path=save_path,
        opset=opset,
        device=device,
        fp16=fp16,
    )


def _convert_pipeline_model_to_onnx(
    pipeline,
    model,
    model_args,
    input_names,
    dynamic_axes,
    use_external_data_format,
    save_path,
    opset=17,
    device=None,
    fp16: bool = False,
):
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

    orig_device = pipeline.device
    orig_dtype = pipeline.vae.dtype

    _dtype = torch.float16 if fp16 else torch.float32
    _device = device if device else pipeline.device
    pipeline = pipeline.to(_device, torch_dtype=_dtype)

    if use_external_data_format:
        tmp_dir = save_path + "_"  # collect the individual weights here
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
    model_path = os.path.join(tmp_dir, "model.onnx") if use_external_data_format else save_path

    model_name, _ = os.path.splitext(save_path)
    model_name = os.path.basename(model_name)

    model_args = tuple(m.to(device=_device, dtype=_dtype) for m in model_args if isinstance(m, torch.Tensor))

    onnx_export(
        model,
        model_args=model_args,
        output_path=model_path,
        ordered_input_names=input_names,
        output_names=["out_sample"],  # has to be different from "sample" for correct tracing
        dynamic_axes=dynamic_axes,
        opset=opset,
        use_external_data_format=use_external_data_format,
    )

    if use_external_data_format:
        model = onnx.load(model_path)
        shutil.rmtree(tmp_dir)
        # collate external tensor files into one
        onnx.save_model(
            model,
            save_path,
            save_as_external_data=use_external_data_format,
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


def convert_onnx_unet_to_tensorrt(pipeline, onnx_path, trt_out_dir, batch_size_range, dimensions_range):
    batch_size_min, batch_size_max = batch_size_range

    unet_in_channels = pipeline.unet.config.in_channels
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size

    def get_shapes(min_size, max_size):
        opt_size = min_size + int((max_size - min_size) * 0.25)
        opt_batch_size = batch_size_min + int((batch_size_min - batch_size_max) * 0.25)
        min_shape = {
            "sample": (batch_size_min * 2, unet_in_channels, min_size // 8, min_size // 8),
            "encoder_hidden_states": (batch_size_min * 2, num_tokens, text_hidden_size),
            "timestep": (batch_size_min * 2,),
        }
        opt_shape = {
            "sample": (opt_batch_size * 2, unet_in_channels, opt_size // 8, opt_size // 8),
            "encoder_hidden_states": (opt_batch_size * 2, num_tokens, text_hidden_size),
            "timestep": (opt_batch_size * 2,),
        }
        max_shape = {
            "sample": (batch_size_max * 2, unet_in_channels, max_size // 8, max_size // 8),
            "encoder_hidden_states": (batch_size_max * 2, num_tokens, text_hidden_size),
            "timestep": (batch_size_max * 2,),
        }
        return min_shape, opt_shape, max_shape

    _convert_onnx_to_tensorrt(onnx_path, trt_out_dir, get_shapes, "unet", batch_size_range, dimensions_range)


def convert_onnx_vae_to_tensorrt(pipeline, onnx_path, trt_out_dir, batch_size_range, dimensions_range):
    batch_size_min, batch_size_max = batch_size_range

    unet_in_channels = pipeline.unet.config.in_channels

    def get_shapes(min_size, max_size):
        opt_size = min_size + int((max_size - min_size) * 0.25)
        opt_batch_size = batch_size_min + int((batch_size_min - batch_size_max) * 0.25)
        min_shape = {
            "sample": (batch_size_min * 2, unet_in_channels, min_size // 8, min_size // 8),
        }
        opt_shape = {
            "sample": (opt_batch_size * 2, unet_in_channels, opt_size // 8, opt_size // 8),
        }
        max_shape = {
            "sample": (batch_size_max * 2, unet_in_channels, max_size // 8, max_size // 8),
        }
        return min_shape, opt_shape, max_shape

    _convert_onnx_to_tensorrt(onnx_path, trt_out_dir, get_shapes, "vae", batch_size_range, dimensions_range)


def _convert_onnx_to_tensorrt(onnx_path, trt_out_dir, shape_fn, name, batch_size_range, dimensions_range):
    batch_size_min, batch_size_max = batch_size_range
    convert = False
    for min_size, max_size in dimensions_range:
        save_path = os.path.join(trt_out_dir, f"{batch_size_min}_{batch_size_max},{min_size}_{max_size}.trt")
        if not os.path.exists(save_path) or os.stat(save_path).st_size == 0:
            convert = True
            break

    if not convert:
        return

    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    TIMING_CACHE = "trt_timing.cache"

    TRT_BUILDER = trt.Builder(TRT_LOGGER)
    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    parse_success = onnx_parser.parse_from_file(onnx_path)

    for idx in range(onnx_parser.num_errors):
        log.error(onnx_parser.get_error(idx))
    if not parse_success:
        raise RuntimeError("ONNX model parsing failed")

    for min_size, max_size in dimensions_range:
        save_path = os.path.join(trt_out_dir, f"{batch_size_min}_{batch_size_max},{min_size}_{max_size}.trt")
        if os.path.exists(save_path) and os.stat(save_path).st_size > 0:
            continue

        log.info(
            f"Making TRT engine for {name}, size range from {min_size}x{min_size} to {max_size}x{max_size}, batch size range: {batch_size_range}.."
        )
        config = TRT_BUILDER.create_builder_config()
        profile = TRT_BUILDER.create_optimization_profile()

        if os.path.exists(TIMING_CACHE):
            with open(TIMING_CACHE, "rb") as f:
                timing_cache = config.create_timing_cache(f.read())
        else:
            timing_cache = config.create_timing_cache(b"")
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

        min_shape, opt_shape, max_shape = shape_fn(min_size, max_size)

        for name in min_shape.keys():
            profile.set_shape(name, min_shape[name], opt_shape[name], max_shape[name])

        config.add_optimization_profile(profile)

        # config.max_workspace_size = 4096 * (1 << 20)
        config.set_flag(trt.BuilderFlag.FP16)
        serialized_engine = TRT_BUILDER.build_serialized_network(network, config)

        ## save TRT engine
        if not os.path.exists(trt_out_dir):
            os.makedirs(trt_out_dir, exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(serialized_engine)

        # save the timing cache
        timing_cache = config.get_timing_cache()
        with timing_cache.serialize() as buffer:
            with open(TIMING_CACHE, "wb") as f:
                f.write(buffer)
                f.flush()
                os.fsync(f)
                log.info(f"Wrote TRT timing cache to {TIMING_CACHE}")

        log.info(f"TRT Engine saved to {save_path}")


def convert_pipeline_to_onnx(pipeline, save_path, opset=17, device=None, fp16: bool = False):
    unet_onnx = os.path.join(save_path, "unet", "model.onnx")
    # vae_onnx = os.path.join(save_path, "vae", "model.onnx")

    convert_pipeline_unet_to_onnx(pipeline, unet_onnx, opset, device=device, fp16=fp16)
    # convert_pipeline_vae_to_onnx(pipeline, vae_onnx, opset, device=device, fp16=fp16)


def convert_pipeline_to_tensorrt(
    pipeline, trt_dir_path, batch_size_range, dimensions_range, opset=17, fp16: bool = False
):
    unet_path = os.path.join(trt_dir_path, "unet")
    # vae_path = os.path.join(trt_dir_path, "vae")

    os.makedirs(unet_path, exist_ok=True)
    # os.makedirs(vae_path, exist_ok=True)

    convert_pipeline_to_onnx(pipeline, trt_dir_path, opset, device="cpu", fp16=False)

    unet_onnx = os.path.join(trt_dir_path, "unet", "model.onnx")
    # vae_onnx = os.path.join(trt_dir_path, "vae", "model.onnx")

    convert_onnx_unet_to_tensorrt(pipeline, unet_onnx, unet_path, batch_size_range, dimensions_range)
    # convert_onnx_vae_to_tensorrt(pipeline, vae_onnx, vae_path, batch_size_range, dimensions_range)
