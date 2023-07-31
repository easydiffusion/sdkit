import os

import torch
from dataclasses import dataclass
import numpy as np
import traceback

from sdkit.utils import log

"""
Current issues:
1. TRT is working only with fp32

2. TRT goes out of memory when converting larger image ranges. Maybe try converting after switching to "balanced"?

3. set CUDA_MODULE_LOADING=LAZY

4. LoRA
 > No clear approach for DirectML
 > TRT: https://github.com/NVIDIA/TensorRT/blob/release/8.6/demo/Diffusion/utilities.py#L90
 > Currently it takes an entire different ONNX file and transfers their weights. One can modify it to target specifically the KQV part of the network for LORAs instead.

5. TRT performance is pretty restricted to a single image size
"""


def apply_directml_unet(pipeline, onnx_path):
    unet_dml = UnetDirectML(onnx_path)

    pipeline.unet.forward = unet_dml.forward


def apply_tensorrt(pipeline, trt_dir):
    old_unet_forward = pipeline.unet.forward
    old_vae_forward = pipeline.vae.decoder.forward

    try:
        trt = TRTModel(pipeline, trt_dir)

        pipeline.unet.forward = trt.forward_unet
        pipeline.vae.decoder.forward = trt.forward_vae

        setattr(pipeline.unet, "_allocate_trt_buffers", trt.allocate_buffers)
        setattr(pipeline.unet, "_non_trt_forward", old_unet_forward)
        setattr(pipeline.unet, "_trt_forward", trt.forward_unet)
        setattr(pipeline.vae.decoder, "_non_trt_forward", old_vae_forward)
        setattr(pipeline.vae.decoder, "_trt_forward", trt.forward_vae)

        log.info("Using TensorRT accelerated UNet and VAE")
    except:
        traceback.print_exc()
        pipeline.unet.forward = old_unet_forward
        pipeline.vae.decoder.forward = old_vae_forward


class UnetDirectML:
    def __init__(self, onnx_path):
        from diffusers.pipelines.onnx_utils import OnnxRuntimeModel
        import onnxruntime as ort

        # batch_size = 1

        # these are supposed to make things faster, experiment with them
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        # sess_options.add_free_dimension_override_by_name("sample_batch", batch_size * 2)
        # sess_options.add_free_dimension_override_by_name("sample_channels", 4)
        # sess_options.add_free_dimension_override_by_name("sample_height", 64)
        # sess_options.add_free_dimension_override_by_name("sample_width", 64)
        # sess_options.add_free_dimension_override_by_name("timestep_batch", batch_size * 2)
        # sess_options.add_free_dimension_override_by_name("encoder_hidden_states_batch", batch_size * 2)
        # sess_options.add_free_dimension_override_by_name("encoder_hidden_states_sequence", 77)

        import wmi

        w = wmi.WMI()
        device_id = 0
        for i, controller in enumerate(w.Win32_VideoController()):
            device_name = controller.wmi_property("Name").value
            if "AMD" in device_name and "Radeon" in device_name:
                device_id = i
                break

        log.info(f"Using DirectML device_id: {device_id}")
        sess = ort.InferenceSession(
            onnx_path,
            providers=["DmlExecutionProvider"],
            sess_options=sess_options,
            provider_options=[{"device_id": device_id}],
        )

        self.unet_dml = OnnxRuntimeModel(model=sess)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE

        device = sample.device

        timestep_dtype = next(
            (input.type for input in self.unet_dml.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        input = {
            "sample": sample.cpu().numpy(),
            "timestep": np.array([timestep.cpu()], dtype=timestep_dtype),
            "encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
        }

        sample = self.unet_dml(**input)[0]
        sample = torch.from_numpy(sample).to(device)
        return [sample]


class TRTModel:
    ENGINE_TYPES = ("unet", "vae")
    ENGINE_SPANS = [(512, 768), (768, 1024), (1024, 1280)]  # pixels

    def __init__(self, pipeline, trt_dir):
        import tensorrt as trt

        self.base_dir = trt_dir

        self.pipeline = pipeline
        self.old_forward = {
            "unet": pipeline.unet.forward,
            "vae": pipeline.vae.decoder.forward,
        }

        self.engine_paths = {}
        self.engines = {}
        self.tensors = {engine_type: {} for engine_type in self.ENGINE_TYPES}

        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(None, "")

        for engine_type in self.ENGINE_TYPES:
            engine_paths = self.engine_paths[engine_type] = {}
            for min, max in self.ENGINE_SPANS:
                engine_paths[(min, max)] = os.path.join(trt_dir, engine_type, f"{min}_{max}.trt")

        print(self.engine_paths)

        for engine_type in self.ENGINE_TYPES:
            for engine_span in self.ENGINE_SPANS:
                self.load_engine(engine_type, engine_span)

        print(self.engines)

    def load_engine(self, engine_type, engine_span: tuple):
        import tensorrt as trt

        try:
            engine_path = self.engine_paths[engine_type][engine_span]
            log.info(f"Loading {engine_type} TensorRT engine from {engine_path}")

            if not os.path.exists(engine_path) or os.stat(engine_path).st_size == 0:
                return

            with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
                trt_context = engine.create_execution_context()
                if engine_type not in self.engines:
                    self.engines[engine_type] = {}
                self.engines[engine_type][engine_span] = engine, trt_context
        except:
            traceback.print_exc()

    def allocate_buffers(self, pipeline, device, dtype, width, height):
        "Call this once before an image is generated, not per sample"

        for engine_type in self.ENGINE_TYPES:
            size = max(width, height) // 256
            engine_span = (256 * size, 256 * (size + 1))
            if engine_span not in self.engines[engine_type]:
                continue

            self._allocate_buffers(engine_type, pipeline, device, dtype, width, height, engine_span)

    def _allocate_buffers(self, engine_type, pipeline, device, dtype, width, height, engine_span):
        tensors = self.tensors[engine_type]
        tensors.clear()

        dtype = torch.float32  # HACK: but TRT generates black images otherwise

        if engine_type == "unet":
            unet_in_channels = pipeline.unet.config.in_channels
            num_tokens = pipeline.text_encoder.config.max_position_embeddings
            text_hidden_size = pipeline.text_encoder.config.hidden_size

            shape_dict = {
                "sample": (2, unet_in_channels, width // 8, height // 8),
                "encoder_hidden_states": (2, num_tokens, text_hidden_size),
                "timestep": (2,),
            }
        elif engine_type == "vae":
            num_channels_latents = pipeline.unet.config.in_channels
            vae_scale_factor = pipeline.vae_scale_factor

            shape_dict = {
                "sample": (2, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor),
            }

        engine, trt_context = self.engines[engine_type][engine_span]

        for i, binding in enumerate(engine):
            if binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = engine.get_binding_shape(binding)

            if binding == "out_sample":
                shape = shape_dict["sample"]

            if engine.binding_is_input(binding):
                trt_context.set_binding_shape(i, shape)

            tensors[binding] = torch.empty(tuple(shape), dtype=dtype, device=device)

    def forward_unet(self, sample, timestep, encoder_hidden_states, **kwargs):
        feed_dict = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        return self._forward("unet", feed_dict)

    def forward_vae(self, sample, **kwargs):
        feed_dict = {
            "sample": sample,
        }
        return self._forward("vae", feed_dict)

    def _forward(self, engine_type, feed_dict):
        from polygraphy import cuda

        # check if we have an engine for this sample, else use the non-trt forward
        factor = 8 if engine_type == "unet" else self.pipeline.vae_scale_factor
        sample = feed_dict["sample"]
        size = max(sample.shape[2], sample.shape[3]) * factor // 256
        size = (256 * size, 256 * (size + 1))
        if size not in self.engines[engine_type]:
            log.warn(
                f"Did not find a {engine_type} TensorRT engine for {size} {sample.shape}. Using non-TRT rendering.."
            )
            if engine_type == "unet":
                return [
                    self.old_forward["unet"](
                        feed_dict["sample"], feed_dict["timestep"], feed_dict["encoder_hidden_states"]
                    )
                ]
            elif engine_type == "vae":
                return self.old_forward["vae"](feed_dict["sample"])

        orig_dtype = sample.dtype
        target_dtype = torch.float32

        tensors = self.tensors[engine_type]
        trt_context = self.engines[engine_type][size][1]

        stream = cuda.Stream()

        for name, tensor in feed_dict.items():
            tensors[name].copy_(tensor.to(target_dtype))

        for name, tensor in tensors.items():
            trt_context.set_tensor_address(name, tensor.data_ptr())

        if not trt_context.execute_async_v3(stream_handle=stream.ptr):
            log.warn(f"Error processing the {engine_type} TensorRT engine for {size}. Using non-TRT rendering..")
            if engine_type == "unet":
                sample = self.old_forward["unet"](
                    feed_dict["sample"], feed_dict["timestep"], feed_dict["encoder_hidden_states"]
                )
            elif engine_type == "vae":
                sample = self.old_forward["vae"](feed_dict["sample"])

            sample = sample.to(orig_dtype)
            return [sample] if engine_type == "unet" else sample

        sample = tensors["out_sample"]
        sample = sample.to(orig_dtype)
        return [sample] if engine_type == "unet" else sample
