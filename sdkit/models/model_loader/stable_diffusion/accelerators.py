import torch
from dataclasses import dataclass
import numpy as np

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


def apply_tensorrt_unet(pipeline, trt_path):
    unet_trt = UnetTRT(trt_path)

    pipeline.unet.forward = unet_trt.forward

    setattr(pipeline.unet, "_allocate_trt_buffers", unet_trt.allocate_buffers)


class UnetDirectML:
    def __init__(self, onnx_path):
        from diffusers.pipelines.onnx_utils import OnnxRuntimeModel
        import onnxruntime as ort

        # batch_size = 1

        # these are supposed to make things faster, but don't seem to make a difference for me
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


class UnetTRT:
    def __init__(self, engine_path):
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.trt_context = self.engine.create_execution_context()
        self.tensors = {}

    def allocate_buffers(self, pipeline, device, dtype, width=512, height=512):
        "Call this once before an image is generated, not per sample"

        unet_in_channels = pipeline.unet.config.in_channels
        num_tokens = pipeline.text_encoder.config.max_position_embeddings
        text_hidden_size = pipeline.text_encoder.config.hidden_size

        self.tensors.clear()

        shape_dict = {
            "sample": (2, unet_in_channels, width // 8, height // 8),
            "encoder_hidden_states": (2, num_tokens, text_hidden_size),
            "timestep": (2,),
        }
        for i, binding in enumerate(self.engine):
            if binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)

            if binding == "out_sample":
                shape = (2, 4, width // 8, height // 8)

            if self.engine.binding_is_input(binding):
                self.trt_context.set_binding_shape(i, shape)

            self.tensors[binding] = torch.empty(tuple(shape), dtype=dtype, device=device)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        from polygraphy import cuda

        feed_dict = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        stream = cuda.Stream()

        for name, tensor in feed_dict.items():
            self.tensors[name].copy_(tensor)

        for name, tensor in self.tensors.items():
            self.trt_context.set_tensor_address(name, tensor.data_ptr())

        if not self.trt_context.execute_async_v3(stream_handle=stream.ptr):
            raise RuntimeError("Inference failed!")

        sample = self.tensors["out_sample"]
        return [sample]
