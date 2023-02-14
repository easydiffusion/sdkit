from threading import local


class Context(local):
    def __init__(self) -> None:
        self._device: str = "cuda:0"
        self._half_precision: bool = True
        self._vram_usage_level = None

        self.models: dict = {}
        self.model_paths: dict = {}
        self.model_configs: dict = {}

        self.device_name: str = None
        self.vram_optimizations: set = set()
        """
        **Do not change this unless you know what you're doing!** Instead set `context.vram_usage_level` to `'low'`, `'balanced'` or `'high'`.

        Possible values:
        * Empty set: Fastest, and consumes the maximum amount of VRAM.

        * `'KEEP_FS_AND_CS_IN_CPU'`: Honestly, not very useful. Slightly slower than `None`, but consumes slightly less VRAM than `None`.
        For the sd-v1-4 model, it consumes atleast 8% less VRAM than `None`. It moves the first_stage and cond_stage
        to the CPU, and keeps the rest of the model in GPU (input, middle and output blocks). The first_stage
        and cond_stage will be moved to the GPU only when they are needed, and moved back to the CPU after that.

        * `'KEEP_ENTIRE_MODEL_IN_CPU'`: Very useful! Slowest option, consumes the least amount of VRAM. For the sd-v1-4 model,
        it consumes atleast 52% less VRAM than using no optimizations. Along with the first_stage and cond_stage, it also moves the input,
        middle and output blocks to the CPU (effectively, the entire model is kept in CPU). Each block will be moved
        to the GPU only when they are needed, and moved back to the CPU after that.

        * `'SET_ATTENTION_STEP_TO_4'`: Pretty useful! Fairly fast performance, consumes a medium amount of VRAM. For the sd-v1-4 model,
        it consumes about 1 GB more than `'KEEP_ENTIRE_MODEL_IN_CPU'`, for a much faster rendering performance.

        * `'SET_ATTENTION_STEP_TO_16'`: Very useful! Lowest GPU memory utilization, but slowest performance.
        """
        self.vram_usage_level = "balanced"

    # hacky approach, but we need to enforce full precision for some devices
    # we also need to force full precision for these devices (haven't implemented this yet):
    # (('nvidia' in device_name or 'geforce' in device_name) and (' 1660' in device_name or ' 1650' in device_name)) or ('Quadro T2000' in device_name)
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, d):
        self._device = d
        if d == "cpu":
            from sdkit.utils import log

            log.info("forcing full precision for device: cpu")
            self._half_precision = False

    @property
    def half_precision(self):
        return self._half_precision

    @half_precision.setter
    def half_precision(self, h):
        self._half_precision = h if self._device != "cpu" else False

    @property
    def vram_usage_level(self):
        return self._vram_usage_level

    @vram_usage_level.setter
    def vram_usage_level(self, level):
        self._vram_usage_level = level

        if level == "low":
            self.vram_optimizations = {"KEEP_ENTIRE_MODEL_IN_CPU", "SET_ATTENTION_STEP_TO_16"}
        elif level == "balanced":
            self.vram_optimizations = {"KEEP_FS_AND_CS_IN_CPU", "SET_ATTENTION_STEP_TO_16"}
        elif level == "high":
            self.vram_optimizations = {"SET_ATTENTION_STEP_TO_2"}
