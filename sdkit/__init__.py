from threading import local

class Context(local):
    models: dict = {}
    model_paths: dict = {}
    model_configs: dict = {}

    device: str = 'cuda'
    device_name: str = None
    half_precision: bool = True
    vram_optimizations: set = {'KEEP_FS_AND_CS_IN_CPU', 'SET_ATTENTION_STEP_TO_4'}
    '''
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
    '''
