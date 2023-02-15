import math

import torch
from einops import rearrange
from ldm.util import default
from torch import einsum

from sdkit import Context
from sdkit.utils import log


def send_to_device(context: Context, model):
    """
    Sends the model to the device, based on the VRAM optimizations set in
    `context.vram_optimizations`.

    Please see the documentation for `diffusionkit.types.Context.vram_optimizations`
    for a summary of the logic used for VRAM optimizations
    """
    if len(context.vram_optimizations) == 0 or context.device == "cpu":
        log.info("No VRAM optimizations being applied")
        model.to(context.device)
        model.cond_stage_model.device = context.device
        return

    log.info(f"VRAM Optimizations: {context.vram_optimizations}")

    # based on the approach at https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/lowvram.py
    # the idea is to keep only one module in the GPU at a time, depending on the desired optimization level
    # using torch's `register_forward_pre_hook()` hook

    context.module_in_gpu = None

    def move_to_gpu(module, _):
        """
        This hook ensures that only this module is in the GPU. It moves the
        other module back to the CPU before loading itself to the GPU.
        """
        if module == context.module_in_gpu:
            return

        if context.module_in_gpu is not None:
            context.module_in_gpu.to("cpu")
            log.debug(
                f"moved {getattr(context.module_in_gpu, 'log_name', context.module_in_gpu.__class__.__name__)} to cpu"
            )

        module.to(context.device)
        if module == model.cond_stage_model:
            module.device = context.device
        context.module_in_gpu = module
        log.debug(
            f"moved {getattr(context.module_in_gpu, 'log_name', context.module_in_gpu.__class__.__name__)} to GPU"
        )

    def wrap_fs_fn(fn, model_to_move):
        def wrap(x):
            move_to_gpu(model_to_move, None)
            return fn(x)

        return wrap

    if (
        "KEEP_FS_AND_CS_IN_CPU" in context.vram_optimizations
        or "KEEP_ENTIRE_MODEL_IN_CPU" in context.vram_optimizations
    ):
        # move the FS, CS and the main model to CPU. And send only the overall reference to the correct device
        tmp = model.cond_stage_model, model.first_stage_model, model.model
        model.cond_stage_model, model.first_stage_model, model.model = (None,) * 3
        model.to(context.device)
        model.cond_stage_model, model.first_stage_model, model.model = tmp

        # set forward_pre_hook (a feature of torch NN module) to move each module to the GPU only when required
        model.first_stage_model.log_name = "model.first_stage_model"
        model.first_stage_model.register_forward_pre_hook(move_to_gpu)
        model.first_stage_model.encode = wrap_fs_fn(model.first_stage_model.encode, model.first_stage_model)
        model.first_stage_model.decode = wrap_fs_fn(model.first_stage_model.decode, model.first_stage_model)

        model.cond_stage_model.log_name = "model.cond_stage_model"
        model.cond_stage_model.register_forward_pre_hook(move_to_gpu)
        model.cond_stage_model.forward = wrap_fs_fn(model.cond_stage_model.forward, model.cond_stage_model)

    if (
        "KEEP_ENTIRE_MODEL_IN_CPU" in context.vram_optimizations
    ):  # apply the same approach, but to the individual blocks in model
        d = model.model.diffusion_model

        tmp = d.input_blocks, d.middle_block, d.output_blocks, d.time_embed
        d.input_blocks, d.middle_block, d.output_blocks, d.time_embed = (None,) * 4
        model.model.to(context.device)
        d.input_blocks, d.middle_block, d.output_blocks, d.time_embed = tmp

        d.time_embed.log_name = "model.model.diffusion_model.time_embed"
        d.time_embed.register_forward_pre_hook(move_to_gpu)

        for i, block in enumerate(d.input_blocks):
            block.log_name = f"model.model.diffusion_model.input_blocks[{i}]"
            block.register_forward_pre_hook(move_to_gpu)

        d.middle_block.log_name = "model.model.diffusion_model.middle_block"
        d.middle_block.register_forward_pre_hook(move_to_gpu)

        for i, block in enumerate(d.output_blocks):
            block.log_name = f"model.model.diffusion_model.output_blocks[{i}]"
            block.register_forward_pre_hook(move_to_gpu)
    else:
        model.model.to(context.device)

    if (
        "KEEP_ENTIRE_MODEL_IN_CPU" not in context.vram_optimizations
        and "KEEP_FS_AND_CS_IN_CPU" not in context.vram_optimizations
    ):
        model.to(context.device)
        model.cond_stage_model.device = context.device


def get_context_kv(attention_context):
    return attention_context, attention_context


# modified version of https://github.com/Doggettx/stable-diffusion/blob/main/ldm/modules/attention.py#L170
# faster iterations/sec than the default SD implementation, and consumes far less VRAM
# On a 3060 12 GB (with the sd-v1-4.ckpt model):
# - without this code, the standard SD sampler runs at 4.5 it/sec, and consumes ~6.6 GB of VRAM
# - using this code makes the sampler run at 5.6 to 5.9 it/sec, and consume ~3.6 GB of VRAM on lower-end PCs, and ~4.9 GB on higher-end PCs
def make_attn_forward(context: Context, attn_precision="fp16"):
    app_context = context

    def get_steps(q, k):
        if context.device == "cpu" or "SET_ATTENTION_STEP_TO_2" in context.vram_optimizations:
            return 2
        elif "SET_ATTENTION_STEP_TO_4" in context.vram_optimizations:
            return 4  # use for balanced
        elif "SET_ATTENTION_STEP_TO_6" in context.vram_optimizations:
            return 6
        elif "SET_ATTENTION_STEP_TO_8" in context.vram_optimizations:
            return 8
        elif "SET_ATTENTION_STEP_TO_16" in context.vram_optimizations:
            return 16
        elif "SET_ATTENTION_STEP_TO_24" in context.vram_optimizations:
            return 24  # use for low

        # figure out the available memory
        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats["active_bytes.all.current"]
        mem_reserved = stats["reserved_bytes.all.current"]
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        # figure out the required memory
        gb = 1024**3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier

        steps = 1
        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(
                f"Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). "
                f"Need: {mem_required / 64 / gb:0.1f} GB free, Have:{mem_free_total / gb:0.1f} GB free"
            )

        return steps

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)
        context_k, context_v = get_context_kv(context)
        k_in = self.to_k(context_k)
        v_in = self.to_v(context_v)
        k_in *= self.scale
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        autocast_device = "cpu" if app_context.device == "cpu" else "cuda"  # doesn't accept (or need) 'cuda:N'

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        steps = get_steps(q, k)
        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            if attn_precision == "fp32":
                with torch.autocast(enabled=False, device_type=autocast_device):
                    q, k = q.float(), k.float()
                    s1 = einsum("b i d, b j d -> b i j", q[:, i:end], k)
            else:
                s1 = einsum("b i d, b j d -> b i j", q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum("b i j, b j d -> b i d", s2, v)
            del s2

        del q, k, v

        r2 = rearrange(r1, "(b h) n d -> b n (h d)", h=h)
        del r1

        return self.to_out(r2)

    return forward


def print_model_size_breakdown(model):
    """
    Useful debugging function for analyzing the memory usage of a model
    """

    def mb(n_bytes):
        return int(n_bytes / float(10**6))

    log.info(f"precision: {model.dtype}")

    # model
    size_input, size_middle, size_output = 0, 0, 0
    for key, val in model.model.diffusion_model.state_dict().items():
        s = val.element_size() * val.nelement()
        if "input" in key:
            size_input += s
        elif "middle" in key:
            size_middle += s
        elif "output" in key:
            size_output += s

    log.info(
        f"model.diffusion_model (input, middle, output blocks): {mb(size_input)} Mb, {mb(size_middle)} Mb, {mb(size_output)} Mb"
    )
    log.info(f"model.diffusion_model (total): {mb(size_input + size_middle + size_output)} Mb")

    # modelFS
    sizeFS = 0
    for _, val in model.first_stage_model.state_dict().items():
        sizeFS += val.element_size() * val.nelement()

    log.info(f"model.first_stage_model: {mb(sizeFS)} Mb")

    # modelCS
    sizeCS = 0
    for _, val in model.cond_stage_model.state_dict().items():
        sizeCS += val.element_size() * val.nelement()

    log.info(f"model.cond_stage_model: {mb(sizeCS)} Mb")

    log.info(f"model (TOTAL): {mb(size_input + size_middle + size_output + sizeFS + sizeCS)} Mb")
