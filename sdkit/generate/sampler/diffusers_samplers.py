from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    DEISMultistepScheduler,
    DDPMScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
)


def _make_plms(ddim_scheduler):
    config = dict(ddim_scheduler.config)
    config["skip_prk_steps"] = True
    return PNDMScheduler.from_config(ddim_scheduler.config)


def _make_unipc(solver_type: str, solver_order: int, time_skip: str):
    def callback(ddim_scheduler):
        # logSNR and time_quadratic timeskips are not supported in diffusers
        if time_skip != "time_uniform":
            return None

        scheduler = UniPCMultistepScheduler.from_config(ddim_scheduler.config)
        scheduler.config.solver_type = solver_type
        scheduler.config.solver_order = solver_order
        scheduler.config.lower_order_final = True
        scheduler.config.thresholding = False
        return scheduler

    return callback


_samplers_init = {
    "plms": _make_plms,
    "ddim": lambda ddim_scheduler: ddim_scheduler,
    "dpm_solver_stability": lambda ddim_scheduler: DPMSolverMultistepScheduler.from_config(
        ddim_scheduler.config, algorithm_type="dpmsolver"
    ),
    "euler_a": lambda ddim_scheduler: EulerAncestralDiscreteScheduler.from_config(ddim_scheduler.config),
    "euler": lambda ddim_scheduler: EulerDiscreteScheduler.from_config(ddim_scheduler.config),
    "lms": lambda ddim_scheduler: LMSDiscreteScheduler.from_config(ddim_scheduler.config),
    "heun": lambda ddim_scheduler: HeunDiscreteScheduler.from_config(ddim_scheduler.config),
    "dpm2": lambda ddim_scheduler: KDPM2DiscreteScheduler.from_config(ddim_scheduler.config),
    "dpm2_a": lambda ddim_scheduler: KDPM2AncestralDiscreteScheduler.from_config(ddim_scheduler.config),
    "dpmpp_2s_a": lambda ddim_scheduler: DPMSolverSinglestepScheduler.from_config(
        ddim_scheduler.config, use_karras_sigmas=True
    ),
    "dpmpp_2m": lambda ddim_scheduler: DPMSolverMultistepScheduler.from_config(
        ddim_scheduler.config, use_karras_sigmas=True
    ),
    "dpmpp_2m_sde": lambda ddim_scheduler: DPMSolverMultistepScheduler.from_config(
        ddim_scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
    ),
    "dpmpp_sde": lambda ddim_scheduler: DPMSolverSDEScheduler.from_config(
        ddim_scheduler.config, use_karras_sigmas=True
    ),
    "dpm_fast": None,  # Not implemented in diffusers yet
    "dpm_adaptive": None,  # Not implemented in diffusers yet
    "ddpm": lambda ddim_scheduler: DDPMScheduler.from_config(ddim_scheduler.config),
    "deis": lambda ddim_scheduler: DEISMultistepScheduler.from_config(ddim_scheduler.config),
    "unipc_snr": _make_unipc("bh1", 3, "logSNR"),  # logSNR is not supported in diffusers yet
    "unipc_tu": _make_unipc("bh2", 2, "time_uniform"),
    "unipc_tq": _make_unipc("bh1", 3, "time_quadratic"),  # time_quadratic is not supported in diffusers yet
    "unipc_snr_2": _make_unipc("vary_coeff", 1, "logSNR"),  # logSNR is not supported in diffusers yet
    "unipc_tu_2": _make_unipc("bh1", 3, "time_uniform"),
}

# plms alias
_samplers_init["pndm"] = _samplers_init["plms"]
# dpm_solver_stability alias
_samplers_init["dpm"] = _samplers_init["dpm_solver_stability"]
# euler_a alias
_samplers_init["euler-ancestral"] = _samplers_init["euler_a"]


def make_samplers(context, ddim_scheduler):
    context.samplers = {}

    for sampler_name, sampler_factory in _samplers_init.items():
        if sampler_factory is not None:
            context.samplers[sampler_name] = sampler_factory(ddim_scheduler)
        else:
            context.samplers[sampler_name] = None
