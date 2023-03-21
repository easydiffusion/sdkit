from sdkit import Context

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)

samplers = {
    "plms": None,
    "ddim": None,
    "dpm_solver_stability": None,
    "euler_a": None,
    "euler": None,
    "lms": None,
    "heun": None,
    "dpm2": None,
    "dpm2_a": None,
    "dpmpp_2s_a": None,
    "dpmpp_2m": None,
    "dpmpp_sde": None,
    "dpm_fast": None,
    "dpm_adaptive": None,
    "unipc_snr": None,
    "unipc_tu": None,
    "unipc_tq": None,
    "unipc_snr_2": None,
    "unipc_tu_2": None,
}


def make_samplers(ddim_scheduler):
    def make(sampler_name):
        if sampler_name in ("pndm", "plms"):
            config = dict(ddim_scheduler.config)
            config["skip_prk_steps"] = True
            scheduler = PNDMScheduler.from_config(ddim_scheduler.config)
        elif sampler_name == "lms":
            scheduler = LMSDiscreteScheduler.from_config(ddim_scheduler.config)
        elif sampler_name == "heun":
            scheduler = HeunDiscreteScheduler.from_config(ddim_scheduler.config)
        elif sampler_name == "euler":
            scheduler = EulerDiscreteScheduler.from_config(ddim_scheduler.config)
        elif sampler_name in ("euler-ancestral", "euler_a"):
            scheduler = EulerAncestralDiscreteScheduler.from_config(ddim_scheduler.config)
        elif sampler_name in ("dpm", "dpm_solver_stability"):
            scheduler = DPMSolverMultistepScheduler.from_config(ddim_scheduler.config)
        elif sampler_name == "dpm2":
            scheduler = KDPM2DiscreteScheduler.from_config(ddim_scheduler.config)
        elif sampler_name == "dpm2_a":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(ddim_scheduler.config)
        elif sampler_name.startswith("unipc_"):
            scheduler = UniPCMultistepScheduler.from_config(ddim_scheduler.config)

            if sampler_name == "unipc_snr":
                scheduler.config.solver_type = "bh1"
                scheduler.config.solver_order = 3
            elif sampler_name == "unipc_tu":
                scheduler.config.solver_type = "bh2"
                scheduler.config.solver_order = 2
            elif sampler_name == "unipc_tq":
                scheduler.config.solver_type = "bh1"
                scheduler.config.solver_order = 3
            elif sampler_name == "unipc_snr_2":
                scheduler.config.solver_type = "vary_coeff"
                scheduler.config.solver_order = 1
            elif sampler_name == "unipc_tu_2":
                scheduler.config.solver_type = "bh1"
                scheduler.config.solver_order = 3

            scheduler.config.lower_order_final = True
            scheduler.config.thresholding = False

            if sampler_name in ("unipc_snr", "unipc_tq", "unipc_snr_2"):
                scheduler = None  # logSNR and time_quadratic timeskips are not supported in diffusers
        elif sampler_name == "ddim":
            scheduler = ddim_scheduler
        else:
            scheduler = None

        samplers[sampler_name] = scheduler

    for sampler_name in samplers.keys():
        make(sampler_name)
