import re
import platform
import subprocess
from typing import Tuple, Dict


NVIDIA_RE = re.compile(r"\b(?:nvidia|geforce|quadro|tesla)\b", re.IGNORECASE)
NVIDIA_HALF_PRECISION_BUG_RE = re.compile(r"\b(?:tesla k40m|16\d\d|t\d{2,}|mx450)\b", re.IGNORECASE)
AMD_HALF_PRECISION_BUG_RE = re.compile(r"\b(?:navi 1\d)\b", re.IGNORECASE)  # https://github.com/ROCm/ROCm/issues/2527
# FORCE_FULL_PRECISION won't be necessary for AMD once this is fixed (and torch2 wheels are released for ROCm 6.2): https://github.com/pytorch/pytorch/issues/132570#issuecomment-2313071756


def has_amd_gpu():
    os_name = platform.system()
    try:
        if os_name == "Windows":
            res = subprocess.run("wmic path win32_VideoController get name".split(" "), stdout=subprocess.PIPE)
            res = res.stdout.decode("utf-8")
            return "AMD" in res and "Radeon" in res
        elif os_name == "Linux":
            with open("/proc/bus/pci/devices", "r") as f:
                device_info = f.read()

            return "amdgpu" in device_info and "nvidia" not in device_info
    except:
        return False

    return False


def mem_get_info(device) -> Tuple[int, int]:
    "Expects a torch.device as the argument"

    if device.type == "cuda":
        import torch

        return torch.cuda.mem_get_info(device)

    return (0, 0)  # none of the other platforms have working implementations of mem_get_info


def memory_allocated(device) -> int:
    "Expects a torch.device as the argument"

    if device.type == "cuda":
        import torch

        return torch.cuda.memory_allocated(device)

    return 0  # none of the other platforms have working implementations of memory_allocated


def memory_stats(device) -> Dict:
    "Expects a torch.device as the argument"

    if device.type == "cuda":
        import torch

        return torch.cuda.memory_stats(device)

    return {}  # none of the other platforms have working implementations of memory_stats


def empty_cache():
    import torch

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def ipc_collect():
    import torch

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.ipc_collect()


def is_cpu_device(device) -> bool:  # used for cpu offloading etc
    "Expects a torch.device or string as the argument"

    device_type = device.split(":")[0] if isinstance(device, str) else device.type
    return device_type in ("cpu", "mps")


def has_half_precision_bug(device_name) -> bool:
    "Check whether the given device requires full precision for generating images due to a firmware bug"
    if NVIDIA_RE.search(device_name):
        return NVIDIA_HALF_PRECISION_BUG_RE.search(device_name) is not None
    return AMD_HALF_PRECISION_BUG_RE.search(device_name) is not None
