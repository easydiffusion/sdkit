import platform
import subprocess
from typing import Union, Tuple, Dict


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


def _has_directml_platform():
    import torch

    try:
        import torch_directml

        torch.directml = torch_directml

        return True
    except ImportError:
        pass

    return False


def get_torch_platform():
    import torch
    from platform import system as os_name

    if _has_directml_platform():
        return "directml", torch.directml

    if torch.cuda.is_available():
        return "cuda", torch.cuda
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", torch.xpu
    if hasattr(torch, "mps") and os_name() == "Darwin":
        return "mps", torch.mps
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        return "mtia", torch.mtia

    return "cpu", torch.cpu


def get_device_count() -> int:
    torch_platform_name, torch_platform = get_torch_platform()

    return torch_platform.device_count()


def get_device_name(device) -> str:
    "Expects a torch.device as the argument"

    torch_platform_name, torch_platform = get_torch_platform()
    if torch_platform_name not in ("xpu", "cuda", "directml"):
        return f"{torch_platform_name}:{device.index}"

    if torch_platform_name == "directml":
        return torch_platform.device_name(device.index)

    return torch_platform.get_device_name(device.index)


def get_device(device: Union[int, str]):
    import torch

    if isinstance(device, str):
        if ":" in device:
            torch_platform_name, device_index = device.split(":")
            device_index = int(device_index)
        else:
            torch_platform_name, device_index = device, 0
    else:
        torch_platform_name, _ = get_torch_platform()
        device_index = device

    if torch_platform_name == "directml" and _has_directml_platform():
        return torch.directml.device(device_index)

    return torch.device(torch_platform_name, device_index)


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
    torch_platform_name, torch_platform = get_torch_platform()
    if torch_platform_name not in ("cuda", "xpu"):
        return

    torch_platform.empty_cache()


def ipc_collect():
    torch_platform_name, torch_platform = get_torch_platform()
    if torch_platform_name != "cuda":
        return

    torch_platform.ipc_collect()


def is_cpu_device(device) -> bool:  # used for cpu offloading etc
    "Expects a torch.device as the argument"

    return device.type in ("cpu", "mps")
