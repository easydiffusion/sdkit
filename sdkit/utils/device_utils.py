import platform
import subprocess


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
