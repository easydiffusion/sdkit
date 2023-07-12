import platform
import subprocess
import wmi


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

def get_directml_device_id():
    os_name = platform.system()

    if os_name != "Windows":
        return None
    
    w = wmi.WMI()
    device_id = None
    for i, controller in enumerate(w.Win32_VideoController()):
        device_name = controller.wmi_property("Name").value
        if ("AMD" in device_name and "Radeon" in device_name) or ("Intel" in device_name and "Arc" in device_name):
            device_id = i
            break

    return device_id

