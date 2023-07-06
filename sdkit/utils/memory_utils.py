import base64
from functools import reduce
from gc import collect, get_objects, get_referrers

import psutil

from sdkit import Context

tensor_ids_snapshot = None
recorded_tensor_names = {}


def gc(context: Context):
    import torch

    collect()
    if "cuda" in context.device:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_device_usage(device, log_info=False, process_usage_only=True, log_prefix=""):
    import torch

    cpu_used = psutil.cpu_percent()
    ram_used, ram_total = psutil.virtual_memory().used, psutil.virtual_memory().total
    vram_free_device, vram_total = torch.cuda.mem_get_info(device) if "cuda" in device else (0, 0)
    if process_usage_only:
        vram_used = torch.cuda.memory_allocated(device) if "cuda" in device else 0
    else:
        vram_used = vram_total - vram_free_device
    vram_peak = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"] if "cuda" in device else 0

    ram_used /= 1024**3
    ram_total /= 1024**3
    vram_used /= 1024**3
    vram_total /= 1024**3
    vram_peak /= 1024**3

    if log_info:
        from sdkit.utils import log

        msg = log_prefix + " - " if log_prefix else ""
        msg += f"CPU utilization: {cpu_used:.1f}%, System RAM used: {ram_used:.1f} of {ram_total:.1f} GiB"
        if "cuda" in device:
            msg += f", GPU RAM used ({device}): {vram_used:.1f} of {vram_total:.1f} GiB (peak: {vram_peak:.1f} GiB)"
        log.info(msg)

    return cpu_used, ram_used, ram_total, vram_used, vram_total, vram_peak


def get_object_id(o):
    """
    Returns a more-readable object id, than the long number returned by the inbuilt `id()` function.
    Internally, this calls `id()` and converts the number to a base64 string.
    """

    obj_id = id(o)
    obj_id = base64.b64encode(obj_id.to_bytes(8, "big")).decode()
    return obj_id.translate({43: None, 47: None, 61: None})[-8:]


def record_tensor_name(t, name="t", log_info=False):
    """
    Records a name for the given tensor object. Helpful while investigating the source of memory leaks.

    For e.g. you can record variables from across the codebase, and see which one is leaking by calling
    `print_largest_tensors_in_memory()` or `take_memory_snapshot()` after calling `gc()` (for garbage-collection).

    `print_largest_tensors_in_memory()` and `take_memory_snapshot()` print the recorded names for each tensor, if available.
    """

    obj_id = get_object_id(t)
    recorded_tensor_names[obj_id] = [] if obj_id not in recorded_tensor_names else recorded_tensor_names[obj_id]
    recorded_tensor_names[obj_id].append(name)

    if log_info:
        print_tensor_info(t, name)


def print_tensor_info(t, name="t"):
    "Prints a summary of the tensor, for e.g. its size, shape, data type, device etc."

    from sdkit.utils import log

    obj_id = get_object_id(t)
    obj_size = t.nelement() * t.element_size() / 1024**2  # MiB
    log.info(
        f" {name} id: {obj_id}, size: {obj_size} MiB, shape: {t.shape}, requires_grad: {t.requires_grad}, type: {t.dtype}, device: {t.device}"
    )


def get_tensors_in_memory(device):
    """
    Returns the list of all the tensor objects in memory, on the given device.
    **Warning: Do not keep a reference to the returned list longer than necessary, since that will
    prevent garbage-collection of all the tensors in memory.**
    """
    import torch

    device = torch.device(device) if isinstance(device, str) else device
    tensors = []
    objs_in_mem = get_objects()
    for obj in objs_in_mem:
        try:
            if (torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data))) and obj.device == device:
                tensors.append(obj)
        except:
            pass

    return tensors


def print_largest_tensors_in_memory(device, num=10):
    """
    Prints a list of the largest tensors in the given device. Choose the number of objects displayed with the `num` argument.

    Prints the recorded names for each tensor, if recorded using `record_tensor_name()`.

    See also: `take_memory_snapshot()` which is probably more useful for investigating memory leaks.
    """

    entries, total_mem = _get_tensor_entries(device)
    n = len(entries)
    entries = entries[: min(num, n)]

    print(f"== {num} largest tensors on {device} ==")
    print(_fmt_tensors_summary(entries))
    print("---")
    print(f"Total memory occupied on {device} by {n} tensors: {total_mem:.1f} MiB")


def take_memory_snapshot(device, print_snapshot=True):
    """
    Records and prints a list of new tensors (in the device) since the last snapshot (created by calling `take_memory_snapshot()`).

    Prints the recorded names for each tensor, if recorded using `record_tensor_name()`.

    See also: `print_largest_tensors_in_memory()`.
    """
    global tensor_ids_snapshot

    is_first_snapshot = tensor_ids_snapshot is None
    tensor_ids_snapshot = set() if is_first_snapshot else tensor_ids_snapshot

    # take the snapshot
    entries, total_mem = _get_tensor_entries(device)
    curr_tensor_ids = set(entry[0] for entry in entries)
    new_tensor_ids = curr_tensor_ids - tensor_ids_snapshot
    tensor_ids_snapshot = curr_tensor_ids

    # print the diff from the prev snapshot
    if is_first_snapshot or not print_snapshot:
        return

    new_tensor_entries = [entry for entry in entries if entry[0] in new_tensor_ids]
    new_tensors_total_mem = reduce(lambda sum, entry: sum + entry[1], new_tensor_entries, 0)  # MiB
    print(new_tensors_total_mem)
    num_new_tensors = len(new_tensor_ids)

    print(f"== {num_new_tensors} new tensors this snapshot on {device} ==")
    print(_fmt_tensors_summary(new_tensor_entries))
    print("---")
    print(f"Total memory occupied on {device} by {len(entries)} tensors: {total_mem:.1f} MiB")
    print(f"{num_new_tensors} new tensors added {new_tensors_total_mem:.1f} MiB this frame")


def _get_tensor_entries(device, sorted_by_size=True):
    entries = []
    tensors = get_tensors_in_memory(device)
    total_mem = 0
    for t in tensors:
        size = t.nelement() * t.element_size() / 1024**2  # MiB
        obj_id = get_object_id(t)
        entry = [obj_id, size, t.shape, len(get_referrers(t)), t.requires_grad, t.dtype]
        entries.append(entry)
        total_mem += size

    del tensors

    if sorted_by_size:
        entries.sort(key=lambda x: x[0], reverse=True)

    return entries, total_mem


def _fmt_tensors_summary(entries):
    summary = []
    for i, o in enumerate(entries):
        obj_id, size, shape, n_referrers, requires_grad, dtype = o
        known_names = f" ({recorded_tensor_names[obj_id]})" if obj_id in recorded_tensor_names else ""
        summary.append(
            f"{i+1}. Id: {obj_id}{known_names}, Size: {size:.1f} MiB, Shape: {shape}, Referrers: {n_referrers}, requires_grad: {requires_grad}, dtype: {dtype}"
        )

    return "\n".join(summary)
