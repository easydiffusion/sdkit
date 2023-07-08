import os
from shutil import copyfileobj
from tqdm import tqdm


def download_file(url: str, out_path: str):
    """
    Features:
    * Downloads large files (without storing them in memory)
    * Resumes downloads from the bytes it has downloaded already
    * Shows a progress bar

    The remote server needs to support the `Range` header, for resume to work.
    """
    import requests
    from sdkit.utils import log

    start_offset = 0 if not os.path.exists(out_path) else os.path.getsize(out_path)
    res = requests.get(url, stream=True)
    if not res.ok:
        return
    total_bytes = int(res.headers.get("Content-Length", "0"))

    res = requests.get(url, stream=True, headers={"Range": f"bytes={start_offset}-", "Accept-Encoding": "identity"})
    if not res.ok:
        return

    write_mode = "wb" if start_offset == 0 else "ab"

    log.info(f"Downloading {url} to {out_path}")
    with open(out_path, write_mode) as f, tqdm.wrapattr(
        res.raw, "read", initial=start_offset, total=total_bytes, desc="Downloading", colour="green"
    ) as res_stream:
        copyfileobj(res_stream, f)
