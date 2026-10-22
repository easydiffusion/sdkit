# Installation

There are two ways to get sdkit v3: download prebuilt binaries from the GitHub
releases, or build from source.

## Downloading prebuilt binaries

sdkit publishes per-platform builds on the
[GitHub releases page](https://github.com/easydiffusion/sdkit/releases). Each release
contains a set of `.tar.gz` archives plus a small JSON manifest per platform target.
Your application fetches the manifest, downloads every file it lists, verifies each
archive's SHA-256 checksum, and extracts it into the backend directory.

### Target naming

Every build is identified by a **target** of the form:

```
<os>-<arch>-<platform>-<variant>
```

| Component | Values | Meaning |
|---|---|---|
| `<os>` | `win`, `linux`, `mac` | Operating system |
| `<arch>` | `x64`, `arm64` | CPU architecture |
| `<platform>` | `cpu`, `cuda`, `vulkan`, `metal` | Acceleration backend |
| `<variant>` | `any`, `sm<arch>` | For CUDA builds, the minimum GPU compute capability (e.g. `sm120`); `any` otherwise |

Examples:

- `win-x64-cuda-sm120` - Windows, 64-bit x86, CUDA for Ada-and-newer GPUs
- `linux-x64-vulkan-any` - Linux, 64-bit x86, Vulkan (AMD/Intel/NVIDIA)
- `mac-arm64-metal-any` - macOS, Apple Silicon, Metal
- `win-x64-cpu-any` - Windows, CPU-only fallback

The platform is picked from the GPUs present on the machine: CUDA when an NVIDIA GPU
is detected, Metal on macOS, Vulkan for other GPUs, and CPU as the fallback. This is
exactly the logic used by the
[Easy Diffusion backend](https://github.com/easydiffusion/easydiffusion/blob/main/ui/easydiffusion/backends/sdkit3.py).

### The release-manifest flow

For a release version `v3.5.0` (check the releases page for the current version), the
manifest lives at:

```
https://github.com/easydiffusion/sdkit/releases/download/v3.5.0/<target>-manifest.json
```

The manifest is a JSON object with a single `files` key. Each entry maps a file name
to its `sha256` checksum and its download `uri`:

```json
{
  "files": {
    "sdkit.tar.gz": {
      "sha256": "…",
      "uri": "v3.5.0/win-x64-cuda-sm120/sdkit.tar.gz"
    }
  }
}
```

For every file: download `<base>/<uri>`, verify the SHA-256 checksum, and extract the
`.tar.gz` into the backend directory. After extraction the directory contains the
`sdkit` server executable (`sdkit.exe` on Windows) plus its shared libraries.

If the manifest returns `404`, the target does not exist for that release -
[request the platform build](https://github.com/easydiffusion/sdkit/issues) or ask on
the [Discord](https://discord.com/invite/u9yhsFmEkB).

### Standalone download script

This script reproduces the manifest flow with nothing but the Python standard
library plus `requests`. Adjust `TARGET` and `VERSION` for your platform.

```python
#!/usr/bin/env python3
"""Download the sdkit v3 backend for the given target."""

import hashlib
import tarfile
import tempfile

import requests

BASE_URL = "https://github.com/easydiffusion/sdkit/releases/download"
VERSION = "v3.5.0"
TARGET = "win-x64-cuda-sm120"       # e.g. linux-x64-vulkan-any, mac-arm64-metal-any
DEST_DIR = "backends/sdkit3/" + TARGET


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest_url = f"{BASE_URL}/{VERSION}/{TARGET}-manifest.json"
    response = requests.get(manifest_url)
    response.raise_for_status()          # 404 => target build does not exist
    files = response.json()["files"]

    for filename, info in files.items():
        download_url = f"{BASE_URL}/{info['uri']}"
        print(f"Downloading {filename} from {download_url}")

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

            actual = sha256_of(tmp_path)
            expected = info["sha256"]
            if actual != expected:
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: expected {expected}, got {actual}"
                )

            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(DEST_DIR)
            print(f"Extracted {filename} into {DEST_DIR}")
        finally:
            import os
            os.unlink(tmp_path)

    print("Done.")


if __name__ == "__main__":
    main()
```

After the files are extracted, continue with [Running the server](server.md).

## Building from source

To build instead of downloading:

1. Clone the repository and its submodules:

    ```bash
    git clone https://github.com/easydiffusion/sdkit.git
    cd sdkit
    git checkout v3
    git submodule init
    git submodule update --recursive
    ```

2. Build with CMake (C++20, CMake ≥ 3.16). The top-level
   [`CMakeLists.txt`](https://github.com/easydiffusion/sdkit/blob/v3/CMakeLists.txt)
   builds the `sdkit` executable and the vendored stable-diffusion.cpp engine:

    ```bash
    cmake -B build
    cmake --build build
    ```

3. Or use the all-in-one build script, which builds every platform/variant listed for
   your operating system (CPU, CUDA, Vulkan, Metal):

    ```bash
    python -m scripts.build_all
    ```

    To build a single target, use
    `python -m scripts.build --platform <cpu|cuda|vulkan|metal> --arch <x64|arm64>`.