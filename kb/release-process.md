# Release process

This page describes how a sdkit release is cut: the release command, the platforms
that get built, how artifacts are hashed and uploaded, and where the CI runners come
from. Build mechanics are in [building-and-platforms](building-and-platforms.md).

## The release command

Releases are cut from a machine with the toolchains set up (see
[Runner setup](#runner-setup) below):

```
git clone https://github.com/easydiffusion/sdkit.git
git checkout v3
git submodule init
git submodule update --recursive

python -m scripts.ci
```

`scripts/ci.py` takes an optional version tag as its first argument (defaulting to
`v3.5.0` in the current code) and runs the two release phases in order:

1. `python -m scripts.build_all` - builds every platform for the current OS.
2. `python -m scripts.upload_all --tag <tag>` - uploads the artifacts to a GitHub
   release.

You can also run the two phases separately, which is useful when iterating: build
everything, inspect `build/<target>/release_artifacts/`, then upload with the tag
you want.

## The platform matrix

`scripts/build_all.py` builds every target for the host OS (the macOS Metal targets
run on a macOS CI runner via `.github/workflows/ci_metal.yaml`, see
[building-and-platforms](building-and-platforms.md)):

| OS | Targets |
|---|---|
| Windows | `win-x64-cpu`, `win-x64-cuda` (one per CUDA SM variant), `win-x64-vulkan`, `win-arm64-vulkan` |
| Linux | `linux-x64-cpu`, `linux-x64-cuda` (one per CUDA SM variant), `linux-x64-vulkan` |
| macOS | `mac-arm64-metal` and `mac-x64-metal` (deployment targets 11.0 / 10.15) |

The exact target names follow the `<os>-<arch>-<platform>-<variant>` pattern
produced by `scripts/build/common.py`, and CUDA builds fan out into one target per
compute-capability variant (sm60, sm75, sm80, sm86, sm89, sm90, sm100, sm120 in the
current variant list).

Each target's release files (the `sdkit` executable plus the stable-diffusion and
ggml shared libraries) are compressed into individual `.tar.gz` archives in
`build/<target>/release_artifacts/`, together with a `<target>-manifest.json`
listing every file, its SHA-256 hash, and its archive name.

## What upload_all.py does

`scripts/upload_all.py` finds every directory under `build/` that contains a
`release_artifacts` subdirectory and, for each one, runs
`python -m scripts.upload --tag <tag> --target <target> [--repo owner/repo]`. Flags:

- `--tag` (required) - the GitHub release tag to upload to. The release must already
  exist on GitHub; the script does not create it.
- `--repo` - defaults to `easydiffusion/sdkit`.
- `--dry-run` - show what would be uploaded without touching anything.
- `--force` - re-upload files even if their hashes match what's already published.

The uploader (`scripts/upload/__main__.py`) authenticates with a GitHub token taken
from a `.creds.json` file in the working directory (`{"github_token": "..."}`) or
the `GITHUB_TOKEN` environment variable. For each target it:

1. Fetches the release's existing assets (paginated).
2. Downloads the remote manifest and compares it with the local manifest. Manifests
   are skipped when the JSON matches exactly, deleted and re-uploaded otherwise.
3. For each file in the manifest, compares the local SHA-256 with the remote
   manifest's hash; matching files are skipped, differing or missing files are
   deleted (if present) and uploaded as `application/gzip` (manifests as JSON).

So an upload is idempotent: re-running it after a partial failure uploads only the
missing or changed files. `scripts/upload/hash_utils.py` provides the hashing
helpers:

- `hash_file_quick(path)` - hashes a local file.
- `hash_url_quick(url)` - hashes a remote file using HTTP range requests.

Both avoid hashing gigabytes when possible: files smaller than 3 MB are hashed in
full, larger ones by sampling 64 KB at the start (offset 1 MB), the middle, and the
end (1 MB from the end) and SHA-256ing the concatenated samples. Note that during
uploads the comparison logic primarily uses the full SHA-256 values recorded in the
manifests (computed by `scripts/build/common.py` with `hashlib.sha256` over the raw
file), with the quick hashes used when a remote manifest is missing a hash entry.

## Runner setup

The release machine needs the toolchains for each OS set up as described below.

### Windows

1. Download and run ["Build Tools for Visual Studio"](https://visualstudio.microsoft.com/downloads/).
2. Install or Edit the following components:
   - MSVC - VS 2022 C++ x64/x86 build tools
   - MSVC - VS 2022 C++ ARM64 build tools
   - C++ Clang Compiler for Windows
   - MSBuild support for LLVM (clang-cl) toolset
   - C++ CMake tools for Windows
3. Install [Ninja](https://github.com/ninja-build/ninja/releases) and make it
   available on `PATH`.

### Linux

1. Download a `runfile (local)` from
   [CUDA Toolkit downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04).
2. Download an SDK tarball from
   [Vulkan SDK downloads](https://vulkan.lunarg.com/sdk/home).

#### Non-Linux host (using Docker)

The Linux build environment can run inside a Docker `ubuntu:latest` container (with
`--gpus=all`) on a non-Linux host, with the SDKs copied into a named volume:

```bash
docker volume create ed-build
docker run -it --gpus=all -v ed-build:/ed -v /path/to/models:/models --name ed-build ubuntu:latest

# in a different tab
docker cp vulkansdk-linux-x86_64-1.4.328.1.tar.xz ed-build:/ed/
docker cp cuda_12.8.1_570.124.06_linux.run ed-build:/ed/
```

#### Inside Linux

```bash
apt update
apt install git curl vim python-is-python3 python3 xz-utils -y
apt install g++ ccache cmake -y

# install ninja
cd
mkdir bin
wget https://github.com/ninja-build/ninja/releases/download/latest/ninja-linux.zip
unzip ninja-linux.zip
mv ninja bin/

# install CUDA Toolkit
cd /ed
./cuda_12.8.1_570.124.06_linux.run
echo "export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/lib64:\$PATH" >> ~/.bashrc

# install Vulkan SDK
cd /ed
tar -xf vulkansdk-linux-x86_64-1.4.328.1.tar.xz
mv 1.4.328.1 vulkan-1.4.328.1
echo "source /ed/vulkan-1.4.328.1/setup-env.sh" >> ~/.bashrc
apt install vulkan-tools vulkan-validationlayers -y
```

Note that `setup-env.sh` from the Vulkan SDK is sourced in `~/.bashrc` so the build
finds Vulkan.

## Checklist for a release

1. Initialize submodules after cloning (see [building-and-platforms](building-and-platforms.md)).
2. Create the GitHub release with the target tag if it doesn't exist yet.
3. `python -m scripts.ci <tag>` (or build_all + upload_all).
4. Verify each target directory in `build/` has a manifest and that the GitHub
   release has all the expected assets for the platforms you built.
