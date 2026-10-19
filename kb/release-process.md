# Release Process

This page describes how a sdkit release is cut: the command that drives it, the
platform matrix that gets built, how the artifacts are hashed and uploaded to GitHub,
and the machine setup required to run it. The step-by-step runner setup lives in
`CI.md` at the repository root - this page links to it rather than repeating the
commands.

## The release command

The whole flow is orchestrated by `scripts/ci.py`:

```bash
python -m scripts.ci v3.4.1
```

The single positional argument is the release tag (it defaults to `v3.4.1` if omitted,
so always pass one explicitly). Running it does exactly two things, in order:

1. `python -m scripts.build_all` - builds every platform/arch combination for the
   current OS (see [building-and-platforms](building-and-platforms.md) for how the
   build scripts work).
2. `python -m scripts.upload_all --tag <tag>` - uploads the artifacts that were just
   built to a GitHub release.

You can also run the two halves separately, which is useful when only one of them
needs to run (e.g. rebuilding after a fix before re-uploading):

```bash
python -m scripts.build_all                      # build everything for this OS
python -m scripts.upload_all --tag v3.4.1        # upload what was built
```

## The platform matrix

`scripts/build_all.py` selects the matrix based on `platform.system()`, so one
invocation builds everything for the OS it's run on:

| OS | (backend, arch) pairs |
|---|---|
| Windows | `cpu/x64`, `cuda/x64`, `vulkan/x64`, `vulkan/arm64` |
| Linux | `cpu/x64`, `cuda/x64`, `vulkan/x64` |
| macOS | `metal/arm64` (min macOS 11.0), `metal/x64` (min macOS 10.15) |

For the CUDA builds, `scripts/build/platform_cuda.py` adds one build variant per
compute capability (`sm60`, `sm75`, `sm80`, `sm86`, `sm89`, `sm90`, `sm100`,
`sm120`), so a single `build_all` run on a CUDA-equipped machine produces one artifact
set per GPU generation. The backend-specific compile flags and the reasoning behind
them are documented in [building-and-platforms](building-and-platforms.md).

Each target writes its artifacts to `build/<target>/release_artifacts/` as
`<target>-<filename>.tar.gz` files, together with a `<target>-manifest.json` mapping
each original filename to its SHA256 and the artifact URI. The manifest is what the
upload step consumes; see [building-and-platforms](building-and-platforms.md) for how
the manifest is created during the build.

## What upload_all.py does

`scripts/upload_all.py` finds every target that has release artifacts, then invokes
the per-target uploader for each one:

```bash
python -m scripts.upload --tag <tag> --target <target> --repo <owner/repo>
```

It supports `--dry-run` (print what would be uploaded without uploading) and
`--force` (re-upload files even if a matching asset already exists). The default
repository is `easydiffusion/sdkit`.

### The per-target uploader

`scripts/upload/__main__.py` implements a `GitHubReleaseUploader` that talks to the
GitHub REST API (`https://api.github.com`). Authentication comes from either a
`.creds.json` file in the working directory (`{"github_token": "..."}`) or the
`GITHUB_TOKEN` environment variable - the token needs write access to releases in the
target repo.

For each target the uploader:

1. Looks up the release by tag. If the release doesn't exist yet, the script reports
   that and stops for that target (releases are created out-of-band, e.g. manually
   or by a CI job, before running the upload).
2. Lists the release's existing assets, paginating through the GitHub API 100 at a
   time.
3. Compares the local `release_artifacts` manifest against the remote assets. A local
   file that matches an existing asset by name **and** content hash is skipped; a
   file whose hash differs is deleted remotely and re-uploaded; a file with no remote
   counterpart is uploaded as a new asset.

So re-running the upload for the same tag is idempotent: only files that actually
changed (or are new) are transferred.

## How artifacts are hashed

Two hashing mechanisms are in play, both in `scripts/upload/hash_utils.py`:

- **SHA256 at build time** - `common.compute_sha256` hashes each distributable file
  with a streaming 4 KB read and records the digest in the target's manifest. This is
  the authoritative hash used by the uploader to decide whether a remote asset is
  up to date.
- **Quick hash for comparisons** - `hash_file_quick` / `hash_url_quick` implement a
  cheap content fingerprint for files larger than 3 MB: 64 KB is read from 1 MB into
  the file, from the middle, and from 1 MB before the end, concatenated, and SHA256'd
  (smaller files are hashed in full). The `hash_url_quick` variant does the same over
  HTTP range requests, which lets the uploader fingerprint a remote asset without
  downloading it entirely.

The quick hash is a fingerprint, not a cryptographic guarantee - it's used for
"does this remote file look the same as this local one" checks, where reading a few
hundred kilobytes beats downloading gigabytes.

## Runner setup

The release command needs a machine that can build every backend for its OS.
`CI.md` documents the setup for the runners used for official releases:

- **Windows**: MSVC + Clang toolchains, Ninja, and the CUDA toolkit (for the
  `cuda/x64` target).
- **Linux**: CUDA toolkit 12.8, the Vulkan SDK, and `vulkan-tools` /
  `vulkan-validationlayers` (for the `cuda/x64` and `vulkan/x64` targets).
- **Docker**: a Linux container recipe is documented there too, for repeatable
  CUDA/Vulkan builds without polluting a host machine.

Follow `CI.md` for the exact install commands. Note that macOS releases are cut on a
macOS runner with the Metal backend; no extra SDK beyond Xcode is needed
(see [building-and-platforms](building-and-platforms.md)).

## Prerequisites checklist before running a release

1. All git submodules are initialized and updated (`git submodule update --recursive`),
   and the submodule commits are the ones you want to ship.
2. The working tree is clean and the version tag exists (or will be created) in the
   repo.
3. A GitHub token with release-write access is available (`.creds.json` or
   `GITHUB_TOKEN`).
4. The GitHub release for the tag already exists (the uploader won't create it).
5. `nvcc` / `vulkaninfo` are on `PATH` if you're cutting a CUDA/Vulkan release
   (see [building-and-platforms](building-and-platforms.md)).
