# Models and detection

This page covers how sdkit discovers models on disk (`ModelManager`) and how it
inspects individual model files to figure out what they are (`model_detection`).
Model loading itself is described in [image-generation](image-generation.md).

## ModelManager

`ModelManager` (`include/model_manager.h`, `src/model_manager.cpp`) is a directory
scanner and index. It does not load models into memory - actual loading happens
lazily in the `ImageGenerator` and `ImageFilters` when a generation or upscale needs
a model. `ModelManager` answers "which files exist and where are they?"

### Supported model types

`ModelType` enumerates the categories the manager indexes:

- `CHECKPOINT` - the main Stable Diffusion model.
- `VAE` - standalone VAE files.
- `HYPERNETWORK` - hypernetworks.
- `GFPGAN` and `CODEFORMER` - face-restoration models.
- `REALESRGAN` - upscaler models.
- `LORA` - LoRA adapters.
- `EMBEDDINGS` - textual inversion embeddings.
- `CONTROLNET` - ControlNet models.
- `TEXT_ENCODER` - text encoder files (e.g. CLIP, T5, LLM modules).

### Directory conventions

Each type has a setter (`setCheckpointDir`, `setVaeDir`, `setHypernetworkDir`,
`setGfpganModelsPath`, `setRealesrganModelsPath`, `setLoraDir`,
`setCodeformerModelsPath`, `setEmbeddingsDir`, `setControlnetDir`,
`setTextEncoderDir`) that `main.cpp` wires to the matching `--*-dir`/`--*-path` CLI
argument (see [server-and-api](server-and-api.md)). Only the RealESRGAN, LoRA, and
embeddings directories have getters, because those are the ones other components read
back (`ImageFilters` for upscalers, `ImageGenerator` for LoRA and embedding
directories).

### Scanning

`scanAllDirectories()` walks each configured directory recursively (skipping
permission-denied entries and non-regular files) and builds a per-type list of
`ModelInfo` records. A file counts as a model if its extension is one of `.sft`,
`.safetensors`, `.pth`, `.pt`, `.ckpt`, or `.gguf` (case-insensitive). Files are
logged every 50 entries so long scans produce feedback, and filesystem errors are
handled per-entry rather than aborting the scan.

The lookup name (`ModelInfo::filename`) depends on the type:

- `CHECKPOINT`: relative path from the checkpoint directory, with extension (so
  subfolders like `sd15/foo.safetensors` are addressable).
- `CONTROLNET`, `EMBEDDINGS`, `LORA`: file name without extension.
- Everything else: plain file name.

Queries: `getModelsByType`, `getModelNamesByType`, `getModelByName`, `hasModel`, and
`getAllModelsGrouped` (a map of type name → file names). The server exposes refresh
routes for checkpoints and for VAE/text encoders; other types are refreshed only by
restarting or via `refresh()` if called programmatically. None of the query methods
are reachable over HTTP right now - see [server-and-api](server-and-api.md).

All state is guarded by a mutex, so scans can run while requests read the index.

## model_detection

`model_detection.h` / `src/model_detection.cpp` answers a different question: given
a model file, what kind of model is it? This is used when the user configures
`forge_additional_modules` in the options - sdkit must figure out whether a given
file is a VAE, a CLIP text encoder, a T5 encoder, or an LLM, since the sd.cpp context
takes each through a different parameter. It is also used by the `modelinfo` tool
(see below) and is unit-tested in `tests/test_model_detection.cpp`.

### Detecting the format

`detectModelFormat(path)` reads the first 8 bytes:

- Files starting with the `GGUF` magic are `"gguf"`.
- Otherwise, the 8 bytes are interpreted as a little-endian uint64 header size; if it
  is between 8 and 100 MB, the file is assumed to be `"safetensors"` (the format
  stores a JSON header whose length is the first 8 bytes).
- Anything else is `"unknown"`.

### Extracting tensor keys

`extractTensorKeys(path, format)`:

- For safetensors, reads the 8-byte header length, then the JSON header, and returns
  every top-level key except `__metadata__` (parsed with the vendored crow JSON
  parser). The header is sanity-checked to be at most 100 MB.
- For GGUF, opens the file with `gguf_init_from_file` using `no_alloc = true` (so no
  tensor data is loaded) and returns `gguf_get_tensor_name` for each tensor. The
  gguf header comes from the submodule's ggml: `stable-diffusion.cpp/ggml/include/gguf.h`.

### Inferring the model type

`inferModelTypeFromTensorKeys(keys)` inspects lowercased tensor names:

- **T5-XXL**: keys matching both `selfattention` and `denserelu` patterns → `"t5xxl"`.
- **LLM**: combinations of token embedding (`token_embd.weight` or
  `embed_tokens.weight`), attention (`blk.*.attn_*` or `model.layers.*.self_attn.*`),
  MLP (dense `ffn_*` or `mlp.*_proj`), output norm, QK norm, or MoE expert/router
  keys (covering both raw GGUF naming and converted safetensors naming) → `"llm"`.
  This detects dense LLMs (e.g. Qwen) and MoE models (e.g. GPT-OSS).
- **CLIP**: keys with `text_model`, `transformer.`, `text_projection`, or
  `position_ids`. The largest numeric layer index found in the keys decides
  CLIP-L (≤ 19, i.e. the 12-layer variant) versus CLIP-G (≥ 20, the 32-layer
  variant) → `"clip_l"` or `"clip_g"`.
- **Everything else** (including empty key lists) defaults to `"vae"`.

`inspectModelType(path)` composes the three steps and returns `"unknown"` when the
format can't be detected or no tensors are found.

## The modelinfo tool

`tools/modelinfo.cpp` is a small CLI built when the project is configured with
`-DSDKIT_BUILD_TOOLS=ON` (see `tools/CMakeLists.txt`). It reuses
`model_detection.cpp` and `logging.cpp`:

```
modelinfo [--print-keys] <model_path>
```

It prints the detected model type (one of `vae`, `clip_l`, `clip_g`, `t5xxl`, `llm`,
or `unknown`) and the number of tensors; with `--print-keys` it also prints every
tensor key name. It's the quickest way to check how a new model file will be
classified before wiring it into `forge_additional_modules`.

## Tests

`tests/test_model_detection.cpp` is a standalone test program that exercises
`inferModelTypeFromTensorKeys` with hand-built key lists: GPT-OSS 20B in both GGUF
and safetensors naming, a standard dense LLM (Qwen-style GGUF), CLIP-G and CLIP-L
safetensors key sets, T5-XXL keys, VAE-like keys, empty keys, and two edge cases
(MoE keys without token embeddings, and an alternate singular-`exp` GGUF MoE
naming). It reports pass/fail counts and exits with a nonzero status on failure.
Note that the tests are not wired into the CMake build; see
[testing](testing.md) for how to compile and run them.
