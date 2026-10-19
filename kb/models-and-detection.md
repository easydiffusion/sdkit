# Model Management

This page documents how sdkit discovers, names, and classifies model files: the
`ModelManager` component, the directory conventions, the scan process, and the file
format detection layer. The loading side (turning a discovered file into a working
inference context) is described on the page for the feature that uses it - see
[image-generation](image-generation.md),
[image-filters-and-upscaling](image-filters-and-upscaling.md), and
[segmentation](segmentation.md).

## ModelManager

`ModelManager` (`include/model_manager.h`, `src/model_manager.cpp`) holds two pieces
of state:

- `model_directories_`: a map from model type to the directory that holds it.
- `models_`: the index of discovered models, keyed by type. Each entry is a
  `ModelInfo`:

```cpp
struct ModelInfo {
    std::string filename;   // lookup name (see table below)
    std::string full_path;  // absolute path on disk
    ModelType type;
    size_t file_size;
};
```

All mutating and lookup methods take the component's mutex, so a scan triggered by a
refresh route will not race with a generation request that is asking for a model.

### Model types and their directories

Each type has its own setter (`setCheckpointDir`, `setVaeDir`, ...) called at startup
from `main.cpp` using the matching command-line flag. The types:

| ModelType | CLI flag | Typical contents |
|---|---|---|
| `CHECKPOINT` | `--ckpt-dir` | Stable Diffusion checkpoints (SD 1.x/2.x, SDXL, Flux, Qwen, etc.) |
| `VAE` | `--vae-dir` | VAE decoders (often bundled with checkpoints) |
| `HYPERNETWORK` | `--hypernetwork-dir` | Hypernetworks |
| `GFPGAN` | `--gfpgan-models-path` | GFPGAN face restoration (type registered; not yet wired into a route) |
| `REALESRGAN` | `--realesrgan-models-path` | RealESRGAN upscalers |
| `LORA` | `--lora-dir` | LoRA adapters |
| `CODEFORMER` | `--codeformer-models-path` | CodeFormer face restoration (registered; not yet wired into a route) |
| `EMBEDDINGS` | `--embeddings-dir` | Textual inversion embeddings |
| `CONTROLNET` | `--controlnet-dir` | ControlNet adapters |
| `TEXT_ENCODER` | `--text-encoder-dir` | Standalone text encoders (CLIP-L, CLIP-G, T5-XXL, LLM) |
| `SEGMENTATION` | `--segmentation-models-path` | SAM models (see [segmentation](segmentation.md)) |

Only the directories that are actually configured (non-empty) are scanned; a type
with no directory simply has no models. The type names appear verbatim as keys in the
`/v1/sdapi/v1/models` response (see [server-and-api](server-and-api.md)).

## Scanning

`scanDirectoryInternal(directory, type)` does the actual work:

1. If the directory does not exist or is not a directory, it logs a warning and
   returns. Missing directories are treated as "no models" rather than an error, so a
   fresh install without a configured model directory still boots.
2. It walks the tree recursively with `std::filesystem::recursive_directory_iterator`,
   using the `error_code` overload with `skip_permission_denied` so unreadable
   subtrees are skipped instead of aborting the scan.
3. For each regular file whose name ends (case-insensitively) with one of the valid
   extensions, it records a `ModelInfo`. Valid extensions:
   `.sft`, `.safetensors`, `.pth`, `.pt`, `.ckpt`, `.gguf`, `.ggml`.
4. File size is read best-effort; a failed stat is logged verbosely and recorded as
   size 0 rather than failing the scan.

The list is written back into `models_[type]`, replacing whatever was there before.

### Lookup names

Different model types have different naming conventions, matching how clients refer
to them:

| Type | Lookup name (`filename` in `ModelInfo`) | Example |
|---|---|---|
| `CHECKPOINT` | path relative to the checkpoint directory, with extension | `sd3.5_large_turbo.gguf` or `subdir/my_model.safetensors` |
| `CONTROLNET` | filename without extension | `control_v11f1p_sd15_canny` |
| `EMBEDDINGS` | filename without extension | `easynegative` |
| `LORA` | filename without extension | `more_details` |
| everything else (`VAE`, `TEXT_ENCODER`, `REALESRGAN`, `SEGMENTATION`, `GFPGAN`, `CODEFORMER`, `HYPERNETWORK`) | filename with extension | `4x-UltraSharp.pth` |

The lookup name is what clients pass in generation parameters (for example the
`lora_paths` list on a txt2img request, or the upscaler name in an extra-batch request
resolves through the RealESRGAN directory), and what is shown in
`/v1/sdapi/v1/models`.

### Refresh methods

- `refresh()` / `scanAllDirectories()` - rescan every configured type.
- `refreshCheckpoints()` - rescan only `CHECKPOINT`.
- `refreshVaeAndTextEncoders()` - rescan only `VAE` and `TEXT_ENCODER`.

The HTTP routes `/v1/sdapi/v1/refresh-checkpoints` and
`/v1/sdapi/v1/refresh-vae-and-text-encoders` map directly onto the two partial
refreshes, which is useful after a client downloads a new model and wants it to show
up without restarting the server.

### Query methods

| Method | Returns |
|---|---|
| `getModelsByType(type)` | all `ModelInfo` entries for the type (empty list if none) |
| `getModelNamesByType(type)` | just the lookup names |
| `getModelByName(name, type)` | the `ModelInfo` matching a lookup name, or a default-constructed one (`type == CHECKPOINT`, size 0) if not found |
| `hasModel(name, type)` | boolean existence check |
| `getAllModelsGrouped()` | map of type name to list of names, in the shape the `/models` route returns |

Callers should treat "not found" (default `ModelInfo`) rather than throwing; the
feature components check `full_path` emptiness themselves when resolving parameters
(see for example how `ImageGenerator` handles a missing VAE or LoRA).

## Format detection

`include/model_detection.h` / `src/model_detection.cpp` provide a small, standalone
layer that classifies a model file. It is used by tools (`tools/modelinfo.cpp`), by
the C++ unit test (`tests/test_model_detection.cpp`), and can be used by feature code
that needs to know what kind of weights a file contains before loading it.

Three functions make up the API:

```cpp
std::string detectModelFormat(const std::string& model_path);
std::vector<std::string> extractTensorKeys(const std::string& model_path, const std::string& format);
std::string inferModelTypeFromTensorKeys(const std::vector<std::string>& tensor_keys);
std::string inspectModelType(const std::string& model_path);
```

- `detectModelFormat` reads the file's magic header and returns `"gguf"`,
  `"safetensors"`, or `"unknown"`.
- `extractTensorKeys` parses the file (GGUF metadata section, or safetensors header)
  and returns every tensor key name.
- `inferModelTypeFromTensorKeys` looks at those keys and reports one of `"clip_l"`,
  `"clip_g"`, `"t5xxl"`, `"llm"`, or `"vae"` (default when nothing else matches) -
  i.e. what the file *is*, as opposed to what the directory says.
- `inspectModelType` is the convenience combination: format detection plus tensor key
  extraction plus inference, returning `"vae"`, `"clip_l"`, `"clip_g"`, `"t5xxl"`,
  `"llm"`, or `"unknown"`.

This is how the server can tell, for example, whether a file dropped in the
text-encoder directory is a CLIP-L, a CLIP-G, a T5-XXL, or an LLM-style encoder
without any naming convention, which matters for models that bundle different
architectures under similar-looking filenames (see
[image-generation](image-generation.md) for how text encoders are picked at load
time).

## Testing model detection

The detection layer has dedicated unit tests in `tests/test_model_detection.cpp`.
They call `inferModelTypeFromTensorKeys` directly on synthetic lists of tensor keys
(built by hand, no real model files involved), so they run fast and don't depend on
large model downloads. Cases covered:

- **GPT-OSS** in both GGUF (`token_embd.weight`, `blk.N.attn_...`, MoE expert keys
  `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`, `output_norm.weight`) and
  safetensors naming (`model.embed_tokens.weight`, `model.layers.N.self_attn...`,
  `mlp.experts...`) - both should be classified as `llm`.
- **A standard dense LLM** (Qwen-style GGUF keys with plain `ffn_gate` / `ffn_up` /
  `ffn_down` per layer) - classified as `llm`.
- **CLIP-G** and **CLIP-L** safetensors key sets (`text_model.encoder.layers.N...`)
  - distinguished by layer count (32 vs 12).
- **T5XXL** (`encoder.block.N.layer.0.SelfAttention...`,
  `encoder.block.N.layer.1.DenseReluDense...`) - classified as `t5xxl`.
- **VAE-like keys** (`decoder.conv_in.weight`, `encoder.conv_out.weight`, ...) -
  classified as `vae` (the default).
- **Empty key list** - defaults to `vae`.
- **Edge cases**: a minimal GPT-OSS key set (MoE MLP + attention + `output_norm`,
  with no `token_embd`) is still detected as `llm`, and an alternate GGUF MoE spelling
  using the singular `ffn_gate_exp` / `ffn_up_exp` / `ffn_down_exp` is also accepted.

The tests use a small home-grown assertion macro (they don't depend on a testing
framework) and print a `PASSED` / `FAILED` line per case, exiting non-zero if any
fail. See [testing](testing.md) for how to build and run them.

### modelinfo tool

`tools/modelinfo.cpp` is a small CLI built only when CMake is configured with
`SDKIT_BUILD_TOOLS=ON` (see [architecture](architecture.md)):

```
modelinfo [--print-keys] <model_path>
```

It prints the detected model type (`clip_l`, `clip_g`, `t5xxl`, `llm`, or `vae`) and
the tensor count, and with `--print-keys` also lists every tensor key. It is handy
for debugging "why is this file being classified as X" questions during model
loading.

## Where model files come from

sdkit itself does not download models; it only scans and loads them. For development
and CI, `CI.md` documents how models are fetched onto the build runners (for example
downloading specific SD 3.5 / Flux / Qwen GGUF and safetensors files into the expected
directories before the tests or demos run). For end users, the model directories are
simply populated ahead of time; the refresh routes let them be updated at runtime.
