# Testing

This page describes how sdkit is tested: what automated coverage exists today, how to
run it, and the manual/scripted checks used for everything else. It's honest about
the gaps - most of the heavy inference features are verified manually, not by CI.

## What exists today

### Unit tests (`tests/test_model_detection.cpp`)

The only automated unit tests cover the model-detection layer
(see [models-and-detection](models-and-detection.md)). They call
`inferModelTypeFromTensorKeys` directly with hand-built lists of tensor keys, so they
need no model files, no GPU, and run in well under a second.

The cases covered:

- **GPT-OSS in GGUF naming** (`token_embd.weight`, `blk.N.attn_...`, MoE expert keys
  `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`, `output_norm.weight`) -
  classified as `llm`.
- **GPT-OSS in safetensors naming** (`model.embed_tokens.weight`,
  `model.layers.N.self_attn...`, `mlp.experts...`) - also classified as `llm`.
- **A standard dense LLM** (Qwen-style GGUF keys with plain `ffn_gate` / `ffn_up` /
  `ffn_down` per layer) - classified as `llm`.
- **CLIP-G** (`text_model.encoder.layers.<0..31>...`) - classified as `clip_g`.
- **CLIP-L** (`text_model.encoder.layers.<0..11>...`) - classified as `clip_l`,
  distinguished from CLIP-G purely by layer count.
- **T5XXL** (`encoder.block.N.layer.0.SelfAttention...`,
  `encoder.block.N.layer.1.DenseReluDense...`) - classified as `t5xxl`.
- **VAE-style keys** (`decoder.conv_in.weight`, `encoder.conv_out.weight`, ...) -
  classified as `vae` (the fallback).
- **Empty key list** - defaults to `vae`.
- **Edge cases** - a minimal GPT-OSS key set (MoE MLP + attention + `output_norm`,
  with no `token_embd`) is still detected as `llm`, and the singular MoE spelling
  (`ffn_gate_exp` / `ffn_up_exp` / `ffn_down_exp`) is accepted too.

The tests use a small home-grown `ASSER_EQ` macro (no test framework dependency) that
prints `PASSED` / `FAILED` per case and exits non-zero if any case failed, so it
works both interactively and as a CI step.

### What's *not* covered by automated tests

There are no automated tests for:

- **Generation** (txt2img / img2img) - needs real checkpoints and a working inference
  backend.
- **Image filters and upscaling** - needs the RealESRGAN upscaler weights and a
  running server to exercise the routes.
- **Segmentation** - needs a SAM model and a running server.
- **The HTTP layer itself** - no route-level tests; the routes are exercised manually
  through the demo pages or scripts.

These features are verified by the two mechanisms below. If you add automated
coverage for any of them, extend `tests/` in the style of
`test_model_detection.cpp` and update this page.

## How to build and run the unit tests

The tests are a plain C++ source with a `main()`; they include `../include/` headers
relative to the file and don't link the sdkit library, so they compile quickly:

```bash
cd build
g++ -std=c++20 ../tests/test_model_detection.cpp -o test_model_detection
./test_model_detection
```

Expected output is one `PASSED`/`FAILED` line per case, followed by a summary line
(`=== Results: N passed, 0 failed ===`) and a zero exit code on success.

For convenience, the CMake build can also produce the `modelinfo` tool
(`SDKIT_BUILD_TOOLS=ON`), which is useful when debugging *why* a real model file is
classified the way it is (see [models-and-detection](models-and-detection.md)) - it's
the manual counterpart to these unit tests for real files.

## Scripted end-to-end check: `generate_image.py`

`generate_image.py` at the repository root is a scripted smoke test for the most
important end-to-end path - text-to-image over HTTP. Start the server, then run:

```bash
python generate_image.py
```

It does the following (see [development](development.md) for the script's internals):

1. POSTs `{"sd_model_checkpoint": <name>}` to `/v1/sdapi/v1/options` to select a
   model.
2. POSTs a txt2img request to `/v1/sdapi/v1/txt2img`.
3. Writes each returned base64 image to a timestamped `.png` in the current
   directory.

A successful run means the server booted, scanned its models, loaded the checkpoint,
ran inference on the configured backend, and returned valid PNGs - a good check after
build, backend, or server-layer changes. It doesn't cover img2img, ControlNet,
upscaling, or segmentation; use the demo pages for those.

## Manual testing via the demo pages

The `demo/` pages exercise the remaining routes interactively from a browser
(see [demo-pages](demo-pages.md) for how they're served and what each page calls):

- **`segmentation-demo.html`** - `/v1/sdapi/v1/models` and `/v1/sdapi/v1/segment`.
  Load an image, add points/boxes/text prompts, and check that masks and scores come
  back and render correctly.
- **`upscaling-demo.html`** - `/v1/sdapi/v1/models` and
  `/v1/sdapi/v1/extra-batch-images`. Upscale an image with a RealESRGAN model and
  compare before/after.
- **`filters-demo.html`** - `/v1/controlnet/detect`. Upload an image and verify the
  Canny edge map that comes back.

A typical manual pass after a change:

1. Build and start the server from `build/` with the model directories you need
   (see [development](development.md)).
2. Open `http://127.0.0.1:8188/`, work through each demo page relevant to your
   change.
3. Run `python generate_image.py` as a final txt2img smoke test.
4. Watch the server log (`--log-level info` or `debug`) for warnings/errors.

## Coverage gaps and how to think about them

The honest summary: the detection layer is the only logic with automated regression
coverage. Everything else relies on manual verification. That means:

- Changes to `ImageGenerator`, `ImageFilters`, or `ImageSegmenter` should always be
  exercised through the corresponding demo page (or `generate_image.py` for the
  generator) before being considered done.
- Changes to routes in `server.cpp` should be hit from the browser demo or a
  `curl`/Python request to verify the request/response shape hasn't drifted.
- Changes to `ModelManager` scanning can be partially verified by pointing
  `--ckpt-dir` (etc.) at a directory and checking the `/v1/sdapi/v1/models` output.

If a change is risky and purely manual verification feels insufficient, add a unit
test in `tests/` following the existing pattern - it's cheap for pure logic, and the
project's convention so far has been "automate what doesn't need model weights".
