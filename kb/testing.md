# Testing

sdkit v3 has a small test suite today: one standalone unit test program
for model detection, a scripted end-to-end smoke test, and manual testing against a
running server. This page describes what exists, how to run it, and honestly where
the coverage gaps are. The development workflow is in
[development](development.md).

## Unit tests: model detection

`tests/test_model_detection.cpp` is a self-contained test program for
`inferModelTypeFromTensorKeys` - the function that classifies model files from their
tensor keys (see [models-and-detection](models-and-detection.md)). It builds key
lists by hand and asserts the inferred type:

- GPT-OSS 20B in GGUF naming and in converted safetensors naming → `"llm"`.
- A standard dense LLM (Qwen-style GGUF keys) → `"llm"`.
- CLIP-G (32-layer keys) → `"clip_g"`; CLIP-L (12-layer keys) → `"clip_l"`.
- T5-XXL keys → `"t5xxl"`.
- VAE-like keys and empty key lists → `"vae"`.
- Two edge cases: MoE keys without token embeddings, and an alternate singular-`exp`
  GGUF MoE naming.

Each case is printed as PASSED/FAILED, and `main` exits nonzero if any test failed.

### Building and running the tests

The tests are **not wired into the CMake build** - there is no test target in
`CMakeLists.txt`. Compile them manually against the sources they exercise. The test
needs `model_detection.cpp` and `logging.cpp`, plus the gguf header from the
submodule and the vendored crow headers:

```
g++ -std=c++20 -Iinclude -Ithird_party -Istable-diffusion.cpp/ggml/include \
    tests/test_model_detection.cpp src/model_detection.cpp src/logging.cpp \
    -o build/test_model_detection
./build/test_model_detection
```

(Assuming the stable-diffusion.cpp submodule has been initialized - see
[development](development.md).) The test only exercises pure functions, so no GPU,
models, or the sd.cpp shared library are required; ggml is needed only for the GGUF
header parsing used indirectly through the detection code paths when running the
full `inspectModelType`, which these unit tests don't call.

## Scripted end-to-end check

`generate_image.py` in the repo root is the quickest automated sanity check against
a real server:

1. Start the server (see [development](development.md)) with your model directories.
2. Edit the `model_name` in the script (it defaults to `sd-v1-5.safetensors`).
3. Run `python generate_image.py`.

The script POSTs the model name to `/v1/sdapi/v1/options`, sends a fixed txt2img
request to `/v1/sdapi/v1/txt2img`, and decodes the returned base64 images into
`generated_images/image_<timestamp>_N.png`. A request failure or a missing `images`
field aborts the script. It exercises: option persistence, model resolution and
loading, the full generation path, progress callbacks, and response serialization.

## Manual testing

For anything interactive, run the server and exercise the API directly:

- `GET /v1/internal/ping` - liveness.
- `GET/POST /v1/sdapi/v1/options` - settings round-trip.
- `POST /v1/sdapi/v1/txt2img` and `img2img` - the main flows.
- `POST /v1/internal/progress` with `{"id_task": ...}` - progress and live previews
  while a generation runs.
- `POST /v1/sdapi/v1/extra-batch-images` - upscaling (requires an upscaler name and
  the RealESRGAN directory configured).
- `POST /v1/controlnet/detect` - Canny preprocessing.

The route shapes are documented in [server-and-api](server-and-api.md).

## Coverage gaps

To be explicit about what is *not* tested today:

- No automated tests for image generation, img2img, LoRA/ControlNet handling, or
  progress/interrupt behavior - the scripted smoke test and manual API testing are
  the only generation-path checks.
- No automated tests for the filter code paths (upscaling, Canny preprocessing).
- No test target in CMake and no test runner in CI; the unit tests are run manually
  (the CI workflow builds and uploads artifacts, it does not run tests - see
  [release-process](release-process.md)).
- Nothing validates the HTTP layer itself (route shapes, error responses,
  option-merge semantics) outside of manual testing.

If you add a feature, adding at least pure-logic unit tests in the style of
`test_model_detection.cpp` is the cheapest way to protect it; see the walkthrough in
[development](development.md).
