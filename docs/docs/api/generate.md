# Image generation

Reference for the generation endpoints: [`txt2img`](#post-v1sdapiv1txt2img),
[`img2img`](#post-v1sdapiv1img2img), [progress polling](#progress-polling), and
[interrupt](#interrupt).

Both generation endpoints share the same parameter set (below); `img2img` adds
source-image fields. **They block until generation completes** - for progress and
live previews, send the request with a `force_task_id` and poll the progress
endpoint (see [Progress polling](#progress-polling)).

## `POST /v1/sdapi/v1/txt2img`

Text-to-image generation.

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/txt2img \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a photograph of an astronaut riding a horse",
           "negative_prompt": "blurry, low quality",
           "width": 512, "height": 512,
           "steps": 20, "cfg_scale": 7.0,
           "sampler_name": "Euler a", "scheduler": "karras"
         }'
```

Response (`200`):

```json
{
  "images": ["<base64 png>", "..."],
  "info": "{\"prompt\": \"...\", \"negative_prompt\": \"...\", \"steps\": 20, \"cfg_scale\": 7.0, \"seed\": 42, \"width\": 512, \"height\": 512}"
}
```

- `images` - array of base64-encoded PNGs, one per `batch_size`.
- `info` - a JSON *string* (double-encoded) echoing the generation parameters.

On failure: `500` with `{"message": "Generation failed: ..."}` - e.g. no model
selected, checkpoint not found, or context creation failed. The task state is
completed with the error, so the failure is also visible via the
[progress endpoint](#progress-polling).

### Parameters

All fields are optional unless noted; missing fields use the defaults below.

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | `""` | The prompt |
| `negative_prompt` | string | `""` | Negative prompt |
| `width` | int | `512` | Image width in pixels (with ControlNet: rounded to a multiple of 64) |
| `height` | int | `512` | Image height in pixels (with ControlNet: rounded to a multiple of 64) |
| `steps` | int | `20` | Number of sampling steps |
| `cfg_scale` | float | `7.0` | Classifier-free guidance scale |
| `seed` | int | `-1` | Random seed. `-1` (or omitted) uses the current wall-clock time, so each run differs; any other value reproduces the same image |
| `batch_size` | int | `1` | Number of images to generate per request |
| `sampler_name` | string | `Euler a` (sd.cpp `euler_a`) | Sampler; A1111/Forge-style names are translated (see [sampler mapping](#sampler-name-mapping)) |
| `scheduler` | string | `automatic` (sd.cpp `discrete`) | Noise scheduler; A1111/Forge-style names are translated (see [scheduler mapping](#scheduler-name-mapping)) |
| `lora_paths` | string[] | `[]` | LoRA lookup names (filename without extension, resolved in `--lora-dir`) |
| `lora_alphas` | float[] | `[]` | LoRA strengths, zipped with `lora_paths`. If shorter than `lora_paths`, a warning is logged and the extra paths are ignored |
| `force_task_id` | string | `"default_task"` | Task ID for [progress polling](#progress-polling) |
| `ref_images` | string[] | `[]` | Base64 reference images, for vision-capable models (e.g. Qwen). Resized to the generation size by default |
| `alwayson_scripts` | object | - | ControlNet extension point; see [Filters & ControlNet](filters.md#controlnet-inside-generation-requests) |

### Sampler name mapping

A1111/Forge sampler names are translated to stable-diffusion.cpp names before use:

| A1111 / webui name | sd.cpp name |
|---|---|
| `Euler` | `euler` |
| `Euler a` | `euler_a` |
| `Heun` | `heun` |
| `DPM2` | `dpm2` |
| `DPM++ 2S a` | `dpm++2s_a` |
| `DPM++ 2M` | `dpm++2m` |
| `DPM++ 2M v2` | `dpm++2mv2` |
| `IPNDM` | `ipndm` |
| `IPNDM_V` | `ipndm_v` |
| `LCM` | `lcm` |
| `DDIM` | `ddim_trailing` |
| `TCD` | `tcd` |

Any other name is passed through unchanged to stable-diffusion.cpp.

### Scheduler name mapping

| A1111 / webui name | sd.cpp name |
|---|---|
| `automatic` | `discrete` |
| `uniform` | `discrete` |
| `karras` | `karras` |
| `exponential` | `exponential` |
| `sgm_uniform` | `sgm_uniform` |
| `simple` | `simple` |
| `align_your_steps` | `ays` |
| `align_your_steps_GITS` | `gits` |

Any other name is passed through unchanged.

## `POST /v1/sdapi/v1/img2img`

Image-to-image generation. Takes all `txt2img` parameters plus:

| Field | Type | Default | Description |
|---|---|---|---|
| `init_images` | string[] | - | Base64 source image(s). **Only the first entry is used** |
| `mask` | string | - | Base64 mask image; resized to the generation size. If omitted, an all-white (fully unmasked) mask is used |
| `denoising_strength` | float | `0.75` | How much of the source image is re-noised: `0` returns (approximately) the source, `1` ignores it |

The init image (and mask, if given) are decoded and resized to `width` × `height`.

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/img2img \
     -H "Content-Type: application/json" \
     -d '{
           "init_images": ["<base64 png>"],
           "prompt": "a watercolor painting of the same scene",
           "denoising_strength": 0.6,
           "steps": 20
         }'
```

## Progress polling

Generation runs as a *task* identified by `force_task_id` (default
`"default_task"`). While a generation request is in flight, poll:

`POST /v1/internal/progress` - request body:

```json
{ "id_task": "my_task" }
```

Response (`200`):

```json
{
  "completed": false,
  "progress": 0.42,
  "live_preview": "<base64 jpeg or empty>",
  "id_live_preview": 3
}
```

| Field | Meaning |
|---|---|
| `completed` | `true` once the task has finished (success or failure) |
| `progress` | Fraction completed, from `0` to `1` |
| `live_preview` | Base64 JPEG preview frame; empty until the first frame is emitted (only when the `live_previews_enable` option is on) |
| `id_live_preview` | Incrementing counter for preview frames |

Errors: `400` if `id_task` is missing, `404` with `{"message": "Task not found"}` if
the task ID is unknown.

The final image is delivered by the *original generation request's* HTTP response -
the progress endpoint reports status only. A typical polling loop:

```python
import requests
import time

BASE = "http://127.0.0.1:8188"
task_id = "my_task"

# Send the generation request (blocks until done) - in a thread
import threading
result = {}

def generate():
    result["response"] = requests.post(
        f"{BASE}/v1/sdapi/v1/txt2img",
        json={"prompt": "a red apple on a table", "steps": 20, "force_task_id": task_id},
    )

thread = threading.Thread(target=generate)
thread.start()

# Poll progress until completed
while True:
    r = requests.post(f"{BASE}/v1/internal/progress", json={"id_task": task_id})
    state = r.json()
    print(f"progress: {state['progress']:.0%}")
    if state["completed"]:
        break
    time.sleep(1)

thread.join()
print(result["response"].json()["images"][0][:80], "...")
```

## Interrupt

`POST /v1/sdapi/v1/interrupt` - no request body. Sets the generator's interrupt
flag; the running generation stops at the next sampling step and returns whatever
was produced so far (the task completes normally with the partial result).

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/interrupt
```

Response: `200 OK`.

Interrupts apply to the *current* generation - the server processes one inference at
a time, so the interrupt takes effect on whichever generation is running when the
request arrives.