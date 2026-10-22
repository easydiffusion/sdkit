# Filters, upscaling & ControlNet

Reference for the image-processing endpoints: [upscaling](#upscaling-images) and
[ControlNet preprocessing](#controlnet-conditioning-images), plus how to pass
[ControlNet inside generation requests](#controlnet-inside-generation-requests).

All endpoints take base64 images and return base64 images (see
[API conventions](index.md#images)).

## Upscaling images

`POST /v1/sdapi/v1/extra-batch-images` - upscales (or simply passes through) a list
of images using a RealESRGAN upscaler.

### Request

| Field | Type | Default | Description |
|---|---|---|---|
| `imageList` | object[] | *required* | Images to process; each entry is `{"data": "<base64>"}` |
| `upscaler_1` | string | `""` | Upscaler name (see mapping below) |
| `upscaling_resize` | int | `0` | Upscaling factor. If `0`, no upscaling happens and the images are returned unchanged |

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/extra-batch-images \
     -H "Content-Type: application/json" \
     -d '{
           "imageList": [{"data": "<base64 png>"}],
           "upscaler_1": "R-ESRGAN 4x+",
           "upscaling_resize": 4
         }'
```

### Response

```json
{ "images": ["<base64 png>", "..."] }
```

Errors:

- `400` `{"message": "Missing imageList parameter"}` - no `imageList` in the body.
- `500` `{"message": "Upscaler not available. Please configure an upscaler model in
  options."}` - upscaling was requested but no upscaler model is loaded. Make sure
  the server was started with `--realesrgan-models-path` pointing at a directory
  containing the model file (e.g. `RealESRGAN_x4plus.pth`), and see the
  [troubleshooting page](../appendix.md).

### Upscaler names

A1111/Forge-style names are translated to the RealESRGAN model file names:

| `upscaler_1` value | Model file expected in `--realesrgan-models-path` |
|---|---|
| `R-ESRGAN 4x+` | `RealESRGAN_x4plus.pth` |
| `R-ESRGAN 4x+ Anime6B` | `RealESRGAN_x4plus_anime_6B.pth` |
| *(any other name)* | passed through unchanged - must match a `.pth` file in the upscaler directory |

The upscaler context is loaded lazily on first use and reused while the same upscaler
name is requested.

## ControlNet conditioning images

`POST /v1/controlnet/detect` - preprocesses input images into ControlNet
conditioning images (e.g. edge maps), which you can then feed as the ControlNet
`image` of a generation request.

### Request

| Field | Type | Default | Description |
|---|---|---|---|
| `controlnet_input_images` | string[] | - | Base64 images to preprocess |
| `controlnet_module` | string | `"canny"` | Preprocessing module. **Only `canny` is implemented**; other module names are logged as unsupported and the images are returned unchanged |

```bash
curl -X POST http://127.0.0.1:8188/v1/controlnet/detect \
     -H "Content-Type: application/json" \
     -d '{
           "controlnet_module": "canny",
           "controlnet_input_images": ["<base64 png>"]
         }'
```

### Response

```json
{ "images": ["<base64 canny edge map>"] }
```

The Canny implementation uses fixed thresholds (high/low `0.08`, weak `0.8`,
strong `1.0`), matching stable-diffusion.cpp's own preprocessor. If an individual
image in the batch fails to process, an empty string is returned for that position
rather than failing the whole batch.

## ControlNet inside generation requests

Generation requests can carry a ControlNet via the webui-style
`alwayson_scripts` field:

| Field | Type | Default | Description |
|---|---|---|---|
| `alwayson_scripts.controlnet.args[0].image` | string | - | Base64 control image (e.g. a Canny edge map from `/v1/controlnet/detect`) |
| `alwayson_scripts.controlnet.args[0].model` | string | - | ControlNet model name. A webui-style hash suffix (`model [a3cd7cd6]`) is stripped automatically |
| `alwayson_scripts.controlnet.args[0].weight` | float | `1.0` | Control strength |

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/txt2img \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "a city skyline",
           "width": 512, "height": 512,
           "alwayson_scripts": {
             "controlnet": {
               "args": [{
                 "image": "<base64 canny edge map>",
                 "model": "control_v11f1p_sd15_canny",
                 "weight": 1.0
               }]
             }
           }
         }'
```

Behavior notes:

- If a ControlNet `model` is present, the generation dimensions are **rounded to the
  nearest multiple of 64** - a 500×512 request becomes 512×512.
- The model name is resolved against `--controlnet-dir`. An unknown ControlNet model
  logs a warning and the generation proceeds *without* ControlNet.
- The control image is decoded, resized to the generation size, and preprocessed
  with the Canny filter. If preprocessing fails, the unfiltered image is used and a
  warning is logged.

!!! tip

    Chain `/v1/controlnet/detect` into a generation request: run the detect endpoint
    on your input photo, then pass the returned edge map as the ControlNet `image`
    with a matching ControlNet model from `--controlnet-dir`.