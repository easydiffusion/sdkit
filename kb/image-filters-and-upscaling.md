# Image Filters and Upscaling

This page documents the `ImageFilters` component (`src/image_filters.cpp`,
`include/image_filters.h`), which provides RealESRGAN upscaling and ControlNet
preprocessing, plus the header-only Canny implementation in
`src/filters/controlnet.hpp`. The routes that call into it are described in
[server-and-api](server-and-api.md); how generation uses the control-image
preprocessing is covered in [image-generation](image-generation.md).

## ImageFilters

`ImageFilters` is created once per server with a shared `ModelManager`. It owns two
things:

- `upscaler_ctx_`: a stable-diffusion.cpp upscaler context, lazily loaded and freed
  in the destructor.
- A reference to the `ModelManager`, used only to find the RealESRGAN models
  directory.

Images flow through the class as base64 strings in and base64 strings out; the
internal representation is the sd.cpp `sd_image_t` (width/height/channel plus a
`malloc`'d byte buffer). Helpers `base64ToImage` / `imageToBase64` (from
`image_utils.h`) handle the conversion, and `freeImage` releases the buffers. On
failure for an individual image in a batch, an empty string is pushed for that
position rather than failing the whole batch.

## Upscaling

### Model selection and lazy loading

`ensureUpscalerLoaded(upscaler_name)` loads the upscaler on first use:

1. An empty upscaler name is rejected.
2. If the same name is already loaded, nothing happens (the context is reused).
3. Otherwise the old context is freed, and a new one is built:
   - The RealESRGAN directory comes from `ModelManager::getRealesrganModelsPath()`
     (set from `--realesrgan-models-path` at startup; see
     [models-and-detection](models-and-detection.md)). An empty directory is a hard error.
   - The model path is `<realesrgan_dir>/<upscaler_name>.pth` - note the hard-coded
     `.pth` extension and the flat directory; subdirectories are not searched here
     (unlike checkpoint scanning, which recurses).
   - `new_upscaler_ctx(path, false, -1, 128, nullptr, nullptr)` creates the context
     (offload disabled, tile size 128). A failed load is logged and returns false.

The upscaler name is what the client passes in `/v1/sdapi/v1/extra-batch-images` as
`upscaler_1`; the server translates webui-style names ("R-ESRGAN 4x+") into
sdkit-style names ("RealESRGAN_x4plus") before calling
(see [server-and-api](server-and-api.md)). The name used here must match a `.pth`
file in the RealESRGAN directory.

### The upscale call

`upscaleImage(input_image, upscale_factor)` calls sd.cpp's `upscale` function with
the factor and returns the first result image. `upscaleBatch` wraps this for a list
of images, converting each to and from base64 around the call.

## ControlNet preprocessing

ControlNet preprocessing turns a regular input image into the conditioning map the
ControlNet adapter expects (for example, edge maps for Canny). It is exposed two ways:

- `applyControlNetFilterBatch(base64_images, module)` - used by the
  `/v1/controlnet/detect` route.
- `applyControlNetFilter(input_image, module)` - the single-image version, which the
  generator calls on its own control images during img2img generation
  (see [image-generation](image-generation.md)).

Both follow the same flow: decode the base64 image into an `sd_image_t`, make a
copy of its pixel buffer (the preprocessing modifies pixels in place), apply the
requested module, and encode the result back to base64.

Only the `"canny"` module has real preprocessing today. Any other module string
falls back to the (unsupported) generic path, which logs a warning and returns the
image unchanged. This means the route can be called with an arbitrary module name,
but only Canny actually changes pixels - the rest is a placeholder for future
filters.

## The Canny implementation

`src/filters/controlnet.hpp` is a header-only, dependency-free Canny edge detector
in the `controlnet` namespace, written so that its output matches
stable-diffusion.cpp's own `preprocess_canny()` (the code comments call this out
explicitly - the non-maximum-suppression and hysteresis logic is kept aligned with
that implementation on purpose).

The entry points:

```cpp
bool preprocess_canny(sd_image_t image, const CannyOptions &options = {});
bool preprocess_canny(sd_image_t image, float high_threshold, float low_threshold,
                      float weak, float strong, bool inverse);
bool preprocess(sd_image_t image, std::string_view module);
```

`CannyOptions` carries the knobs: `high_threshold` (0.08), `low_threshold` (0.08),
`weak` (0.8), `strong` (1.0), and `inverse` (false). The default-argument overload is
what the server uses (`applyControlNetFilter` calls
`preprocess_canny(result_image, 0.08f, 0.08f, 0.8f, 1.0f, false)` - the same values
as the defaults).

### Algorithm stages

The image is converted to a per-pixel float representation in `[0, 1]`
(`sdImageToFloatImage`), then:

1. **Grayscale** - luminance weights `0.2989 R + 0.5870 G + 0.1140 B` are applied,
   collapsing the image to one channel.
2. **Gaussian blur** - a 5x5 kernel with sigma 1.4 is convolved over the grayscale
   image to reduce noise before edge detection.
3. **Sobel gradient** - 3x3 Sobel kernels in X and Y are convolved to get the two
   gradient components.
4. **Magnitude and direction** - `tensorHypot` combines the X and Y gradients into a
   magnitude; `tensorArctan2` gives the angle. The magnitude is normalized so its
   maximum is 1.0.
5. **Non-maximum suppression** - pixels are compared against their two neighbours in
   the gradient direction (using four 45-degree angle buckets) and zeroed unless they
   are local maxima. This thins edges to single-pixel lines.
6. **Hysteresis thresholding** - a pixel is marked `strong` if its value is at least
   `max_value * high_threshold`, `weak` if it is above `high * low_threshold`, and 0
   otherwise. Weak pixels are promoted to strong if any of their 6 checked
   neighbours (the right, up, and down/left directions used by the algorithm) are
   strong - this links faint edges to strong ones and discards isolated noise. A
   3-pixel border is zeroed first so the neighbour loop can be simple.
7. **Write-back** - the single-channel grayscale result is broadcast into all
   channels of the original `sd_image_t` buffer (a grayscale result in an RGB image),
   and `inverse` optionally flips it (useful for producing a white-on-black map).

The helpers (`FloatImage`, `convolve`, `gaussianKernel`, `grayscale`,
`tensorHypot`, `tensorArctan2`, `normalize`, `nonMaxSuppression`,
`thresholdHysteresis`) live in a `detail` namespace and are only used by these
functions; they operate entirely on the CPU with no third-party dependencies beyond
`stable-diffusion.h` for the `sd_image_t` type.

## Threading

`ImageFilters` has no internal mutex. The upscaler context (`upscaler_ctx_`) and the
ControlNet pixel buffers are not thread-safe, so concurrent upscaling or
preprocessing calls into the same `ImageFilters` instance must be serialized by the
caller. In practice the server does this implicitly: the route handler and the
generator each run their filtering inside the generator's own lock (see
[architecture](architecture.md) for the locking story), so two requests cannot hit
the same upscaler context at once. If you add new call sites, keep that invariant in
mind.
