# Segmentation

This page documents the SAM3-based segmentation feature: the `ImageSegmenter`
component (`src/image_segmenter.cpp`, `include/image_segmenter.h`), the third-party
`sam3.cpp` engine it wraps, and the API route that drives it
(`/v1/sdapi/v1/segment` - see [server-and-api](server-and-api.md)).

## The SAM3 engine

sdkit uses [sam3.cpp](https://github.com/cmdr2/sam3.cpp) as the segmentation
backbone. It is vendored as a git submodule under `third_party/sam3.cpp` (along with
the shared `third_party/ggml` it builds against - see
[architecture](architecture.md)). It is built as a **static** library (`sam3`) that
the sdkit executable links, so it does not ship as a separate runtime file.

sam3.cpp supports two families of models:

- **Full SAM3 models** - include a text-encoder head, so they can be driven by a text
  prompt directly.
- **Visual-only models** - SAM2, SAM2.1, EdgeTAM, and the SAM3 visual variant. These
  can only be prompted with points or boxes; a text prompt is rejected.

The component detects which family a loaded model belongs to via
`sam3_is_visual_only(*model_)` and routes the request accordingly.

## SegmentationParams and SegmentationResult

`include/image_segmenter.h` declares the public interface:

```cpp
struct SegmentationParams {
    std::string image_base64;   // input image (data URI prefix allowed)
    std::string model_name;     // lookup name of the segmentation model

    std::string text_prompt;    // full SAM3 models only
    std::vector<std::pair<float, float>> positive_points;  // (x, y)
    std::vector<std::pair<float, float>> negative_points;  // (x, y)
    std::vector<std::vector<float>> boxes;                 // [x0, y0, x1, y1] each

    float score_threshold = 0.5f;
};

struct SegmentationResult {
    std::vector<std::string> masks;   // base64 grayscale PNGs
    std::vector<float> scores;        // detection score per mask
};
```

The server's `handleSegment` builds the params from the request JSON and hands them
to `segment()`. `masks` and `scores` are parallel arrays (index `i` of `masks`
corresponds to index `i` of `scores`). Empty masks are skipped during encoding, so a
returned mask always has at least one non-empty pixel.

## ImageSegmenter lifecycle

`ImageSegmenter` is created once per server with the shared `ModelManager`. It owns
its own mutex: sam3.cpp has no thread-safety guarantees, so **every** call into this
class (including the destructor) must take `mutex_`. The destructor frees the state
and model under the lock.

### Lazy model loading

`ensureModelLoaded(model_name)` loads a segmentation model on first use (or when the
requested model differs from the currently loaded one):

1. If the same model name is already loaded, return immediately.
2. An empty model name throws `No segmentation model specified`.
3. If `ModelManager::getSegmentationDir()` is empty, throw `Segmentation is not
   available...` - the message tells the operator to restart the server with
   `--segmentation-models-path` (see [building-and-platforms](building-and-platforms.md)
   for how this flag is set at startup).
4. `ModelManager::getModelByName(name, ModelType::SEGMENTATION)` resolves the lookup
   name to an absolute path (see [models-and-detection](models-and-detection.md) for the
   naming convention for `SEGMENTATION` models - filename with extension).
   An unknown name throws `Segmentation model not found: ...`.
5. The old state and model are released first, so a model switch is clean.
6. `sam3_params` is populated: `model_path`, `use_gpu = true` (falls back to CPU when
   no GPU backend is available), and `n_threads` from
   `std::thread::hardware_concurrency()` (defaulting to 4 if it reports 0).
7. `sam3_load_model` creates the model and `sam3_create_state` creates the per-image
   inference state. Either failure throws with a descriptive message.

`loaded_model_name_` records what's currently loaded so subsequent calls with the
same model reuse the context.

## The segment() flow

`segment(params)` locks the mutex, then:

1. **Ensure model** - `ensureModelLoaded(params.model_name)`.
2. **Decode the image** - `base64ToImage(params.image_base64, 3)` requests exactly 3
   channels (RGB). Decode failure throws.
3. **RAII guard** - a local `ImageGuard` struct frees the decoded image on scope exit
   so no path leaks it, including exceptions.
4. **Copy into sam3 format** - the pixel bytes are copied into a `sam3_image` struct
   (width, height, `channels = 3`, and a `std::vector<unsigned char>` holding the
   RGB data).
5. **Image encoding** - `sam3_encode_image(*state_, *model_, image)` pushes the
   image through the visual encoder; failure throws `Failed to encode image for
   segmentation`.
6. **Prompt validation** - at least one of `text_prompt`, `positive_points` /
   `negative_points`, or `boxes` must be present; otherwise throw `No prompt
   provided...`.
7. **Run the requested prompts**:
   - **Text prompt** (`sam3_segment_pcs`): only allowed on full SAM3 models. A visual-
     only model throws an explanatory error listing the visual-only families
     (SAM2 / SAM2.1 / EdgeTAM / SAM3-visual). A `sam3_pcs_params` struct is filled
     with `text_prompt` and `score_threshold`.
   - **Points** (`sam3_segment_pvs`): `positive_points` and `negative_points` are
     converted into `pvs.pos_points` / `pvs.neg_points`; `multimask = false`.
   - **Boxes** (`sam3_segment_pvs`): each box in `params.boxes` must have exactly 4
     values (`x0, y0, x1, y1`) or a descriptive error is thrown; one `sam3_pvs_params`
     is created per box with `use_box = true` and `multimask = false`.
8. **Result collection** - each `sam3_result` from the above calls is passed to
   `appendPvsResult`, which for every detection:
   - skips masks with no data or non-positive dimensions
   - wraps the mask bytes (1 channel) in an `sd_image_t` and encodes it as base64 PNG
     via `imageToBase64`
   - pushes the mask string and the detection score into the result vectors.
   A failed PNG encode logs a warning and skips that mask.
9. Return the `SegmentationResult`. The server handler turns it into
   `{"masks": [...], "scores": [...]}` (see [server-and-api](server-and-api.md)).

A text prompt and point/box prompts can be combined in one request; each gets its own
segmentation call and the results are merged into the same output arrays.

## Threading and GPU

sam3.cpp runs its inference on whatever backend ggml has available (CPU, CUDA,
Vulkan, Metal), so segmentation benefits from the same GPU support as diffusion
(see [building-and-platforms](building-and-platforms.md)). The mutex mentioned above
is the only concurrency control: if two HTTP requests ask for segmentation at the
same time, one will block the other until it finishes.

## Error handling

All of `segment()`'s failure paths throw `std::runtime_error` with a human-readable
message. The server handler catches this and converts it into a `500` response whose
`message` field is the exception's text, so the client sees the reason (e.g. "No
prompt provided..." or "Segmentation model not found: <name>") directly.
