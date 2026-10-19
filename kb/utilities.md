# Utilities

This page documents the small, foundational components that the larger features
(generation, filters, segmentation, server) are built on: `logging.h`/`logging.cpp`,
`string_utils.h`/`string_utils.cpp`, `image_utils.h`/`image_utils.cpp`, and
`task_state.h`/`task_state.cpp`. Each is small enough to be understood in one pass,
and together they cover most of the plumbing that isn't part of a specific feature.

## Logging (`include/logging.h`, `src/logging.cpp`)

A simple, thread-safe, levelled logger used by every component.

- **Levels**: `Verbose`, `Debug`, `Info`, `Warning`, `Error`. Set once at startup by
  `main.cpp` from the `--log-level` flag; defaults to `Info`. `set_log_level` also
  accepts a string (`"verbose"`, `"debug"`, `"info"`, `"warning"`, `"error"`),
  falling back to `Info` with a warning for anything else.
- **Macros**: `LOG_VERBOSE`, `LOG_DEBUG`, `LOG_INFO`, `LOG_WARNING`, `LOG_ERROR` are
  printf-style wrappers around `log_message(level, format, ...)`.
- **Output**: `log_message` formats the message into a 4 KB stack buffer (truncating
  longer messages), then writes a single line to `stderr` for Warning/Error and
  `stdout` for the rest, both flushed immediately. Lines are prefixed with a
  millisecond timestamp and an ANSI colour code (magenta for Verbose, cyan for
  Debug, green for Info, yellow for Warning, red for Error) followed by a reset.
- **Thread safety**: a `static std::mutex log_mutex` guards the log level and the
  write, so concurrent HTTP handlers and background threads interleave cleanly.
- **stable-diffusion.cpp bridge**: `sd_log_cb` is the callback registered with the
  inference engine. It maps `SD_LOG_DEBUG` / `SD_LOG_INFO` / `SD_LOG_WARN` /
  `SD_LOG_ERROR` to the matching `LogLevel`, strips a trailing newline, and forwards
  to `log_message`. This is how sd.cpp's own messages appear in sdkit's log stream
  (see [image-generation](image-generation.md) for where the callback is installed).

The server also routes crow's log output through this system via a `FilteredLogHandler`
(see [server-and-api](server-and-api.md)), which drops the noisy `/ping` and
`/internal/progress` request logs before passing everything else to `log_message`.

## String utils (`include/string_utils.h`, `src/string_utils.cpp`)

Three functions for handling paths and Unicode across platforms:

- `utf8_to_wstring(const std::string&)` -> `std::wstring`
- `wstring_to_utf8(const std::wstring&)` -> `std::string`
- `path_to_utf8(const std::filesystem::path&)` -> `std::string`

On Windows the conversions go through the Win32 API
(`MultiByteToWideChar` / `WideCharToMultiByte` with `CP_UTF8`), because
`std::filesystem` paths are natively UTF-16 there and `path.string()` may return a
locale-dependent encoding. On POSIX systems the conversion uses
`std::codecvt_utf8<wchar_t>`, and `path_to_utf8` simply calls `path.string()`
(POSIX paths are already UTF-8). Model-manager scanning and the upscaler loader both
use `path_to_utf8` so that model files with non-ASCII names work on Windows (see
[models-and-detection](models-and-detection.md) and
[image-filters-and-upscaling](image-filters-and-upscaling.md)).

## Image utils (`include/image_utils.h`, `src/image_utils.cpp`)

The bridge between base64 strings over HTTP and the `sd_image_t` buffers that
stable-diffusion.cpp and the filters operate on. Uses stb_image, stb_image_write,
stb_image_resize2, and the `base64.hpp` helper.

- `imageToBase64(const sd_image_t&)` -> `std::string` - encodes the image as a PNG
  in memory (`stbi_write_png_to_func` writing into a `std::vector`) then base64-encodes
  the bytes. Returns an empty string on failure. A tightly packed row stride of
  `width * channel` is assumed.
- `base64ToImage(const std::string&, int desired_channels = 3)` -> `sd_image_t` -
  strips a data URI prefix (`data:image/png;base64,`) if present, decodes the base64,
  and loads the PNG/JPEG bytes via `stbi_load_from_memory` into an `sd_image_t`. The
  `desired_channels` argument forces the loaded image to that many channels (1 for
  masks, 3 for RGB input); passing 0 keeps the file's native channel count. Returns
  an empty image on decode or load failure (logged, not thrown - callers check the
  `data` pointer).
- `freeImage(sd_image_t&)` - releases the pixel buffer with `stbi_image_free` and
  zeroes the struct.
- `resizeImage(sd_image_t&, int target_width, int target_height, bool clamp_to_8)` -
  resizes in place using `stbir_resize_uint8_linear`. With `clamp_to_8 = true` the
  target dimensions are rounded *down* to multiples of 8 (the latent-space
  requirement for diffusion models - see [image-generation](image-generation.md));
  if the result would be 0 the value is forced to 8. The old buffer is freed and
  replaced by the resized one. Returns `true` if the image is already the right size
  or the resize succeeded; `false` on invalid input, unsupported channel count, or a
  failed resize.

Note that `ImageGenerator` and the filters each have their own local `freeImage`
helper in the generator/filter translation units; they use `free`/`stbi_image_free`
on the same buffers, so be careful not to double-free across translation units - a
buffer produced by `base64ToImage` belongs to the caller and should be freed once.

## Task state (`include/task_state.h`, `src/task_state.cpp`)

The per-request generation progress and interruption tracking, exposed to clients via
`/v1/internal/progress` and `/v1/sdapi/v1/interrupt`
(see [server-and-api](server-and-api.md)).

- **`TaskState`** - one entry per task ID: `task_id`, `completed`, `progress`
  (0.0-1.0), `live_preview` (base64 JPEG of the in-progress frame),
  `id_live_preview` (monotonic counter, bumped each time a new preview arrives),
  `result_images` (base64 PNGs), `info` (the JSON string describing the job), and
  `interrupted`.
- **`TaskStateManager`** - holds a `std::map<std::string, TaskState>` guarded by a
  mutex. Every public method takes the lock, so the progress polling endpoint, the
  generation handler, and the interrupt endpoint can all touch the map concurrently.
- **Methods**:
  - `createTask(task_id)` - registers a new task with zeroed progress and no preview.
  - `updateTaskProgress(task_id, progress, live_preview = "")` - called from the
    sd.cpp progress and preview callbacks (see [image-generation](image-generation.md)).
    A non-empty preview also bumps `id_live_preview`, which clients use to detect a
    new frame.
  - `completeTask(task_id, images, info)` - marks the task completed, sets progress
    to 1.0, and stores the final images and info text.
  - `interruptTask(task_id)` - sets `interrupted = true`.
  - `getTaskState(task_id)` - returns the state, or a default-constructed one with
    `progress = 0.0` and `completed = false` if the task is unknown (so the endpoint
    can tell a client "nothing yet" without crashing).
  - `taskExists(task_id)` - boolean check.
  - `clearTask(task_id)` - erases the entry. Useful when the client is done with a
    task and wants to drop its preview/result memory.

The map is not pruned automatically - a task stays in memory until `clearTask` is
called for it, so a long-running server with many finished tasks will grow over time
unless the client cleans up.
