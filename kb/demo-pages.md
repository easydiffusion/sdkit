# Demo Pages

The `demo/` directory holds self-contained browser pages that exercise the sdkit HTTP
API interactively. They are served by the server itself (same-origin, so the browser's
`fetch` calls don't need CORS handling), which makes them the quickest way to check a
feature end-to-end without a separate client. See [server-and-api](server-and-api.md)
for how the server serves them.

## What's in `demo/`

| File | What it demos | API routes used |
|---|---|---|
| `index.html` | A shell page that hosts all the other demos in an iframe, with a sidebar listing each one. Not a demo itself. | none (just embeds the others) |
| `segmentation-demo.html` | SAM2/SAM3 segmentation: upload an image, click to add positive/negative points, drag to draw boxes, or type a text prompt. Shows masks overlaid on the image with per-mask scores. | `/v1/sdapi/v1/models` (to populate the segmentation model list) and `/v1/sdapi/v1/segment` (see [segmentation](segmentation.md)) |
| `upscaling-demo.html` | RealESRGAN upscaling: upload an image, pick a model, upscale, and compare before/after. | `/v1/sdapi/v1/models` (to populate the upscaler list) and `/v1/sdapi/v1/extra-batch-images` (see [image-filters-and-upscaling](image-filters-and-upscaling.md)) |
| `filters-demo.html` | ControlNet preprocessing preview: upload an image and view the Canny edge map. | `/v1/controlnet/detect` (see [image-filters-and-upscaling](image-filters-and-upscaling.md)) |

Note that there is no text-to-image demo page in `demo/`. Generation is exercised
through the `generate_image.py` script at the repository root (see
[testing](testing.md)) rather than through a browser page.

## How the server serves them

The routes are defined in `Server::setupRoutes()` in `src/server.cpp`:

- `GET /` serves `demo/index.html` - the shell page.
- `GET /demo/<name>.html` serves any of the specific demo pages.

`handleDemoPage` looks for the file relative to the current working directory first,
then relative to the executable's directory (on Windows via `GetModuleFileNameA`),
and rejects anything that isn't a plain `.html` filename (no path separators, no
`..`). It responds with `404` and a hint if the file can't be found, which usually
means the server was started from the wrong directory - start it from the repository
root, or place the `demo/` folder next to the executable.

## How the pages call the API

Each demo page is a standalone HTML file with inline CSS and JavaScript (no build
step, no external dependencies). Every page defines the same small `api` helper
around `fetch`:

```js
function api(path, opts) {
  return fetch(path, opts).then(async (r) => {
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.message || r.statusText);
    return body;
  });
}
```

It unwraps the JSON body, and on a non-2xx response throws with the server's
`message` field (the same field the routes use for errors - see
[server-and-api](server-and-api.md)), which the page displays as an error banner.

Typical request flow, using segmentation as an example:

1. On load, the page calls `api("/v1/sdapi/v1/models")` and populates a `<select>`
   with the names under the `SEGMENTATION` key, so the user can pick the model the
   server has loaded.
2. The user drags an image onto the page (or picks a file), which is read as a data
   URL and drawn onto a canvas.
3. The user clicks on the canvas to place positive points, shift-clicks for negative
   points, or drags to draw a box; alternatively a text prompt is typed into a box.
4. Clicking "Run" POSTs to `/v1/sdapi/v1/segment` with the image (base64), the model
   name, and whichever prompts were supplied.
5. The response's `masks` array (base64 grayscale PNGs) is decoded onto per-mask
   canvases and composited over the original image with different hues per mask;
   `scores` is shown next to each mask, and individual masks can be toggled on and
   off.

The upscaling and filters demos follow the same pattern with their own routes: they
send base64 image data in the request and draw the base64 image(s) in the response
onto the canvas.

## Adding a new demo page

When a feature lands, a demo page is a cheap way to make it visible and testable.
The steps:

1. Create `demo/<feature>-demo.html`, self-contained (inline `<style>` and
   `<script>`, no external assets), following the layout of an existing page. Include
   the `api` helper shown above so errors surface as user-visible banners.
2. Call the new feature's route (or routes) from the page. If the feature needs a
   model list, populate it from `/v1/sdapi/v1/models` as the existing demos do.
3. Register the page in `demo/index.html`'s `DEMOS` array with an `id`, `title`,
   `icon`, `file` (as `demo/<feature>-demo.html`), and a one-line `desc`. The shell
   builds the sidebar navigation from this list; no other change is needed.
4. Serve the page from the server - no route change is required, because
   `handleDemoPage` accepts any `<name>.html` in the `demo/` directory.
5. Start the server from the repository root (`./build/sdkit --ckpt-dir ...`), open
   `http://127.0.0.1:8188/` in a browser, and check that the new page appears in the
   sidebar, loads without console errors, and exercises the new feature.

Keep the page self-contained: the demos are intentionally dependency-free so they can
be opened directly from the server without any build or bundling step, and they
double as executable documentation for the API's request/response shapes.

## Cross-links

- Route details and error shapes: [server-and-api](server-and-api.md).
- What the segmentation demo exercises: [segmentation](segmentation.md).
- What the upscaling and filters demos exercise:
  [image-filters-and-upscaling](image-filters-and-upscaling.md).
- How demos fit into manual testing: [testing](testing.md).
