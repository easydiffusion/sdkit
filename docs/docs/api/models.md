# Models & options

The server keeps a small set of *options* in memory and persists them to an
`options.json` file next to the executable. The most important option,
`sd_model_checkpoint`, selects which checkpoint generation uses.

Options follow the Automatic1111 webui naming.

## `GET /v1/sdapi/v1/options`

Returns the current options as a JSON object.

```bash
curl http://127.0.0.1:8188/v1/sdapi/v1/options
```

Response:

```json
{
  "sd_model_checkpoint": "",
  "live_previews_enable": true,
  "CLIP_stop_at_last_layers": 1,
  "sdxl_clip_l_skip": false,
  "samples_format": "png",
  "forge_additional_modules": []
}
```

On a fresh install (no `options.json` yet) the server seeds exactly these defaults:

| Option | Default | Meaning |
|---|---|---|
| `sd_model_checkpoint` | `""` | Checkpoint *lookup name* (path relative to `--ckpt-dir`, with extension). If empty, generation fails with "No model selected". |
| `live_previews_enable` | `true` | Emit live preview frames while generating |
| `CLIP_stop_at_last_layers` | `1` | CLIP skip (number of layers to skip at the end) |
| `sdxl_clip_l_skip` | `false` | Skip CLIP-L for SDXL |
| `samples_format` | `"png"` | Output image format |
| `forge_additional_modules` | `[]` | List of standalone VAE / text-encoder files; each is inspected by content and routed to the right slot (VAE, CLIP-L, CLIP-G, T5-XXL, LLM) |

## `POST /v1/sdapi/v1/options`

Sets one or more options. The body is a JSON object whose keys are option names; it
is merged key-by-key into the current options, so you only need to send the fields
you want to change.

**Selecting a checkpoint:**

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/options \
     -H "Content-Type: application/json" \
     -d '{"sd_model_checkpoint": "sd3.5_large_turbo.gguf"}'
```

Response: `200 OK` on success. The merged options are written to `options.json`
immediately, so they survive a server restart. If saving fails, the server returns
`500` with `{"message": "Failed to save options"}`; a malformed JSON body returns
`400` with `{"message": "Invalid JSON"}`.

!!! note "Checkpoint names"

    `sd_model_checkpoint` is the path relative to the `--ckpt-dir` you started the
    server with, including the file extension - e.g. `mymodel.safetensors` or
    `subdir/mymodel.gguf`. If the name does not resolve to a file, generation
    requests fail with a "Model not found" error (see
    [troubleshooting](../appendix.md)).

**Enabling/disabling live previews:**

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/options \
     -H "Content-Type: application/json" \
     -d '{"live_previews_enable": false}'
```

## `POST /v1/sdapi/v1/refresh-checkpoints`

Rescans the checkpoint directory (`--ckpt-dir`). Use it after adding or removing
model files on disk so they show up without restarting the server.

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/refresh-checkpoints
```

Response: `200 OK`.

## `POST /v1/sdapi/v1/refresh-vae-and-text-encoders`

Rescans the VAE directory (`--vae-dir`) and the text-encoder directory
(`--text-encoder-dir`).

```bash
curl -X POST http://127.0.0.1:8188/v1/sdapi/v1/refresh-vae-and-text-encoders
```

Response: `200 OK`.

Both refresh endpoints are safe to call while a generation is in flight - the model
index is mutex-guarded, so rescanning never races with a request that is resolving a
model path.