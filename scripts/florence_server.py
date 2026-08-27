#!/usr/bin/env python3
"""Standalone Florence-2 HTTP API server with permissive CORS headers.

Run it in a venv with everything installed.

Endpoints:
    GET  /                -> API summary
    GET  /health          -> liveness + loaded model info
    GET  /v1/models       -> allowed Florence models for this server
    GET  /v1/tasks        -> supported task aliases / task tokens
    POST /v1/run          -> run a Florence task

Request body for POST /v1/run:
    {
      "model": "florence-community/Florence-2-large-ft",
      "task": "caption" | "detailed" | "more-detailed" | "od" |
              "dense" | "ground" | "ocr" | "ocr-region" | raw Florence token,
      "image": "data:image/png;base64,..." | raw base64 | local path | http(s) URL,
      "text_input": "optional extra text for tasks like ground / ovd",
      "max_new_tokens": 256,
      "num_beams": 3
    }
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import sys
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

_ = os.environ.setdefault("HF_HOME", "C:/models/huggingface")
_ = os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_ = os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor
except Exception as exc:  # pragma: no cover - import-time environment issue
    raise SystemExit(
        f"Missing Florence server dependencies. Run this file with sahara/.venv/Scripts/python.exe so it can "
        f"import torch, Pillow, and transformers.\nImport error: {exc}"
    ) from exc


def import_florence_model_loader() -> tuple[Any, str]:
    try:
        from transformers import AutoModelForImageTextToText as loader

        return loader, "AutoModelForImageTextToText"
    except ImportError:
        try:
            from transformers import AutoModelForMultimodalLM as loader

            return loader, "AutoModelForMultimodalLM"
        except ImportError:
            from transformers import Florence2ForConditionalGeneration as loader

            return loader, "Florence2ForConditionalGeneration"


FlorenceModelLoader, MODEL_LOADER_NAME = import_florence_model_loader()

DEFAULT_MODELS: list[str] = [
    "florence-community/Florence-2-base",
    "florence-community/Florence-2-base-ft",
    "florence-community/Florence-2-large",
    "florence-community/Florence-2-large-ft",
]
DEFAULT_MODEL = "florence-community/Florence-2-large-ft"

OFFICIAL_TO_COMMUNITY: dict[str, str] = {
    "microsoft/Florence-2-base": "florence-community/Florence-2-base",
    "microsoft/Florence-2-base-ft": "florence-community/Florence-2-base-ft",
    "microsoft/Florence-2-large": "florence-community/Florence-2-large",
    "microsoft/Florence-2-large-ft": "florence-community/Florence-2-large-ft",
}

TASK_SPECS: list[dict[str, Any]] = [
    {
        "id": "caption",
        "token": "<CAPTION>",
        "label": "Caption",
        "needs_text": False,
        "description": "Short general caption.",
    },
    {
        "id": "detailed",
        "token": "<DETAILED_CAPTION>",
        "label": "Detailed caption",
        "needs_text": False,
        "description": "Longer descriptive caption.",
    },
    {
        "id": "more-detailed",
        "token": "<MORE_DETAILED_CAPTION>",
        "label": "More detailed caption",
        "needs_text": False,
        "description": "Most detailed captioning prompt.",
    },
    {
        "id": "od",
        "token": "<OD>",
        "label": "Object detection",
        "needs_text": False,
        "description": "Open-set object detection boxes returned by Florence.",
    },
    {
        "id": "dense",
        "token": "<DENSE_REGION_CAPTION>",
        "label": "Dense region captioning",
        "needs_text": False,
        "description": "Caption many detected regions.",
    },
    {
        "id": "ground",
        "token": "<CAPTION_TO_PHRASE_GROUNDING>",
        "label": "Phrase grounding",
        "needs_text": True,
        "description": "Locate a text phrase in the image.",
    },
    {
        "id": "ocr",
        "token": "<OCR>",
        "label": "OCR",
        "needs_text": False,
        "description": "Recognize text from the image.",
    },
    {
        "id": "ocr-region",
        "token": "<OCR_WITH_REGION>",
        "label": "OCR with regions",
        "needs_text": False,
        "description": "Recognize text and return region geometry.",
    },
    {
        "id": "ovd",
        "token": "<OPEN_VOCABULARY_DETECTION>",
        "label": "Open vocabulary detection",
        "needs_text": True,
        "description": "Detect user-provided category names.",
    },
    {
        "id": "region-desc",
        "token": "<REGION_TO_DESCRIPTION>",
        "label": "Region to description",
        "needs_text": True,
        "description": "Describe a region specified by Florence <loc_*> tokens.",
    },
    {
        "id": "ref-seg",
        "token": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "label": "Referring expression segmentation",
        "needs_text": True,
        "description": "Return segmentation text output for a phrase prompt.",
    },
]

TASK_ALIASES: dict[str, str] = {str(spec["id"]): str(spec["token"]) for spec in TASK_SPECS}
TASK_REQUIRING_TEXT: set[str] = {str(spec["token"]) for spec in TASK_SPECS if bool(spec["needs_text"])}
TASK_METADATA_BY_TOKEN: dict[str, dict[str, Any]] = {str(spec["token"]): spec for spec in TASK_SPECS}
TASK_METADATA_BY_ID: dict[str, dict[str, Any]] = {str(spec["id"]): spec for spec in TASK_SPECS}


class FlorenceError(RuntimeError):
    pass


class FlorenceRuntime:
    def __init__(self, *, device_request: str, allowed_models: list[str], default_model: str | None = None) -> None:
        self.device_request = device_request
        self.device = resolve_device(device_request)
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.allowed_models = normalize_allowed_models(allowed_models)
        self.default_model = default_model or (self.allowed_models[0] if self.allowed_models else DEFAULT_MODEL)
        self.lock = RLock()
        self.processor: Any | None = None
        self.model: Any | None = None
        self.loaded_requested_model_id: str | None = None
        self.loaded_resolved_model_id: str | None = None
        self.last_load_seconds: float | None = None
        self.total_requests = 0
        self.started_at = time.time()

        if self.default_model not in self.allowed_models:
            self.allowed_models.insert(0, self.default_model)

    def status_payload(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "model_loader": MODEL_LOADER_NAME,
            "allowed_models": self.allowed_models,
            "default_model": self.default_model,
            "loaded_model": {
                "requested_id": self.loaded_requested_model_id,
                "resolved_id": self.loaded_resolved_model_id,
                "load_seconds": self.last_load_seconds,
            },
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "requests_served": self.total_requests,
        }

    def get_models_payload(self) -> dict[str, Any]:
        models = []
        for model_id in self.allowed_models:
            alias_ids = [alias for alias, target in OFFICIAL_TO_COMMUNITY.items() if target == model_id]
            models.append(
                {
                    "id": model_id,
                    "aliases": alias_ids,
                    "recommended": model_id.endswith("large-ft"),
                    "source": "huggingface",
                }
            )
        return {
            "models": models,
            "default_model": self.default_model,
            "aliases": OFFICIAL_TO_COMMUNITY,
            **self.status_payload(),
        }

    def get_tasks_payload(self) -> dict[str, Any]:
        return {
            "tasks": TASK_SPECS,
            "task_aliases": TASK_ALIASES,
        }

    def ensure_model_loaded(self, requested_model_id: str) -> tuple[str, str, float]:
        resolved_model_id = resolve_model_id(requested_model_id)
        self._validate_model_allowed(requested_model_id, resolved_model_id)

        if self.loaded_resolved_model_id == resolved_model_id and self.model is not None and self.processor is not None:
            return requested_model_id, resolved_model_id, 0.0

        self._unload_model()

        start = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(resolved_model_id)
        self.model = load_model(resolved_model_id, dtype=self.dtype).to(self.device).eval()
        sync_cuda_if_needed(self.device)
        load_seconds = time.perf_counter() - start

        self.loaded_requested_model_id = requested_model_id
        self.loaded_resolved_model_id = resolved_model_id
        self.last_load_seconds = load_seconds
        return requested_model_id, resolved_model_id, load_seconds

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_model_id = str(payload.get("model") or self.default_model).strip()
        if not requested_model_id:
            raise FlorenceError("Request is missing a non-empty 'model'.")

        requested_task = str(payload.get("task") or "caption").strip()
        if not requested_task:
            raise FlorenceError("Request is missing a non-empty 'task'.")
        task_token = normalize_task(requested_task)

        text_input_raw = payload.get("text_input")
        text_input = str(text_input_raw).strip() if text_input_raw is not None else None
        if task_token in TASK_REQUIRING_TEXT and not text_input:
            spec = TASK_METADATA_BY_TOKEN.get(task_token)
            label = spec["label"] if spec else requested_task
            raise FlorenceError(f"Task '{label}' requires a non-empty 'text_input'.")

        image_source = payload.get("image")
        if image_source is None:
            image_source = payload.get("image_b64")
        if image_source is None:
            raise FlorenceError("Request must include an 'image' field (data URL, base64, path, or URL).")

        image = load_image(image_source)
        max_new_tokens = int(payload.get("max_new_tokens", 256))
        num_beams = int(payload.get("num_beams", 3))
        if max_new_tokens <= 0:
            raise FlorenceError("'max_new_tokens' must be > 0.")
        if num_beams <= 0:
            raise FlorenceError("'num_beams' must be > 0.")

        with self.lock:
            requested_id, resolved_id, load_seconds = self.ensure_model_loaded(requested_model_id)
            processor = self.processor
            model = self.model
            if processor is None or model is None:
                raise FlorenceError("Model failed to load.")

            prompt = build_prompt(task_token, text_input)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = move_batch_to_device(inputs, device=self.device, dtype=self.dtype)

            sync_cuda_if_needed(self.device)
            infer_start = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                )
            sync_cuda_if_needed(self.device)
            inference_seconds = time.perf_counter() - infer_start
            self.total_requests += 1

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                text=generated_text,
                task=task_token,
                image_size=(image.width, image.height),
            )

        parsed_json = to_jsonable(parsed)
        normalized = normalize_result(parsed_json, task_token)
        return {
            "ok": True,
            "model": {
                "requested_id": requested_id,
                "resolved_id": resolved_id,
                "loaded_now": load_seconds > 0,
            },
            "task": {
                "requested": requested_task,
                "token": task_token,
                "prompt": prompt,
                "label": TASK_METADATA_BY_TOKEN.get(task_token, {}).get("label", requested_task),
            },
            "image_size": [image.width, image.height],
            "timing": {
                "load_seconds": round(load_seconds, 4),
                "inference_seconds": round(inference_seconds, 4),
                "total_seconds": round(load_seconds + inference_seconds, 4),
            },
            "generated_text": generated_text,
            "parsed": parsed_json,
            "normalized": normalized,
        }

    def _unload_model(self) -> None:
        if self.model is None and self.processor is None:
            return
        self.model = None
        self.processor = None
        _ = gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _validate_model_allowed(self, requested_model_id: str, resolved_model_id: str) -> None:
        if resolved_model_id not in self.allowed_models:
            allowed = ", ".join(self.allowed_models)
            raise FlorenceError(
                f"Unsupported model {requested_model_id!r}. Use one of the models from /v1/models: {allowed}"
            )


class FlorenceHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "FlorenceHTTP/1.0"

    @property
    def runtime(self) -> FlorenceRuntime:
        return self.server.runtime  # type: ignore[attr-defined]

    def do_OPTIONS(self) -> None:
        self._send_json(HTTPStatus.NO_CONTENT, None)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "name": "Florence-2 server",
                        "summary": "Standalone Florence-2 HTTP API with permissive CORS.",
                        "run_with": "sahara/.venv/Scripts/python.exe",
                        "endpoints": {
                            "GET /health": "liveness + loaded model info",
                            "GET /v1/models": "allowed models",
                            "GET /v1/tasks": "supported task aliases / tokens",
                            "POST /v1/run": "run a Florence task",
                        },
                        **self.runtime.status_payload(),
                    },
                )
                return
            if path == "/health":
                self._send_json(HTTPStatus.OK, {"ok": True, **self.runtime.status_payload()})
                return
            if path in {"/v1/models", "/models"}:
                self._send_json(HTTPStatus.OK, self.runtime.get_models_payload())
                return
            if path in {"/v1/tasks", "/tasks"}:
                self._send_json(HTTPStatus.OK, self.runtime.get_tasks_payload())
                return
            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
        except Exception as exc:
            self._handle_unexpected_error(exc)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path not in {"/v1/run", "/run"}:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
                return
            payload = self._read_json_body()
            result = self.runtime.run(payload)
            self._send_json(HTTPStatus.OK, result)
        except FlorenceError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except json.JSONDecodeError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")
        except Exception as exc:
            self._handle_unexpected_error(exc)

    def _read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise FlorenceError(f"Invalid Content-Length: {raw_length!r}") from exc
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise FlorenceError("JSON body must be an object.")
        return data

    def _handle_unexpected_error(self, exc: Exception) -> None:
        traceback.print_exc()
        self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), error_type=type(exc).__name__)

    def _send_error(self, status: HTTPStatus, message: str, *, error_type: str = "error") -> None:
        self._send_json(status, {"ok": False, "error": error_type, "message": message})

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        if payload is None:
            body = b""
        else:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


class FlorenceHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], runtime: FlorenceRuntime) -> None:
        super().__init__(server_address, FlorenceHTTPRequestHandler)
        self.runtime = runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Allowed model id or local model directory. Repeat to expose multiple models. "
            "Defaults to the 4 florence-community Florence-2 checkpoints."
        ),
    )
    parser.add_argument(
        "--default-model",
        default=DEFAULT_MODEL,
        help=f"Default model id used when the request omits 'model' (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def normalize_allowed_models(model_ids: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for model_id in model_ids:
        resolved = resolve_model_id(model_id)
        if resolved not in seen:
            normalized.append(resolved)
            seen.add(resolved)
    return normalized


def resolve_device(device_request: str) -> str:
    request = (device_request or "auto").strip().lower()
    if request == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    if request.startswith("cuda") and not torch.cuda.is_available():
        print("warning: CUDA requested but not available; falling back to CPU", file=sys.stderr)
        return "cpu"
    return request


def resolve_model_id(model_id: str) -> str:
    model_id = str(model_id).strip()
    return OFFICIAL_TO_COMMUNITY.get(model_id, model_id)


def normalize_task(task: str) -> str:
    task = task.strip()
    if task in TASK_ALIASES:
        return TASK_ALIASES[task]
    if task in TASK_ALIASES.values():
        return task
    known = ", ".join(sorted(TASK_ALIASES))
    raise FlorenceError(f"Unsupported task {task!r}. Use one of: {known}, or a raw Florence token.")


def load_image(image_source: Any) -> Image.Image:
    if not isinstance(image_source, str):
        raise FlorenceError("'image' must be a string (data URL, base64, path, or URL).")
    source = image_source.strip()
    if not source:
        raise FlorenceError("'image' must not be empty.")

    if source.startswith("data:image/"):
        try:
            _, encoded = source.split(",", 1)
        except ValueError as exc:
            raise FlorenceError("Invalid image data URL.") from exc
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    if source.startswith(("http://", "https://")):
        with urlopen(source) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    path = Path(source)
    if path.exists():
        return Image.open(path).convert("RGB")

    try:
        return Image.open(io.BytesIO(base64.b64decode(source))).convert("RGB")
    except Exception as exc:
        raise FlorenceError(
            "Could not decode 'image'. Expected a data URL, raw base64 image, existing local path, or http(s) URL."
        ) from exc


def move_batch_to_device(batch: dict[str, Any], *, device: str, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def sync_cuda_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(model_id: str, *, dtype: torch.dtype) -> Any:
    try:
        return FlorenceModelLoader.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        return FlorenceModelLoader.from_pretrained(model_id, torch_dtype=dtype)


def build_prompt(task_token: str, text_input: str | None) -> str:
    return task_token + text_input if text_input else task_token


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def normalize_result(parsed: Any, task_token: str) -> dict[str, Any]:
    root = parsed.get(task_token) if isinstance(parsed, dict) and task_token in parsed else parsed

    if task_token in {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<REGION_TO_DESCRIPTION>"}:
        return {
            "kind": "text",
            "text": root if isinstance(root, str) else json.dumps(root, ensure_ascii=False),
        }

    if isinstance(root, str):
        return {"kind": "text", "text": root}

    if isinstance(root, dict):
        labels = ensure_list(root.get("labels"))
        scores = ensure_list(root.get("scores"))
        bboxes = ensure_list(root.get("bboxes"))
        quad_boxes = ensure_list(root.get("quad_boxes"))
        polygons = ensure_list(root.get("polygons"))
        regions = max(len(labels), len(scores), len(bboxes), len(quad_boxes), len(polygons))

        items = []
        for index in range(regions):
            item: dict[str, Any] = {"index": index}
            if index < len(labels):
                item["label"] = labels[index]
            if index < len(scores):
                item["score"] = scores[index]
            if index < len(bboxes):
                item["bbox"] = bboxes[index]
            if index < len(quad_boxes):
                item["quad_box"] = quad_boxes[index]
            if index < len(polygons):
                item["polygon"] = polygons[index]
            items.append(item)

        if items:
            return {
                "kind": "regions",
                "items": items,
                "labels": labels,
                "scores": scores,
                "bboxes": bboxes,
                "quad_boxes": quad_boxes,
                "polygons": polygons,
            }

    return {"kind": "object", "value": root}


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def main() -> int:
    args = parse_args()
    allowed_models = args.model or list(DEFAULT_MODELS)
    runtime_default_model = resolve_model_id(args.default_model)
    if (
        args.model
        and args.default_model == DEFAULT_MODEL
        and runtime_default_model not in normalize_allowed_models(allowed_models)
    ):
        runtime_default_model = normalize_allowed_models(allowed_models)[0]

    runtime = FlorenceRuntime(
        device_request=args.device,
        allowed_models=allowed_models,
        default_model=runtime_default_model,
    )
    server = FlorenceHTTPServer((args.host, args.port), runtime)

    print(f"Florence server listening on http://{args.host}:{args.port}")
    print(f"Run with interpreter: {sys.executable}")
    print(f"Device: {runtime.device}")
    print(f"DType:  {str(runtime.dtype).replace('torch.', '')}")
    print(f"Loader: {MODEL_LOADER_NAME}")
    print("Allowed models:")
    for model_id in runtime.allowed_models:
        print(f"  - {model_id}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Florence server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
