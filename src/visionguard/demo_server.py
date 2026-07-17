from __future__ import annotations

import json
import mimetypes
import secrets
import threading
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from visionguard.demo import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    DemoConfig,
    VisionGuardDemoEngine,
)


@dataclass(frozen=True)
class DemoServerConfig:
    model: Path
    output_dir: Path
    host: str = "127.0.0.1"
    port: int = 7860
    imgsz: int = 512
    confidence: float = 0.25
    device: str = "auto"
    max_upload_mb: int = 150
    max_video_seconds: float = 120.0


def _safe_child(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Requested path escapes the configured root.")
    return candidate


def _parse_multipart(content_type: str, body: bytes) -> dict[str, Any]:
    if "multipart/form-data" not in content_type.lower():
        raise ValueError("Expected multipart/form-data upload.")
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    fields: dict[str, Any] = {}
    if not message.is_multipart():
        raise ValueError("Malformed multipart upload.")
    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename:
            fields[field_name] = {"filename": Path(filename).name, "content": payload}
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[field_name] = payload.decode(charset, errors="replace")
    return fields


def _handler_factory(
    engine: VisionGuardDemoEngine,
    server_config: DemoServerConfig,
    static_dir: Path,
    output_dir: Path,
) -> type[BaseHTTPRequestHandler]:
    inference_lock = threading.Lock()
    upload_limit = server_config.max_upload_mb * 1024 * 1024

    class DemoRequestHandler(BaseHTTPRequestHandler):
        server_version = "VisionGuardDemo/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self._send_file(static_dir / "index.html")
                return
            if path == "/api/config":
                self._send_json(
                    {
                        "model": engine.config.model.name,
                        "device": engine.config.device,
                        "imgsz": engine.config.imgsz,
                        "confidence": engine.config.confidence,
                        "max_upload_mb": server_config.max_upload_mb,
                        "max_video_seconds": server_config.max_video_seconds,
                        "image_extensions": sorted(IMAGE_SUFFIXES),
                        "video_extensions": sorted(VIDEO_SUFFIXES),
                    }
                )
                return
            if path.startswith("/static/"):
                try:
                    file_path = _safe_child(static_dir, path.removeprefix("/static/"))
                except ValueError as exc:
                    self._send_error(400, str(exc))
                    return
                self._send_file(file_path)
                return
            if path.startswith("/results/"):
                try:
                    file_path = _safe_child(output_dir, path.removeprefix("/results/"))
                except ValueError as exc:
                    self._send_error(400, str(exc))
                    return
                self._send_file(file_path)
                return
            self._send_error(404, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/infer":
                self._send_error(404, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_error(400, "Invalid Content-Length")
                return
            if length <= 0:
                self._send_error(400, "Upload body is empty")
                return
            if length > upload_limit:
                self._send_error(
                    413, f"Upload exceeds the {server_config.max_upload_mb} MB limit"
                )
                return

            try:
                fields = _parse_multipart(
                    self.headers.get("Content-Type", ""), self.rfile.read(length)
                )
                upload = fields.get("file")
                if not isinstance(upload, dict) or not upload.get("content"):
                    raise ValueError("The multipart field 'file' is required.")
                filename = str(upload.get("filename") or "upload")
                suffix = Path(filename).suffix.lower()
                if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
                    raise ValueError(f"Unsupported file extension: {suffix or 'none'}")

                session_id = secrets.token_hex(8)
                session_dir = output_dir / session_id
                session_dir.mkdir(parents=True, exist_ok=False)
                input_path = session_dir / f"input{suffix}"
                input_path.write_bytes(upload["content"])
                try:
                    with inference_lock:
                        report = engine.infer(input_path, session_dir)
                finally:
                    input_path.unlink(missing_ok=True)
                report["result_url"] = f"/results/{session_id}/{report['output_file']}"
                report["report_url"] = f"/results/{session_id}/report.json"
                self._send_json(report)
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                self._send_error(400, f"{type(exc).__name__}: {exc}")

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_error(404, "File not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            payload = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"[demo] {self.address_string()} - {format_string % args}")

    return DemoRequestHandler


def run_demo_server(config: DemoServerConfig) -> None:
    project_root = Path(__file__).resolve().parents[2]
    static_dir = project_root / "demo" / "static"
    if not (static_dir / "index.html").is_file():
        raise FileNotFoundError(f"Demo frontend is missing: {static_dir / 'index.html'}")
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = VisionGuardDemoEngine(
        DemoConfig(
            model=config.model,
            output_dir=output_dir,
            imgsz=config.imgsz,
            confidence=config.confidence,
            device=config.device,
            max_video_seconds=config.max_video_seconds,
        )
    )
    handler = _handler_factory(engine, config, static_dir, output_dir)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    print(f"VisionGuard demo ready: http://{config.host}:{config.port}")
    print(f"Model: {engine.config.model} | device={engine.config.device}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
