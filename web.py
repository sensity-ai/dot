#!/usr/bin/env python3
"""Tiny local web control panel for DOT live."""

from __future__ import annotations

import argparse
import cgi
import json
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

PROCESS: subprocess.Popen | None = None
STARTED_AT: float | None = None


INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DOT Live</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101214; color: #f4f4f1; }
    main { max-width: 820px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 28px; margin: 0 0 22px; }
    .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }
    label { display: grid; gap: 7px; font-size: 13px; color: #c8c8c2; }
    input, select, button { font: inherit; border-radius: 8px; border: 1px solid #34383c; background: #181b1f; color: #f4f4f1; padding: 10px 12px; min-width: 0; }
    button { cursor: pointer; background: #f0d36a; color: #181400; border: 0; font-weight: 650; }
    button.stop { background: #d95f5f; color: white; }
    section { margin-top: 18px; padding: 16px; border: 1px solid #292d31; border-radius: 8px; background: #15181b; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
    pre { white-space: pre-wrap; margin: 0; color: #cdd2d6; }
  </style>
</head>
<body>
<main>
  <h1>DOT Live</h1>
  <div class="grid">
    <label>Source <input id="source" value="data/source_face.webm"></label>
    <label>Upload source <input id="uploadFile" type="file" accept="image/*,video/*"></label>
    <label>Camera <input id="camera" value="1" inputmode="numeric"></label>
    <label>Backend <select id="backend"><option>simswap</option><option>onnx</option></select></label>
    <label>Style <select id="style"><option>swap</option><option>avatar</option></select></label>
    <label>Preset <select id="preset"><option>natural-max</option><option>natural</option><option>balanced</option><option>fast</option></select></label>
    <label>Output <select id="output"><option>window</option><option>both</option><option>virtualcam</option></select></label>
  </div>
  <div class="actions">
    <button onclick="upload()">Upload</button>
    <button onclick="start()">Start</button>
    <button class="stop" onclick="stop()">Stop</button>
    <button onclick="status()">Refresh</button>
  </div>
  <section><pre id="status">Loading...</pre></section>
</main>
<script>
async function api(path, body) {
  const options = body ? { method: "POST", body: new URLSearchParams(body) } : {};
  const res = await fetch(path, options);
  const text = await res.text();
  try { return JSON.parse(text); } catch { return { ok: false, text }; }
}
async function upload() {
  const file = uploadFile.files[0];
  if (!file) {
    render({ ok: false, error: "choose a source image or video first" });
    return;
  }
  const body = new FormData();
  body.append("source", file);
  const res = await fetch("/api/upload", { method: "POST", body });
  const data = await res.json();
  if (data.path) source.value = data.path;
  render(data);
}
function form() {
  return {
    source: source.value,
    camera: camera.value,
    backend: backend.value,
    style: style.value,
    preset: preset.value,
    output: output.value
  };
}
function render(data) {
  document.getElementById("status").textContent = JSON.stringify(data, null, 2);
}
async function start() { render(await api("/api/start", form())); }
async function stop() { render(await api("/api/stop", {})); }
async function status() { render(await api("/api/status")); }
status();
setInterval(status, 3000);
</script>
</body>
</html>
"""


def process_status() -> dict:
    global PROCESS
    if PROCESS is not None and PROCESS.poll() is not None:
        code = PROCESS.returncode
        PROCESS = None
        return {"running": False, "last_exit": code}
    if PROCESS is None:
        return {"running": False}
    uptime = round(time.time() - (STARTED_AT or time.time()), 1)
    return {"running": True, "pid": PROCESS.pid, "uptime_seconds": uptime}


def stop_process() -> dict:
    global PROCESS
    if PROCESS is None or PROCESS.poll() is not None:
        PROCESS = None
        return {"ok": True, "running": False}
    PROCESS.send_signal(signal.SIGINT)
    try:
        PROCESS.wait(timeout=10)
    except subprocess.TimeoutExpired:
        PROCESS.terminate()
        PROCESS.wait(timeout=5)
    PROCESS = None
    return {"ok": True, "running": False}


def safe_upload_name(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_SUFFIXES))
        raise ValueError(f"unsupported file type {suffix!r}; allowed: {allowed}")
    stem = Path(filename).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    safe_stem = safe_stem.strip("_") or "source"
    return f"{int(time.time())}_{safe_stem}{suffix}"


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, INDEX.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            self._json(process_status())
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        global PROCESS, STARTED_AT
        length = int(self.headers.get("Content-Length", "0"))
        parsed = urlparse(self.path)

        if parsed.path == "/api/upload":
            self.handle_upload(length)
            return

        form = parse_qs(self.rfile.read(length).decode("utf-8"))

        if parsed.path == "/api/stop":
            self._json(stop_process())
            return

        if parsed.path != "/api/start":
            self._json({"ok": False, "error": "not found"}, 404)
            return

        current = process_status()
        if current.get("running"):
            self._json({"ok": False, "error": "already running", **current}, 409)
            return

        def value(name: str, default: str) -> str:
            return form.get(name, [default])[0]

        style = value("style", "swap")
        preset = value("preset", "natural")
        cmd = [
            str(ROOT / "run.sh"),
            "--source", value("source", "data/source_face.webm"),
            "--camera", value("camera", "1"),
            "--backend", value("backend", "simswap"),
            "--style", style,
            "--preset", "fast" if style == "avatar" else preset,
            "--output", value("output", "window"),
        ]
        PROCESS = subprocess.Popen(cmd, cwd=ROOT)
        STARTED_AT = time.time()
        self._json({"ok": True, "cmd": cmd, **process_status()})

    def handle_upload(self, length: int) -> None:
        if length <= 0:
            self._json({"ok": False, "error": "empty upload"}, 400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._json({"ok": False, "error": "upload too large; max is 500MB"}, 413)
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._json({"ok": False, "error": "expected multipart/form-data"}, 400)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(length),
            },
        )
        field = form["source"] if "source" in form else None
        if field is None or not getattr(field, "filename", None):
            self._json({"ok": False, "error": "missing source file"}, 400)
            return

        try:
            filename = safe_upload_name(field.filename)
        except ValueError as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / filename
        written = 0
        with path.open("wb") as out:
            while True:
                chunk = field.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    path.unlink(missing_ok=True)
                    self._json({"ok": False, "error": "upload too large; max is 500MB"}, 413)
                    return
                out.write(chunk)

        rel = path.relative_to(ROOT).as_posix()
        prepared_rel = None
        if path.suffix.lower() in IMAGE_SUFFIXES:
            prepared = UPLOAD_DIR / f"prepared_{path.stem}.jpg"
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "live.py"),
                        "--source",
                        rel,
                        "--prepare-source",
                        prepared.relative_to(ROOT).as_posix(),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=45,
                    check=True,
                )
                prepared_rel = result.stdout.strip().splitlines()[-1]
            except Exception as exc:
                self._json({
                    "ok": True,
                    "path": rel,
                    "bytes": written,
                    "source_prepare_warning": str(exc),
                })
                return

        self._json({
            "ok": True,
            "path": prepared_rel or rel,
            "original_path": rel,
            "prepared": prepared_rel is not None,
            "bytes": written,
        })

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="DOT local web control panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DOT web control: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web control.")
    finally:
        stop_process()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
