"""Local browser demo. Run: python -m cra373.demo.demo_app --port 7860."""
from __future__ import annotations
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from .dashboard_port import reserve_http_server
from .scenarios import NAMES, load_scenario
from .pipeline import analyze

HTML = (Path(__file__).with_name("index.html")).read_bytes()
_live_lock = Lock()
_live_payload: bytes | None = None

def publish_live(payload: dict) -> None:
    """Publish a complete same-step simulation snapshot for the dashboard."""
    global _live_payload
    encoded = json.dumps(payload, allow_nan=False).encode()
    with _live_lock:
        _live_payload = encoded

def make_server(host: str = "127.0.0.1", port: int = 7860, *, fallback_ports: int = 0) -> ThreadingHTTPServer:
    """Bind now so a busy dashboard port is found before simulation startup."""
    return reserve_http_server(host, port, Handler, fallback_ports=fallback_ports)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if self.path.startswith("/api/live"):
            return
        super().log_message(format, *args)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            html = HTML
            if parse_qs(url.query).get("snapshot") == ["1"]:
                with _live_lock:
                    payload = _live_payload
                if payload is None:
                    self.send_error(503, "Waiting for an Isaac Sim frame")
                    return
                # Freeze one real published frame so a browser capture shows all six cards.
                safe_json = payload.replace(b"<", b"\\u003c")
                script = b"<script>window.__cra373Snapshot=" + safe_json + b"</script>"
                html = HTML.replace(b"<!-- SNAPSHOT_SCRIPT -->", script, 1)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif url.path == "/api/live":
            with _live_lock:
                payload = _live_payload
            if payload is None:
                payload = b'{"status":"waiting","message":"Waiting for Isaac Sim frames"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif url.path == "/api/scenario":
            name = parse_qs(url.query).get("name", ["Flat"])[0]
            if name not in NAMES:
                self.send_error(404, "unknown scenario")
                return
            payload = json.dumps(analyze(load_scenario(name)), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    print(f"CRA373 terrain demo: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
