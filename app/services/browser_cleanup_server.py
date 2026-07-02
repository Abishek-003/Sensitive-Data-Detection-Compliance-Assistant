from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

_cleanup_callback: Callable[[str], None] | None = None
_server_lock = threading.Lock()
_server_started = False


class _CleanupHandler(BaseHTTPRequestHandler):
    def _set_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/clear-session":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            payload = {}

        session_id = str(payload.get("session_id", "") or "")
        if session_id and _cleanup_callback is not None:
            try:
                _cleanup_callback(session_id)
            except Exception:
                pass

        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def ensure_browser_cleanup_server(callback: Callable[[str], None], port: int) -> None:
    global _cleanup_callback, _server_started

    with _server_lock:
        _cleanup_callback = callback
        if _server_started:
            return

        server = ThreadingHTTPServer(("127.0.0.1", port), _CleanupHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _server_started = True
