"""Wall-clock guarantees for streamed model calls.

Two independent guards are asserted here, each against a real HTTP server that
behaves the way the failure mode needs:

1. A server that accepts the request and then says nothing at all must be cut
   off by the transport-level read timeout. The between-chunk deadline check
   cannot help, because no chunk ever arrives for it to inspect.
2. A server that trickles one event every couple of seconds must be cut off by
   the between-chunk deadline, because the trickling resets the read timeout.

No model server is required, so this is fast and safe to run anywhere.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
import httpx  # noqa: E402

BUDGET_S = 8.0
TOLERANCE_S = 6.0


class _SilentHandler(BaseHTTPRequestHandler):
    """Accepts the request, then never writes a response byte."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        time.sleep(120)


class _TrickleHandler(BaseHTTPRequestHandler):
    """Emits one well-formed SSE event every TRICKLE_S, forever."""

    protocol_version = "HTTP/1.1"
    TRICKLE_S = 2.0

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        deadline = time.time() + 120
        try:
            while time.time() < deadline:
                body = (
                    "data: " + json.dumps({
                        "id": "1", "object": "chat.completion.chunk", "created": 0,
                        "model": "fake",
                        "choices": [{"index": 0, "delta": {"content": "."}}],
                    }) + "\n\n"
                ).encode()
                self.wfile.write(b"%x\r\n" % len(body) + body + b"\r\n")
                self.wfile.flush()
                time.sleep(self.TRICKLE_S)
        except Exception:
            pass


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _call(port, budget=BUDGET_S):
    client = app.build_client("http://127.0.0.1:%d/v1" % port)
    started = time.time()
    try:
        result = app.chat(client, "fake", [{"role": "user", "content": "hi"}],
                          max_tokens=16, timeout=budget)
        return time.time() - started, result["finish_reason"], result["content"]
    except app.ModelDeadline:
        return time.time() - started, "ModelDeadline", ""
    except Exception as exc:
        return time.time() - started, "error:%s" % type(exc).__name__, str(exc)


def t_silent_server_cut_off():
    """Guard 1: transport read timeout fires when nothing is ever sent."""
    server, port = _serve(_SilentHandler)
    try:
        elapsed, how, _ = _call(port)
    finally:
        server.shutdown()
        server.server_close()
    assert elapsed < BUDGET_S + TOLERANCE_S, (
        "silent server ran %.1fs; read timeout not enforced (budget %.0fs)"
        % (elapsed, BUDGET_S)
    )
    # The SDK raises APITimeoutError, which is not an httpx.TimeoutException.
    # It must still reach the caller as a ModelDeadline so the UI can say
    # "Stopped" instead of showing a raw traceback.
    assert how == "ModelDeadline", (
        "silent server surfaced as %s, expected ModelDeadline" % how
    )
    return "silent server cut off in %.1fs as ModelDeadline" % elapsed


def t_trickling_server_cut_off():
    """Guard 2: the between-chunk deadline fires when bytes keep arriving."""
    server, port = _serve(_TrickleHandler)
    try:
        elapsed, how, _ = _call(port)
    finally:
        server.shutdown()
        server.server_close()
    assert elapsed < BUDGET_S + TOLERANCE_S, (
        "trickling server ran %.1fs; deadline not enforced (budget %.0fs)"
        % (elapsed, BUDGET_S)
    )
    assert how == "deadline", "expected finish_reason 'deadline', got %r" % how
    return "trickling server cut off in %.1fs with finish_reason=deadline" % elapsed


def t_sdk_timeout_is_not_an_httpx_timeout():
    """Lock in why TIMEOUT_ERRORS exists at all."""
    from openai import APITimeoutError

    assert not issubclass(APITimeoutError, httpx.TimeoutException)
    assert httpx.TimeoutException in app.TIMEOUT_ERRORS
    assert APITimeoutError in app.TIMEOUT_ERRORS
    return "APITimeoutError covered by TIMEOUT_ERRORS"


def t_read_timeout_tracks_budget():
    """The read window must equal the budget, never exceed it."""
    got = {
        "text": app._stream_timeouts(app.MODEL_DEADLINE_S).read,
        "tool": app._stream_timeouts(30).read,
        "vision": app.vision_timeouts().read,
        "tiny": app._stream_timeouts(2).read,
    }
    assert got["text"] == app.MODEL_DEADLINE_S, got
    assert got["tool"] == 30, got
    assert got["vision"] == app.VISION_TIMEOUT_S, got
    # The old floor of 30s meant a nearly-exhausted budget still waited 30s.
    assert got["tiny"] <= 5.0, got
    v = app.vision_timeouts()
    assert (v.connect, v.write, v.pool) == (5.0, 30.0, 10.0), v
    return "text=%s tool=%s vision=%s tiny=%s" % (
        got["text"], got["tool"], got["vision"], got["tiny"]
    )


def t_client_does_not_retry():
    """Retries would silently multiply a stalled request past its budget."""
    client = app.build_client(app.DEFAULT_BASE_URL)
    assert client.max_retries == 0, client.max_retries
    t = client.timeout
    assert t.connect and t.read and t.write and t.pool, t
    return "max_retries=%s read=%s" % (client.max_retries, t.read)


TESTS = [
    ("silent server cut off by transport read timeout", t_silent_server_cut_off),
    ("trickling server cut off by between-chunk deadline", t_trickling_server_cut_off),
    ("read window tracks the remaining budget", t_read_timeout_tracks_budget),
    ("client sets granular timeouts and never retries", t_client_does_not_retry),
    ("SDK timeout mapped to ModelDeadline", t_sdk_timeout_is_not_an_httpx_timeout),
]


def main():
    print("== wall-clock guarantees ==")
    passed = failed = 0
    for name, fn in TESTS:
        try:
            detail = fn()
            passed += 1
            print("  PASS  %s  (%s)" % (name, detail))
        except AssertionError as exc:
            failed += 1
            print("  FAIL  %s\n          %s" % (name, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("  ERROR %s: %r" % (name, exc))
    print("\n%d/%d timeout checks passed" % (passed, passed + failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())