import io
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402

RESULTS = []


def check(name, fn):
    started = time.time()
    try:
        detail = fn()
        RESULTS.append(("PASS", name, detail))
    except Exception as exc:
        RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
    print(f"  {RESULTS[-1][0]}  {name}  ({time.time() - started:.1f}s)  {RESULTS[-1][2]}")


BASE = os.environ.get("LLM_BASE_URL", app.DEFAULT_BASE_URL)


def t_guard_installed():
    assert getattr(socket.socket, "_servant_guarded", False)
    return "socket.connect wrapped at import time"


def t_block_external_ip():
    before = app.net_stats()[0]
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
    except ConnectionError as exc:
        assert "Sovereignty guard" in str(exc), str(exc)
        assert app.net_stats()[0] == before + 1
        return "raised ConnectionError, counter +1"
    raise AssertionError("external connect was NOT blocked")


def t_block_external_hostname():
    try:
        socket.create_connection(("example.com", 80), timeout=5)
    except ConnectionError as exc:
        return f"blocked: {str(exc)[:60]}"
    raise AssertionError("external hostname was NOT blocked")


def t_connect_ex_blocked():
    s = socket.socket()
    try:
        code = s.connect_ex(("1.1.1.1", 443))
    finally:
        s.close()
    assert code == 10061, code
    return "connect_ex returned 10061"


def t_allow_loopback_ip():
    s = socket.socket()
    s.settimeout(5)
    try:
        s.connect(("127.0.0.1", 11434))
    finally:
        s.close()
    return "connected to 127.0.0.1:11434"


def t_allow_localhost_name():
    s = socket.socket()
    s.settimeout(5)
    try:
        s.connect(("localhost", 11434))
    finally:
        s.close()
    return "connected to localhost:11434"


def t_netlog_persisted():
    path = Path(app.NETLOG_PATH)
    assert path.exists(), "netlog.jsonl missing"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "netlog empty"
    import json

    last = json.loads(lines[-1])
    assert last["verdict"] in ("ALLOWED", "BLOCKED")
    assert "dest" in last and "ts" in last
    return f"{len(lines)} lines, last verdict={last['verdict']} dest={last['dest']}"


def t_net_stats():
    blocked, allowed = app.net_stats()
    assert blocked > 0 and allowed > 0
    return f"blocked={blocked} allowed={allowed}"


def t_strip_thinking():
    assert app.strip_thinking("<think>hmm long</think>hello") == "hello"
    assert app.strip_thinking("<think>unterminated reasoning") == ""
    assert app.strip_thinking("plain answer") == "plain answer"
    assert app.strip_thinking("") == ""
    return "closed + unterminated + clean"


def t_parser_clean():
    got = app.parse_first_json_object('{"tool": "run_python", "args": {"code": "print(1)"}}')
    assert got["tool"] == "run_python" and got["args"]["code"] == "print(1)"
    return "clean JSON"


def t_parser_fenced():
    got = app.parse_first_json_object(
        'Sure, here it is:\n```json\n{"tool": "make_docx", "args": {"title": "Note"}}\n```'
    )
    assert got and got["tool"] == "make_docx"
    return "extracted from fenced block with preamble"


def t_parser_embedded():
    got = app.parse_first_json_object('I will call {"tool": "search_kb", "args": {"query": "x"}} now.')
    return "extracted embedded object" if got else "FAILED to extract"


def t_parser_trailing_comma():
    got = app.parse_first_json_object('{"tool": "run_python", "args": {"code": "x",},}')
    return "repaired trailing comma" if got else "FAILED to repair"


def t_parser_single_quotes():
    got = app.parse_first_json_object("{'tool': 'make_docx', 'args': {}}")
    return "repaired single quotes" if got else "FAILED to repair"


def t_parser_rejects_prose():
    assert app.parse_first_json_object("The answer is 42.") is None
    assert app.parse_first_json_object("") is None
    assert app.parse_first_json_object(None) is None
    return "prose and empties correctly rejected"


def t_tesseract():
    exe = app.find_tesseract()
    assert exe and Path(exe).exists(), exe
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    img = Image.new("RGB", (820, 240), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "FINDING F-014", fill="black", font=font)
    d.text((20, 90), "Severity: HIGH", fill="black", font=font)
    d.text((20, 160), "Due date: 2026-11-04", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img2 = Image.open(io.BytesIO(buf.getvalue()))
    text, engine = app.ocr_image(img2)
    assert engine == "tesseract", engine
    for expected in ("F-014", "Severity: HIGH", "2026-11-04"):
        assert expected in text, f"missing {expected!r} in {text!r}"
    return f"engine={engine} chars={len(text)} all fields exact"


def t_ocr_first_not_vision():
    from PIL import Image

    img = Image.new("RGB", (400, 120), "white")
    text, engine = app.extract_text_from_image(img, "", None, "")
    assert "tesseract" in engine, engine
    return f"tesseract path chosen without a vision model (engine={engine})"


def t_private_endpoint():
    assert app.is_private_endpoint("http://localhost:11434/v1")
    assert app.is_private_endpoint("http://127.0.0.1:8000/v1")
    assert app.is_private_endpoint("http://192.168.1.50:11434/v1")
    assert not app.is_private_endpoint("https://api.openai.com/v1")
    return "localhost/127.0.0.1/private ok, public rejected"


def t_models_listed():
    models = app.list_models(app.build_client(BASE))
    assert models, "no models returned"
    roles = app.guess_roles(models)
    assert roles.chat, "no chat model guessed"
    assert not app._is_embed_name(roles.chat), f"chat guessed as embed model {roles.chat}"
    assert roles.embed, "no embedding model guessed"
    return f"{models} -> chat={roles.chat} embed={roles.embed} vision={roles.vision!r}"


def t_prompts():
    kb = app.build_system_prompt("kb", "SOME DOCUMENT BODY")
    gen = app.build_system_prompt("general")
    assert "Not found in the knowledge base." in kb
    assert "untrusted data" in kb and "<context>" in kb and "SOME DOCUMENT BODY" in kb
    assert "run_python" not in kb, "run_python leaked into the KB prompt"
    assert "search_kb" in kb and "make_docx" in kb
    assert "How to run" in gen
    assert "run_python" in gen and "search_kb" not in gen
    assert app.ANTI_THINK in kb and app.ANTI_THINK in gen
    return "KB refusal+citation+anti-injection+no run_python; General how-to-run+tool split"


def t_chat_tool_turn():
    client = app.build_client(BASE)
    result = app.chat(
        client,
        app.guess_roles(app.list_models(client)).chat,
        [
            {
                "role": "user",
                "content": app.ANTI_THINK
                + '\n\nReply with exactly this JSON and nothing else: '
                '{"tool": "search_kb", "args": {"query": "pump failure"}}',
            }
        ],
        max_tokens=app.TOOL_TURN_TOKEN_CAP,
    )
    assert result["usage"]["completion_tokens"] <= app.TOOL_TURN_TOKEN_CAP
    got = app.parse_first_json_object(result["content"])
    assert got and got["tool"] == "search_kb", result["content"][:200]
    return (
        f"{result['elapsed']:.1f}s tokens={result['usage']['completion_tokens']}"
        f"/{app.TOOL_TURN_TOKEN_CAP} reasoning_discarded={result['reasoning_len']} parsed=search_kb"
    )


def t_vision_timeout():
    from PIL import Image

    img = Image.new("RGB", (300, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64

    b64 = base64.b64encode(buf.getvalue()).decode()
    client = app.build_client(BASE)
    model = app.guess_roles(app.list_models(client)).chat
    started = time.time()
    text, engine = app.vision_transcribe(client, model, b64)
    dt = time.time() - started
    assert dt < 90, f"vision took {dt:.0f}s, timeout not enforced"
    assert not text, "vision unexpectedly returned content"
    return f"gave up in {dt:.0f}s with '{engine}' instead of hanging"


def main():
    print("== guard ==")
    for n, f in [
        ("guard installed at import", t_guard_installed),
        ("block external IP 8.8.8.8", t_block_external_ip),
        ("block external hostname example.com", t_block_external_hostname),
        ("connect_ex blocked", t_connect_ex_blocked),
        ("allow loopback 127.0.0.1", t_allow_loopback_ip),
        ("allow localhost by name", t_allow_localhost_name),
        ("netlog.jsonl persisted", t_netlog_persisted),
        ("net_stats counters", t_net_stats),
    ]:
        check(n, f)

    print("== text processing ==")
    for n, f in [
        ("strip_thinking", t_strip_thinking),
        ("parser: clean", t_parser_clean),
        ("parser: fenced", t_parser_fenced),
        ("parser: embedded", t_parser_embedded),
        ("parser: trailing comma", t_parser_trailing_comma),
        ("parser: single quotes", t_parser_single_quotes),
        ("parser: rejects prose", t_parser_rejects_prose),
        ("private endpoint check", t_private_endpoint),
    ]:
        check(n, f)

    print("== ocr ==")
    for n, f in [
        ("tesseract accuracy", t_tesseract),
        ("tesseract before vision", t_ocr_first_not_vision),
    ]:
        check(n, f)

    print("== connector (live) ==")
    for n, f in [
        ("models listed", t_models_listed),
        ("prompt contracts", t_prompts),
        ("chat tool-call turn", t_chat_tool_turn),
        ("vision 60s timeout", t_vision_timeout),
    ]:
        check(n, f)

    passed = sum(1 for r in RESULTS if r[0] == "PASS")
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\n{passed}/{len(RESULTS)} passed")
    for status, name, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
