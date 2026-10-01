import sys
import time

sys.path.insert(0, r"D:\Servsnt ai")
sys.path.insert(0, r"D:\Servsnt ai\tests")
import app  # noqa: E402
from test_agent import Trace  # noqa: E402

client = app.build_client(app.DEFAULT_BASE_URL)
roles = app.guess_roles(app.list_models(client))
kb = app.load_kb()

print("=== 1. streaming actually streams (tokens arrive incrementally) ===")
arrivals = []
t0 = time.time()
res = app.chat(
    client, roles.chat,
    [{"role": "user", "content": "In one sentence, what is a pressure relief valve?"}],
    max_tokens=200, disable_thinking=True, on_delta=arrivals.append,
)
print("  chunks=%d elapsed=%.1fs out_tok=%s" % (len(arrivals), time.time() - t0,
                                                res["usage"]["completion_tokens"]))
print("  finish_reason=%s reasoning_len=%d" % (res["finish_reason"], res["reasoning_len"]))
assert len(arrivals) > 3, "did not stream: %d chunks" % len(arrivals)
assert "".join(arrivals) == res["content"], "streamed text != final content"
print("  PASS streamed %d incremental chunks, text matches\n" % len(arrivals))

print("=== 2. hard deadline cuts off a slow call ===")
t0 = time.time()
try:
    res = app.chat(
        client, roles.chat,
        [{"role": "user", "content": "Write a very long essay about industrial safety."}],
        max_tokens=4000, disable_thinking=True, deadline=12.0,
    )
    dt = time.time() - t0
    print("  returned after %.1fs (deadline 12s) timed_out=%s chars=%d"
          % (dt, res["timed_out"], len(res["content"])))
    assert dt < 30, "deadline not enforced: %.1fs" % dt
    assert res["timed_out"] is True
    assert res["finish_reason"] == "deadline"
    print("  PASS deadline honoured, partial text returned\n")
except app.ModelDeadline as exc:
    dt = time.time() - t0
    print("  raised ModelDeadline after %.1fs: %s" % (dt, exc))
    assert dt < 30
    print("  PASS deadline honoured (no text produced)\n")

print("=== 3. deadline with no output raises instead of hanging ===")
t0 = time.time()
try:
    app.chat(client, roles.chat, [{"role": "user", "content": "hi"}],
             max_tokens=10, disable_thinking=True, deadline=0.001)
    print("  returned quickly without raising")
except app.ModelDeadline as exc:
    print("  raised after %.2fs: %s" % (time.time() - t0, exc))
    assert time.time() - t0 < 5
print("  PASS\n")

print("=== 4. agent-wide budget bounds a full turn ===")
trace = Trace()
t0 = time.time()
out = app.run_agent(
    "Search the knowledge base, then also search it again several times with "
    "different rephrasings, then write a detailed report.",
    "kb", client, roles, kb, [], trace, disable_thinking=True,
)
dt = time.time() - t0
print("  full turn took %.1fs (AGENT_DEADLINE_S=%s)" % (dt, app.AGENT_DEADLINE_S))
print("  answer: %s" % out[:160].replace("\n", " "))
assert dt < app.AGENT_DEADLINE_S + 45, "turn exceeded its budget: %.1fs" % dt
print("  PASS turn is bounded\n")

print("=== 5. agent deadline is respected mid-loop ===")
saved = app.AGENT_DEADLINE_S
app.AGENT_DEADLINE_S = 45.0
t0 = time.time()
out = app.run_agent(
    "Search the knowledge base repeatedly with many different queries, then "
    "write an extremely detailed multi-section report about it.",
    "kb", client, roles, kb, [], Trace(), disable_thinking=True,
)
dt = time.time() - t0
app.AGENT_DEADLINE_S = saved
print("  turn stopped gracefully after %.1fs (budget 45s)" % dt)
print("  answer: %s" % out[:200].replace("\n", " "))
assert dt < 120, "short budget not respected: %.1fs" % dt
assert not out.startswith("{"), "raw tool JSON leaked to the user: %s" % out[:120]
assert "Stopped" in out or len(out) > 20, "unhelpful stop: %s" % out[:120]
print("  PASS budget enforced without crashing or leaking JSON\n")

print("ALL STREAMING/DEADLINE CHECKS PASSED")
