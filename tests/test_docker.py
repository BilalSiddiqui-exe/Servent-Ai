"""Static checks on the Docker packaging.

The Docker daemon is not available on the dev box, so `docker compose up` cannot be
exercised here. These assertions pin the parts of T-11 that are contractual: loopback
port binding, the LLM_BASE_URL override, the workspace volume, telemetry off, non-root,
and that the hardened runner argv is fully app-controlled.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print("  %s %s%s" % ("PASS" if ok else "FAIL", name, (" - " + detail) if detail else ""))


print("== Dockerfile ==")
check("based on a pinned python image", "FROM python:3.13-slim" in DOCKERFILE)
check("tesseract installed for scanned pages", "tesseract-ocr" in DOCKERFILE)
check("switches to a non-root user", "USER servant" in DOCKERFILE and "useradd" in DOCKERFILE)
check("listens on 0.0.0.0 inside the container", "--server.address=0.0.0.0" in DOCKERFILE)
check("telemetry disabled in the image", "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false" in DOCKERFILE)
check("hub offline in the image", "HF_HUB_OFFLINE=1" in DOCKERFILE)
check("has a healthcheck", "HEALTHCHECK" in DOCKERFILE and "_stcore/health" in DOCKERFILE)
check("only the workspace is writable", "mkdir -p /app/workspace/out" in DOCKERFILE)
check("copies app and streamlit config",
      "COPY app.py" in DOCKERFILE and "COPY .streamlit/" in DOCKERFILE)

print("\n== docker-compose.yml ==")
check("port bound to loopback only", '"127.0.0.1:8501:8501"' in COMPOSE,
      "must not publish on 0.0.0.0")
check("no published port on all interfaces", "0.0.0.0:8501" not in COMPOSE)
check("LLM_BASE_URL is overridable", "LLM_BASE_URL: \"${LLM_BASE_URL:-" in COMPOSE)
check("workspace is a volume", "servant-workspace:/app/workspace" in COMPOSE)
check("named volume declared", "\nvolumes:\n  servant-workspace:" in COMPOSE)
check("telemetry off in compose", 'STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false"' in COMPOSE)
check("host gateway resolvable", "host.docker.internal:host-gateway" in COMPOSE)
check("drops all capabilities", "cap_drop:" in COMPOSE and "- ALL" in COMPOSE)
check("no new privileges", "no-new-privileges:true" in COMPOSE)
check("pids and memory capped", "pids_limit:" in COMPOSE and "mem_limit:" in COMPOSE)
check("docker socket mount is off by default",
      "# - /var/run/docker.sock" in COMPOSE)

print("\n== .dockerignore ==")
ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
check("workspace excluded from build context", "workspace/" in ignore)
check("pycache excluded", "__pycache__" in ignore)

print("\n== hardened code runner ==")
saved = os.environ.get("SERVANT_RUNNER_IMAGE")
os.environ["SERVANT_RUNNER_IMAGE"] = "python:3.13-slim"
argv = app._runner_argv()
check("runner produces a docker argv", bool(argv) and argv[0].endswith("docker") or "docker" in argv[0],
      " ".join(argv[:4]) if argv else "none")
if argv:
    joined = " ".join(argv)
    check("runner has no network", "--network none" in joined)
    check("runner caps memory", "--memory 512m" in joined)
    check("runner caps cpus", "--cpus 1" in joined)
    check("runner caps pids", "--pids-limit 100" in joined)
    check("runner drops capabilities", "--cap-drop ALL" in joined)
    check("runner mounts workspace read-only", ":ro" in joined)
    check("runner image is app-controlled, not model-controlled",
          argv[-1].endswith("/work/run.py") and "python:3.13-slim" in joined)
    check("no shell in the runner command", "sh" != argv[-2] and "-c" not in argv)
del os.environ["SERVANT_RUNNER_IMAGE"]
check("runner disabled when the env var is absent", app._runner_argv() is None)
if saved:
    os.environ["SERVANT_RUNNER_IMAGE"] = saved

print("\n== local subprocess path still works ==")
out = app.run_python("print('runner-ok')")
check("plain subprocess execution unaffected", "runner-ok" in out and "exit=0" in out, out[:60])

print("\n== docker cli present ==")
which = subprocess.run(["docker", "--version"], capture_output=True, text=True)
check("docker cli on PATH", which.returncode == 0, which.stdout.strip() or which.stderr.strip()[:60])
daemon = subprocess.run(["docker", "info"], capture_output=True, text=True)
check("docker daemon reachable (build NOT verified here if this fails)",
      daemon.returncode == 0,
      "daemon unavailable - `docker compose up --build` is untested on this machine")

passed = sum(1 for _, ok, _ in checks if ok)
print("\n%d/%d docker packaging checks passed" % (passed, len(checks)))
for name, ok, detail in checks:
    if not ok:
        print("  FAILED: %s %s" % (name, detail))
sys.exit(0 if passed == len(checks) else 1)
