"""Bare-mode smoke test: runs the Streamlit UI code path outside a browser session
to surface NameError/TypeError before anyone opens the page."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

import app  # noqa: E402

failures = []

try:
    app.main()
    print("PASS  main() ran to completion in bare mode")
except Exception as exc:
    import traceback

    print(f"FAIL  main() raised {type(exc).__name__}: {exc}")
    traceback.print_exc()
    failures.append(exc)

print(f"blocked={app.net_stats()[0]} allowed={app.net_stats()[1]}")
sys.exit(1 if failures else 0)
