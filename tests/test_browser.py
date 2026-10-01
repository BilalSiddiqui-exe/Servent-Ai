"""Four-flow browser verification, driven through a real browser.

Flow 1  sidebar: model role selection, thinking toggle, KB status, sovereignty monitor
Flow 2  grounded KB answer with a citation
Flow 3  exact refusal for an unanswerable question
Flow 4  General mode: code executed for real, then a Word file downloaded
Plus:    streaming is observed token-by-token in the DOM before the answer settles
"""

import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"D:\Servsnt ai")
SHOTS = ROOT / "workspace" / "shots"
PORT = 8512
URL = f"http://127.0.0.1:{PORT}"
REFUSAL = "Not found in the knowledge base."

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    # Detail is diagnostic, so only show it when something actually went wrong.
    shown = detail if not ok else ""
    print("  %s %s%s" % ("PASS" if ok else "FAIL", name, (" - " + shown) if shown else ""))
    return bool(ok)


def start_app():
    log = open(str(ROOT / "workspace" / "browser_streamlit.log"), "w",
               encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
         "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(90):
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen(URL, timeout=3).read()
            print("app up on %s" % URL)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("streamlit died on startup")
    raise SystemExit("app never became reachable")


def settle_old(page, seconds):
    """Unused legacy helper kept out of the flow."""
    time.sleep(seconds)


def ask(page, question):
    before = page.locator('[data-testid="stChatMessage"]').count()
    box = page.locator('[data-testid="stChatInputTextArea"]').first
    box.click()
    box.fill(question)
    page.keyboard.press("Enter")
    return before + 2  # the user turn plus the assistant turn that must follow


def settle(page, seconds, min_msgs=0):
    """Wait until the answer has actually started and then stopped changing."""
    deadline = time.time() + seconds
    last, stable, started = None, 0, False
    while time.time() < deadline:
        msgs = page.locator('[data-testid="stChatMessage"]')
        count = msgs.count()
        if count >= min_msgs:
            started = True
        if started:
            try:
                current = msgs.last.inner_text()
            except Exception:
                current = None
            stopping = page.get_by_role("button", name="Stop").count() > 0
            if current and current == last and not stopping:
                stable += 1
                if stable >= 4:
                    return
            else:
                stable = 0
            last = current
        time.sleep(1.5)
    time.sleep(2)


def switch_mode(page, label):
    page.locator('label:has-text("%s")' % label).first.click()
    time.sleep(3)


def _trace_excerpt(page, limit=400):
    """Whatever the step trace currently shows, for diagnosing a failed check."""
    try:
        for summary in page.locator("summary").all():
            if "Step trace" in (summary.inner_text() or ""):
                parent = summary.locator("xpath=ancestor::details[1]").first
                return parent.inner_text()[:limit]
    except Exception:
        pass
    return page.inner_text("body")[-limit:]


def reveal_trace(page):
    """Ensure the step-trace accordion is open.

    Clicking an open accordion closes it, which used to hide the trace and make
    the tool checks fail for the wrong reason. Read aria-expanded instead.
    """
    try:
        for summary in page.locator("summary").all():
            text = summary.inner_text() or ""
            if "Step trace" in text or "Working locally" in text:
                if (summary.get_attribute("aria-expanded") or "").lower() == "false":
                    summary.click()
                    time.sleep(1)
                return True
    except Exception:
        pass
    return "calling " in page.inner_text("body")


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    proc = start_app()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            console_errors = []
            page.on("console", lambda m: console_errors.append(m.text)
                    if m.type == "error" else None)
            page.goto(URL, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(9000)
            page.screenshot(path=str(SHOTS / "01_sidebar.png"), full_page=True)

            # ---------------- Flow 1: sidebar controls ----------------
            print("\n== Flow 1: sidebar ==")
            body = page.inner_text("body")
            check("sidebar shows the KB", "Knowledge base" in body)
            check("thinking toggle present and ON by default",
                  page.get_by_text("Disable model thinking").count() > 0)
            toggle = page.get_by_text("Disable model thinking", exact=False).first
            box = toggle.locator("xpath=ancestor::label[1]//input").first
            check("thinking toggle is enabled (checked)", box.is_checked())
            check("role selection present",
                  "Chat model" in body or "chat model" in body.lower())
            check("embedding model shown",
                  "nomic" in body.lower() or "Embedding model" in body)
            check("network monitor present", "Network" in body or "network" in body)

            # index demo data through the real sidebar uploader
            uploads = page.locator('input[type="file"]')
            check("file uploader exists", uploads.count() > 0)
            if uploads.count():
                uploads.first.set_input_files([
                    str(ROOT / "demo_data" / "inspection_report.pdf"),
                    str(ROOT / "demo_data" / "sop_hot_work.pdf"),
                ])
                time.sleep(4)
                page.get_by_role("button", name="Index files").click()
                time.sleep(60)
                page.screenshot(path=str(SHOTS / "02_indexed.png"), full_page=True)
                body = page.inner_text("body")
                check("documents indexed in the sidebar",
                      "inspection_report" in body or "chunks" in body.lower())

            # the chat gate requires a successful connection test first
            page.get_by_role("button", name="Test connection").click()
            time.sleep(6)
            body = page.inner_text("body")
            check("connection test succeeded", "connected" in body.lower()
                  or "Ready" in body, "no connected badge")

            # ---------------- Flow 2: grounded answer ----------------
            print("\n== Flow 2: grounded KB answer ==")
            switch_mode(page, "Knowledge Base")
            page.wait_for_timeout(1500)
            expect = ask(page, "What remaining thickness was measured on stage 7 of column DC-2?")
            seen = False
            # Prefill of a long KB prompt on a CPU-only model can take over a
            # minute before the first token, so poll for up to two minutes.
            for _ in range(240):
                time.sleep(0.5)
                if "▌" in page.inner_text("body"):
                    seen = True
                    break
            streaming_seen = seen
            settle(page, 150, expect)
            body = page.inner_text("body")
            page.screenshot(path=str(SHOTS / "03_answer.png"), full_page=True)
            cites = re.findall(r"inspection_report\.pdf\s*p\.\d+", body)
            check("answer cites a page", bool(cites), cites[0] if cites else "no citation")
            check("trace panel rendered", "calling " in page.inner_text("body")
                  or "step 1" in page.inner_text("body"))
            check("token cursor observed mid-answer (live stream)", streaming_seen,
                  "no cursor seen" if not streaming_seen else "cursor rendered")

            # ---------------- Flow 3: exact refusal ----------------
            print("\n== Flow 3: exact refusal ==")
            expect = ask(page, "What is the crude distillation throughput in barrels per day?")
            settle(page, 150, expect)
            body = page.inner_text("body")
            page.screenshot(path=str(SHOTS / "04_refusal.png"), full_page=True)
            check("refusal is the exact string", REFUSAL in body,
                  REFUSAL if REFUSAL in body else body[-160:].replace("\n", " "))

            # ---------------- Flow 4: General mode + download ----------------
            print("\n== Flow 4: General mode, real execution, Word download ==")
            radio = page.get_by_text("General", exact=True).first
            radio.click()
            time.sleep(3)
            expect = ask(page, "Write a Python script that prints the sum of the first 10 "
                               "squares, run it, and tell me the output.")
            settle(page, 200, expect)
            body = page.inner_text("body")
            page.screenshot(path=str(SHOTS / "05_general.png"), full_page=True)
            check("code was actually executed", reveal_trace(page) and "run_python" in page.inner_text("body"),
                  "trace did not mention run_python; trace excerpt: %r"
                  % _trace_excerpt(page))
            check("real output reported (385)", "385" in body)
            check("how to run present", "How to run" in body or "how to run" in body)

            expect = ask(page, "Create a Word document called toolbox_talk.docx with the "
                               "title 'Toolbox Talk' and three bullet points about lockout.")
            settle(page, 220, expect)
            body = page.inner_text("body")
            page.screenshot(path=str(SHOTS / "06_docx.png"), full_page=True)
            check("docx tool ran", reveal_trace(page) and "make_docx" in page.inner_text("body"))
            check("download button offered", page.get_by_text("Download", exact=False).count() > 0)

            out_dir = ROOT / "workspace" / "out"
            made = sorted(p.name for p in out_dir.glob("*.docx"))
            check("a .docx landed in workspace/out", bool(made), ", ".join(made))
            check("requested filename was honoured or the mismatch is flagged",
                  "toolbox_talk.docx" in made or "You asked for" in page.inner_text("body"),
                  "created=%s" % ", ".join(made))
            target = out_dir / "toolbox_talk.docx"
            if not target.exists() and made:
                target = out_dir / made[-1]
            if target.exists():
                # The answer also offers a PDF button, so pick the .docx one by name.
                word_buttons = page.get_by_role("button", name=re.compile(r"\.docx$"))
                button = word_buttons.last if word_buttons.count() else \
                    page.get_by_text("Download", exact=False).last
                with page.expect_download(timeout=60000) as dl:
                    button.click()
                path = dl.value.path()
                size = Path(path).stat().st_size
                head = Path(path).read_bytes()[:2]
                check("downloaded file is a real .docx",
                      head == b"PK" and size > 3000, "%d bytes, magic=%r" % (size, head))
            else:
                check("downloaded file is a real .docx", False, "nothing to download")

            static_bug = [e for e in console_errors if "500" in e or "static" in e.lower()]
            real_errors = [e for e in console_errors if e not in static_bug]
            check("no app-level console errors", not real_errors,
                  "; ".join(real_errors[:2])[:200])
            if static_bug:
                print("  NOTE %d static-asset 500s (Streamlit/WinError 123, "
                      "profile path has a space) - cosmetic, app is functional"
                      % len(static_bug))

            # ------------- Flow 5: multi-chat, chips, PDF export -------------
            print("\n== Flow 5: new chat, switch back, prompt chip, PDF export ==")
            first_title = page.locator('[data-testid="stChatMessage"]').count()
            check("chat is not empty before switching",
                  first_title > 0, "%d messages" % first_title)

            page.get_by_role("button", name="New chat", exact=True).click()
            settle(page, 40, 0)
            body = page.inner_text("body")
            check("new chat starts empty",
                  page.locator('[data-testid="stChatMessage"]').count() == 0,
                  body[-140:].replace("\n", " "))
            check("new chat is selected", "New chat" in body)

            # the earlier conversation must still be reachable
            chat_box = page.locator('[data-testid="stSelectbox"]').first
            check("chat switcher is in the sidebar", chat_box.count() > 0)
            if chat_box.count():
                chat_box.click()
                time.sleep(1.5)
                options = page.get_by_role("option")
                check("both chats are listed", options.count() >= 2,
                      "%d options" % options.count())
                if options.count():
                    options.first.click()
                    settle(page, 60, 0)
            back = page.locator('[data-testid="stChatMessage"]').count()
            check("switching back restores the earlier turns", back == first_title,
                  "expected %d, got %d" % (first_title, back))

            # a chip sends its own text as the question, with no typing needed
            chip = page.get_by_role("button", name="Draft a toolbox talk", exact=True)
            check("prompt chip is offered", chip.count() > 0)
            if chip.count():
                before_chip = page.locator('[data-testid="stChatMessage"]').count()
                chip.first.click()
                settle(page, 240, before_chip + 2)
                body = page.inner_text("body")
                check("chip sends its text as the question",
                      "Draft a toolbox talk" in body,
                      body[-160:].replace("\n", " "))

            export = page.get_by_role("button", name="Export this chat as PDF", exact=True)
            check("chat PDF export button present", export.count() > 0)
            if export.count():
                export.first.click()
                settle(page, 90, 1)
                pdf_button = page.get_by_role("button", name="Download chat PDF", exact=True)
                check("chat PDF download offered", pdf_button.count() > 0)
                if pdf_button.count():
                    with page.expect_download(timeout=60000) as dl:
                        pdf_button.first.click()
                    data = Path(dl.value.path()).read_bytes()
                    check("downloaded chat PDF is a real PDF",
                          data.startswith(b"%PDF") and len(data) > 800,
                          "%d bytes, magic=%r" % (len(data), data[:4]))
                    try:
                        import pymupdf
                        doc = pymupdf.open(stream=data, filetype="pdf")
                        text = "".join(p.get_text() for p in doc)
                        check("PDF text is selectable and carries the conversation",
                              "Answer" in text and len(text) > 100,
                              "%d chars" % len(text))
                        doc.close()
                    except Exception as exc:
                        check("PDF text is selectable and carries the conversation", False,
                              f"{type(exc).__name__}: {exc}")

            # attachment drop zone for screenshots
            check("screenshot drop zone offered",
                  "Drop or paste a screenshot" in page.inner_text("body"))

            exceptions = page.locator('[data-testid="stException"]')
            check("no app exceptions after the new controls",
                  exceptions.count() == 0,
                  exceptions.nth(0).inner_text()[:200] if exceptions.count() else "")
            page.screenshot(path=str(SHOTS / "07_chats.png"), full_page=True)

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        logpath = ROOT / "workspace" / "browser_streamlit.log"
        if logpath.exists():
            errs = [ln for ln in logpath.read_text(encoding="utf-8",
                                                   errors="replace").splitlines()
                    if "Error" in ln or "Traceback" in ln or "line " in ln]
            if errs:
                print("\n=== app traceback (last 25 lines) ===")
                for ln in errs[-25:]:
                    print("  " + ln)

    passed = sum(1 for _, ok, _ in results if ok)
    print("\n%d/%d browser checks passed" % (passed, len(results)))
    for name, ok, detail in results:
        if not ok:
            print("  FAILED: %s %s" % (name, detail))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
