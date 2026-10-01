# Servant AI

A self-hosted, air-gapped AI workbench for site engineers and safety officers. It answers
strictly from documents you index, or works as a general assistant that writes and **actually
runs** code and produces real Word files — all against a local model server, with every
outbound connection logged.

Two modes:

- **Knowledge Base mode** answers only from your uploaded documents, cites the page, and
  replies exactly `Not found in the knowledge base.` when the answer is not there.
- **General mode** writes code, executes it in a bounded subprocess, and shows you the real
  stdout before explaining it.

---

## 0. Install on another machine by cloning this repository

Nothing here needs to be copied by hand. Git and GitHub handle it.

### Step 1 — get the code

```bash
git clone https://github.com/BilalSiddiqui-exe/Servent-Ai.git
cd Servant-Ai
```

Private repository? Then authenticate first, or use your token instead of your password:

```bash
git clone https://<your-github-user>:<token>@github.com/BilalSiddiqui-exe/Servent-Ai.git
```

Download a snapshot without Git installed: **Code → Download ZIP**, then unzip and `cd` into
the `Servent-Ai` folder. A ZIP works for running the app, but not for `git pull` updates.

### Step 2 — check you have the tools

```bash
python --version      # needs 3.10 or newer
git --version
```

If Python is missing: <https://www.python.org/downloads/> (tick *Add python.exe to PATH*).

### Step 3 — install the Python dependencies

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Without a virtualenv, just run `pip install -r requirements.txt` directly.

### Step 4 — install and start a local model server

Skip this if you already run Ollama, LM Studio, llama.cpp or vLLM on that machine.

```bash
# Install from https://ollama.com, then:
ollama pull qwen3.5:4b        # the chat model
ollama pull nomic-embed-text  # the embedding model; KB mode needs this
ollama serve                  # leave running in its own terminal
```

Verify it answers before going further:

```bash
curl http://127.0.0.1:11434/api/tags
```

### Step 5 — install Tesseract (only for scanned pages and images)

Skip this step if you only index text PDFs, DOCX, TXT or MD.

- **Windows**: install from <https://github.com/UB-Mannheim/tesseract/wiki>, then set the
  environment variable to the full path of `tesseract.exe`:

  ```powershell
  [Environment]::SetEnvironmentVariable("TESSERACT_CMD", "C:\Program Files\Tesseract-OCR\tesseract.exe", "User")
  ```

  Restart the terminal afterwards so it takes effect.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

### Step 6 — run it

```bash
streamlit run app.py
```

Open <http://127.0.0.1:8501>, then press **Test connection** in the sidebar.

### Docker instead of steps 3 to 5

```bash
docker compose up --build
```

The model server stays on the host, so `ollama serve` must already be running. On Linux you
can additionally apply the host firewall before starting:

```bash
sudo ./deploy/firewall.sh apply
```

### What is deliberately not in the repository

`workspace/` is git-ignored, so a fresh clone starts empty and nothing private is transferred:

| Path | Holds | Get it by |
|---|---|---|
| `workspace/kb.pkl` | your indexed knowledge base | index your own documents in the app |
| `workspace/chats/` | stored conversations | they are created as you chat |
| `workspace/out/` | generated `.docx` and `.pdf` | they appear as you create them |
| `workspace/netlog.jsonl` | connection log | it is written as the app runs |

There is no `.env` and no API key: the app talks only to a local model server and is
hard-coded to `api_key="local"`.

### Keeping it up to date

```bash
git pull
```

If you have edited `app.py` locally, `git pull` will refuse to overwrite your changes. Stash
them first with `git stash`, pull, then restore with `git stash pop`.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Developed on 3.13 |
| A local OpenAI-compatible server | [Ollama](https://ollama.com) is the reference; llama.cpp, vLLM and LM Studio also work |
| A chat model | e.g. `qwen3.5:4b` (natively multimodal) |
| An embedding model | e.g. `nomic-embed-text` — KB mode cannot work without one |
| Tesseract OCR | Optional. Only needed for **scanned** pages and images |

```bash
pip install -r requirements.txt

ollama pull qwen3.5:4b
ollama pull nomic-embed-text
ollama serve
```

Tesseract (skip if you do not need scanned documents):

- **Windows**: install from <https://github.com/UB-Mannheim/tesseract/wiki>, then set
  `TESSERACT_CMD` to the full path of `tesseract.exe`.
- **Linux**: `sudo apt install tesseract-ocr`

---

## 2. Run it (native)

```bash
streamlit run app.py
```

Open <http://127.0.0.1:8501>. The server binds to loopback only.

Then, in the sidebar:

1. Press **Test connection**.
2. Pick the **Chat model** and **Embedding model** from the dropdowns.
3. Upload files, press **Index files**.

Point it at a different server with the `LLM_BASE_URL` environment variable:

```bash
LLM_BASE_URL=http://127.0.0.1:8080/v1 streamlit run app.py
```

### Docker

```bash
docker compose up --build
```

Compose binds to `127.0.0.1:8501`, keeps the workspace in a named volume, and sets
`LLM_BASE_URL` for you. To reach Ollama from inside the container, Ollama must listen on all
interfaces — on Linux start it with `OLLAMA_HOST=0.0.0.0 ollama serve`, and on Windows or
macOS the `host.docker.internal` entry in `docker-compose.yml` already resolves.

> **Verification status:** the Docker files here have been checked statically
> (`python tests/test_docker.py`, 34/34 packaging assertions) but **not** built or run,
> because the dev machine has the Docker CLI without a running daemon. Treat
> `docker compose up --build` as untested and please report anything that breaks.

---

## 3. Switching models

Nothing in the app is model-specific. In the sidebar, choose any chat model and any embedding
model your server exposes.

- **Changing the embedding model** invalidates the stored vectors. The app detects this,
  blocks retrieval, and offers **Re-index with the selected embedding model**. This is
  deliberate: silently mixing vector spaces produces confidently wrong answers.
- **Thinking models** (`qwen3`, `deepseek-r1`, …) think by default, which is very slow on
  CPU. The **Disable model thinking (faster)** toggle is on by default and sends
  `reasoning_effort="none"`. If your server rejects that parameter the app notices once and
  falls back automatically, then tells you in the sidebar.
- Measured on a 4-core CPU with `qwen3.5:4b`: turning thinking off moved a typical answer
  from **43–135 s down to 3–55 s** with no loss of tool-calling accuracy.

---

## 4. Demo script (5–8 minutes)

Index everything in `demo_data/` first. It is entirely fictional.

1. **Grounded answer with a citation.** KB mode: *"What remaining thickness was measured on
   stage 7 of column DC-2?"* → answers with a page citation.
2. **Refusal.** *"What is the crude distillation throughput in barrels per day?"* → exactly
   `Not found in the knowledge base.`
3. **Scanned page.** Index `inspection_report_scanned.png` and ask about the HIGH findings.
   Tesseract reads it locally; no vision model needed.
4. **Real code execution.** General mode: *"Write a Python script that prints the sum of the
   first 10 squares, run it, and tell me the output."* → the step trace shows the code and
   the real `exit=0` / `stdout:` before the explanation.
5. **Word deliverable.** *"Create a Word document called toolbox_talk.docx with the title
   'Toolbox Talk' and three bullet points about lockout."* → a real `.docx` you can download.
6. **Sovereignty.** Press **Run sovereignty self-test**, or open `workspace/netlog.jsonl`.
7. **Whole-document summary.** KB mode: *"Summarise the whole inspection report"* → every
   chunk is compressed in map passes, then reduced into one summary that keeps page
   citations. The step trace shows `map pass 1/3`, `2/3`… so it is visible that nothing was
   skipped.
8. **Multi-chat and PDF.** Use **New chat** in the sidebar to start a second conversation
   and switch between them; chats survive a browser refresh. **Export this chat as PDF**
   downloads the conversation, and every answer offers its own **Download this answer as
   PDF**. The PDF has selectable text, not a screenshot.
9. **Screenshots without indexing.** Drop an image on *"Drop or paste a screenshot here"*,
   press **Ask about these files**, and it is read locally with OCR. The prompt chips
   underneath the answer area send a ready-made question in one click.

---

## 4a. Host firewall (Linux, optional but recommended)

The compose network is pinned to `172.30.0.0/24` precisely so a host firewall can name an
exact source range. `deploy/firewall.sh` generates and applies an nftables ruleset that
allows only the model server and drops **and logs** everything else leaving that network,
including DNS.

```bash
./deploy/firewall.sh print     # show the rules, change nothing - safe on any machine
sudo ./deploy/firewall.sh apply
sudo ./deploy/firewall.sh status   # rules plus drop counters
sudo ./deploy/firewall.sh verify   # confirms the model path still answers
sudo ./deploy/firewall.sh remove
```

The ruleset is produced by `app.host_firewall_rules()` rather than being hand-written in the
script, so it is asserted in `tests/test_features.py` and reviewed before it is loaded. The
script deletes only its own table and never runs `flush ruleset`, so it cannot disturb other
firewall rules on the host. On Windows there is no nftables, so `print` and the unit tests
are the verification available there.

---

## 5. Proving it is air-gapped

Do not take the badge on trust. Any of these is enough:

1. **In-app.** The sidebar counter and `workspace/netlog.jsonl` log every connection attempt
   as `ALLOWED` or `BLOCKED`. A fresh start shows only `127.0.0.1:11434` / `::1:11434`.
2. **Turn the network off.** Disable Wi-Fi and Ethernet, or pull the cable, and run the whole
   demo again. Everything still works.
3. **Watch from outside the app.** While the app runs:

   ```bash
   # Linux
   sudo tcpdump -i any 'not host 127.0.0.1 and not net 10.0.0.0/8'
   # Windows (PowerShell, run as admin)
   pktmon filter add -p 8080      # or use Wireshark
   ```

   Only loopback traffic appears.

### Honest limits

Stating these plainly matters more than a clean-sounding claim:

- The in-process guard wraps `socket.socket.connect` for **the Python process hosting the
  app**. It is real enforcement for the app, not a claim of impossible construction.
- **Code run by General mode executes in a separate subprocess that the in-process guard does
  not cover.** So that this is not just a caveat, `run_python` can delegate to a throwaway
  container instead. Set `SERVANT_RUNNER_IMAGE=python:3.13-slim` and the script runs with:

  ```
  docker run --rm --network none --memory 512m --cpus 1 --pids-limit 100 \
      --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 \
      -v <workspace>:/work:ro -w /work <image> python /work/run.py
  ```

  Every argument is built by the app; none of them come from the model, and there is no shell
  in the command, so generated code cannot influence how it is launched. This is **off by
  default** because it needs `/var/run/docker.sock` mounted into the app container, which is
  itself a large privilege — a reasonable trade for a demo, a poor one for production.
- Blocking is by destination: non-loopback, non-private addresses are refused. A machine on
  a private LAN could still reach private-range hosts; the network flag in the log records
  what was allowed and why.
- There is **no authentication and no per-user access control.** Do not expose this port
  beyond a trusted machine.
- The connector speaks to a local server only. It never contacts a hosted API, and no API
  keys are read or stored.

---

## 6. What is enforced in code, not just asked in the prompt

Small models are unreliable at following rules about their own honesty, so these are checked
structurally:

- **No fake file claims.** If an answer names a `.docx` that no successful `make_docx` call
  produced, the claim is corrected visibly instead of being passed off as done.
- **Code is really executed.** Fenced Python in an answer is run automatically, once per
  answer, under a 20 s timeout, and the output is shown in the step trace.
- **Nothing hangs.** Every model call is streamed and cut off at a hard deadline; the whole
  turn has a wall-clock budget. A slow model returns a partial answer and says so.
- **No leaked tool JSON.** A tool call truncated mid-JSON produces a clear message, not raw
  JSON in the answer.
- **Injection defence.** KB context is declared untrusted data, KB mode has no code
  execution, and the refusal string is matched exactly.
- **Traversal-safe writes.** `make_docx` only ever writes inside `workspace/out/`.

---

## 7. Layout

```
app.py                 the whole application
Dockerfile             image: non-root, telemetry off, tesseract, healthcheck
docker-compose.yml     loopback port, pinned subnet, workspace volume, unprivileged
deploy/firewall.sh     L4 host firewall: print, apply, status, verify, remove
requirements.txt       pinned lower bounds
.streamlit/            loopback binding, telemetry off, local theme
demo_data/             fictional report, SOP, scanned page, injection probe
tools/                 regenerates demo_data
tests/                 test suites (see below)
.github/               CI: builds and checks the Docker image on Linux
workspace/             runtime: kb.pkl, chats/, out/, netlog.jsonl, run.py
files/                 original requirement documents
```

Your indexed documents, chats, produced `.docx`/`.pdf` files and the connection log all live
in `workspace/`. Delete that folder to reset.

---

## 8. Tests

```bash
python tests/test_sla.py       # guard, connector, parser, OCR, timeouts
python tests/test_kb.py        # ingestion, retrieval, tools, docx
python tests/test_agent.py     # live end-to-end model behaviour
python tests/test_stream.py    # streaming and deadline behaviour
python tests/test_timeout.py   # wall-clock guarantees, needs no model server
python -m unittest tests.test_features   # summarisation, PDF, chat store, L4 rules
python tests/test_docker.py    # Docker packaging contract (static)
python -m pyflakes app.py      # must be clean
```

`tests/test_timeout.py` starts a deliberately misbehaving local HTTP server and proves the two
guards that stop a slow model from hanging the page:

- a server that accepts the request and then says nothing is cut off by the transport read
  timeout, which httpx enforces inside the socket;
- a server that trickles one event every couple of seconds is cut off by the deadline check
  between chunks, which the trickle would otherwise keep resetting.

Neither needs a model server, so it runs anywhere in about 20 seconds.

`tests/test_browser.py` drives the real UI in Chrome and verifies all five flows: the sidebar,
a grounded answer, the exact refusal, real code execution plus a Word download, and
multi-chat with prompt chips, a screenshot drop zone and a PDF export:

```bash
pip install playwright
python tests/test_browser.py
```

`tests/test_docker.py` reports 34/35 on a machine with no Docker daemon; the single failure
is the daemon check itself, not the packaging contract. To verify the image on Linux without
installing Docker locally, push the repository to GitHub: `.github/workflows/docker-build.yml`
builds it on `ubuntu-latest`, checks it runs as a non-root user, and imports cleanly. It never
publishes the image anywhere. `workspace/` is git-ignored, so no indexed document, stored chat
or generated file is ever uploaded.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Cannot reach a local model server` | `ollama serve` is not running, or the endpoint is wrong. Press **Test connection**. |
| `embed-model-changed`, retrieval blocked | You changed the embedding model. Re-index. |
| Answers take minutes | Thinking is on. Toggle **Disable model thinking**. |
| Scanned page yields no text | Tesseract not found. Set `TESSERACT_CMD`, or enable the vision model in the sidebar. |
| `Answer cut short` / `budget ran out` | The machine is too slow for the question. Narrow it, or use a smaller model. |
| The model saved the wrong filename | The answer now says so explicitly. Ask again naming the exact file. |
| Answer says it cannot see a screenshot you indexed | Fixed in this version. The screenshot is read by OCR and put in the context; if it still happens, re-index the file. |
| Chat list is gone after a restart | Chats live in `workspace/chats/chats.json`. Deleting `workspace/` clears them by design. |
| Static assets 500 in the browser on Windows | A known Streamlit issue when the user profile path contains a space. Cosmetic; the app works. |
