# 02. Technical Architecture Document: Servant AI

**Version:** MVP 0.1  |  Companion to `01_Product_Requirements_Document.md`. Existing baseline code: `app.py` (single-file Streamlit app).

## 1. Architecture principles
1. **One integration point for models:** the OpenAI-compatible HTTP API. Servant AI never imports a model library or assumes a model name.
2. **Everything local:** every network destination is loopback or a private address. Enforced in code and at the OS/Docker level.
3. **Graceful with weak models:** protocol is plain-text JSON, not native function calling; parse failures fall back to a normal answer.
4. **Single file until the demo works;** split into modules afterwards (section 10).

## 2. System overview
```
+--------------------------- User's machine / on-prem server ---------------------------+
|                                                                                       |
|  Browser (localhost:8501)                                                             |
|        |                                                                              |
|  +-----v--------------------- Servant AI app (Streamlit, Python) ----------------+    |
|  |  UI: mode switch | model picker | KB manager | chat + step trace | downloads  |    |
|  |                                                                               |    |
|  |  Agent loop  <-- mode prompt (KB rules | General rules) + tool protocol       |    |
|  |     |                                                                         |    |
|  |     +--> Tools: search_kb | run_python | make_docx                           |    |
|  |     +--> Ingestion: PDF/DOCX/TXT parsers -> OCR/vision -> chunker -> embedder |    |
|  |     +--> KB store: chunks + vectors (workspace/kb.pkl -> later SQLite/Chroma) |    |
|  |     +--> Sovereignty guard: socket.connect wrapper + NETLOG                   |    |
|  +-----------------------------+-------------------------------------------------+    |
|                                | HTTP (loopback/private only)                         |
|  +-----------------------------v-------------------------------------------------+    |
|  |  Local model server (Ollama / llama.cpp / vLLM / LM Studio)                   |    |
|  |  /v1/models   /v1/chat/completions   /v1/embeddings                           |    |
|  |  Chat model | Embedding model | Optional vision model                         |    |
|  +-------------------------------------------------------------------------------+    |
+---------------------------------------------------------------------------------------+
                          NO route to the internet
```

## 3. Components
### 3.1 Model connector
- `OpenAI(base_url=LLM_BASE_URL, api_key="local")`. Default `http://localhost:11434/v1` (Ollama).
- `client.models.list()` populates the model dropdowns. If it fails, the UI shows an error and manual text inputs.
- Three roles chosen independently: **chat**, **embedding**, **vision (optional)**. A natively multimodal chat model (e.g. `qwen3.5:4b`) can be selected for both chat and vision.
- Response post-processing: strip `<think>...</think>`; tolerate an empty `content` with reasoning in a separate field.

### 3.2 Ingestion pipeline
| Input | Handling |
|---|---|
| PDF with text | PyMuPDF `page.get_text()` per page; source label `file.pdf p.N` |
| Scanned PDF page (no text) | Render page to PNG at 150 dpi, then vision model transcription, else OCR fallback |
| DOCX | python-docx paragraphs (and tables, TODO) |
| PNG/JPG screenshot | Vision model transcription prompt, else OCR fallback (Tesseract) |
| TXT/MD | Decode UTF-8 |

Then: chunk (900 chars, 150 overlap) -> embed in batches of 32 -> append to KB.

### 3.3 Knowledge base store
- MVP: `{"chunks": [(source, text)], "vecs": np.ndarray}` pickled to `workspace/kb.pkl`; cosine similarity via NumPy.
- **Must also store** `embed_model` and vector dimension; on mismatch, block queries and prompt re-index.
- Post-MVP: swap for Chroma/Qdrant/LanceDB or SQLite + `sqlite-vec` behind a `KBStore` interface.

### 3.4 Agent loop
```
messages = [system(protocol + mode rules [+ retrieved context in KB mode])] + last 8 turns + user
repeat up to MAX_STEPS (6):
    reply = chat(messages)
    call = parse_first_json_object(reply)     # {"tool": name, "args": {...}}
    if no valid call or tool not allowed in this mode: return reply as final answer
    result = run tool (exceptions become "tool error: ...")
    append assistant reply + "TOOL RESULT (name): ..." and continue
return "Stopped: reached max steps."
```
Tools per mode:
| Tool | General | KB | Notes |
|---|---|---|---|
| `run_python(code)` | yes | **no** | 20 s timeout, working dir `workspace/`; production: Docker `--network none` |
| `search_kb(query)` | no | yes | top-k retrieval, returns `[source] text` blocks |
| `make_docx(title, body, filename)` | yes | yes | markdown-ish body to Word; saved in `workspace/out/` |

### 3.4.1 KB mode prompt contract
Retrieval happens **before** the loop (top 8 chunks injected as CONTEXT) and the model may call `search_kb` again. Rules: answer only from context, cite `[file p.N]`, otherwise reply "Not found in the knowledge base."

### 3.4.2 General mode prompt contract
Answer from own knowledge; when writing code, run it with `run_python` first, then explain the code and how to run it in a terminal.

### 3.5 Deliverable generation
`make_docx` parses `#/##/###` headings, `-`/`*` bullets, and paragraphs. To add: tables (pipe syntax), bold, a signature/approval block template, header/footer. Optional PDF: `soffice --headless --convert-to pdf` (offline).

### 3.6 Sovereignty guard
- `install_guard()` (cached with `st.cache_resource`) wraps `socket.socket.connect`: destinations that are not loopback/private raise `ConnectionError` and are logged as BLOCKED; all others logged ALLOWED.
- **Known limit:** covers this Python process only (not subprocesses, not other processes). Real enforcement is at OS/Docker level (section 6).

## 4. Data & storage layout
```
servant_ai/
  app.py
  Dockerfile, docker-compose.yml, README.md
  docs/                     # these five documents
  workspace/                # created at runtime, mounted as a Docker volume
    kb.pkl                  # knowledge base (chunks, vectors, embed model name)
    out/                    # generated .docx files
    run.py                  # last executed script
    netlog.jsonl            # (to add) persistent connection log
```

## 5. Key design decisions
| Decision | Why |
|---|---|
| OpenAI-compatible API only | Works with every local server; new models need zero code changes |
| JSON-in-prompt tool protocol | Works with models lacking native tool calling; degrades to plain text |
| Retrieval before loop in KB mode | Small models are unreliable at deciding when to retrieve |
| Streamlit UI | Fastest path for one night; pure Python; no CDN assets (check!) |
| NumPy KB | Zero extra dependencies for MVP |

## 6. Deployment architecture
- **Native:** `pip install streamlit openai numpy pymupdf python-docx` then `streamlit run app.py`.
- **Docker Compose:** the app container only; model server on host reached through `LLM_BASE_URL=http://host.docker.internal:11434/v1`. Host Ollama on Linux needs `OLLAMA_HOST=0.0.0.0`.
- **Strict air-gap (production):** put app + model server on a compose network with `internal: true`, plus host firewall DROP+LOG for container egress. Model weights are provided by the user; never bundled or downloaded at runtime.
- Pre-stage everything (wheels, images, weights) on a connected machine, then transfer.

## 7. Configuration
| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Local OpenAI-compatible endpoint |
| `HF_HUB_OFFLINE` | `1` (recommended) | Prevent hub calls |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Disable Streamlit telemetry |

## 8. Error handling
- Model server down: banner + manual model inputs; no crash.
- Malformed tool JSON: treated as final answer.
- Tool exception: returned to the model as `tool error: ...` so it can retry.
- Embedding dimension mismatch: re-index prompt (to add).
- Long docs: chunk cap warnings (to add).

## 9. Testing strategy
1. Smoke: app boots, model list appears, chat answers.
2. KB: index demo report; 5 answerable + 2 unanswerable questions; check citations.
3. General: recursion script run; script with a deliberate error is fixed by the agent.
4. Vision: screenshot of a table; scanned page.
5. Sovereignty: tcpdump silent, unplug test, negative test (`requests.get("https://example.com")` blocked in-process; blocked by network off for subprocess).

## 10. Future modularization
`app.py` -> `connector.py`, `ingest.py`, `kb.py`, `agent.py`, `tools.py`, `guard.py`, `ui.py`. Add a `router.py` (rule-based first, LLM classifier later) and a `models.yaml` registry once multiple models are used concurrently.
