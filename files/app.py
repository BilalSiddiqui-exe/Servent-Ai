"""
Servant AI - MVP (single file)
Sovereign, model-agnostic, agentic workbench.

Talks to ANY local OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio).
Run:  streamlit run app.py
"""
import base64, io, ipaddress, json, os, pathlib, pickle, re, socket, subprocess, sys, time
import numpy as np
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------- config
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # any local OpenAI-compatible server
WORK = pathlib.Path("workspace"); WORK.mkdir(exist_ok=True)
OUT = WORK / "out"; OUT.mkdir(exist_ok=True)
KB_PATH = WORK / "kb.pkl"

st.set_page_config(page_title="Servant AI", layout="wide")


# ------------------------------------------------- sovereignty guard
@st.cache_resource
def install_guard():
    """Block + log every outbound socket connection that is not loopback/private network."""
    log, orig = [], socket.socket.connect

    def guarded(self, address):
        if isinstance(address, tuple) and len(address) >= 2:
            host = str(address[0])
            try:
                ip = ipaddress.ip_address(host)
                ok = ip.is_loopback or ip.is_private
            except ValueError:
                ok = host == "localhost"
            log.append((time.strftime("%H:%M:%S"), f"{host}:{address[1]}", "ALLOWED" if ok else "BLOCKED"))
            if not ok:
                raise ConnectionError(f"[Servant AI sovereignty guard] blocked outbound connection to {host}")
        return orig(self, address)

    socket.socket.connect = guarded
    return log


NETLOG = install_guard()
client = OpenAI(base_url=BASE_URL, api_key="local")


# ---------------------------------------------------------- sidebar
def list_models():
    try:
        return sorted(m.id for m in client.models.list().data)
    except Exception:
        return []


st.sidebar.title("Servant AI")
mode = st.sidebar.radio("Mode", ["General", "Knowledge Base"],
                        help="Knowledge Base: answers ONLY from your uploaded files. General: free assistant + tools.")
models = list_models()
if models:
    CHAT_MODEL = st.sidebar.selectbox("Chat model", models)
    EMBED_MODEL = st.sidebar.selectbox("Embedding model", models,
                                       index=next((i for i, m in enumerate(models) if "embed" in m or "bge" in m or "nomic" in m), 0))
    VISION_MODEL = st.sidebar.selectbox("Vision model (optional, for screenshots/scans)", ["(none)"] + models)
    VISION_MODEL = "" if VISION_MODEL == "(none)" else VISION_MODEL
else:
    st.sidebar.error(f"No local model server found at {BASE_URL}. Start Ollama / llama.cpp / vLLM first.")
    CHAT_MODEL = st.sidebar.text_input("Chat model", "qwen3:8b")
    EMBED_MODEL = st.sidebar.text_input("Embedding model", "nomic-embed-text")
    VISION_MODEL = st.sidebar.text_input("Vision model (optional)", "")

st.sidebar.caption(f"Endpoint: {BASE_URL}")


# --------------------------------------------------------- knowledge base
if "kb" not in st.session_state:
    st.session_state.kb = pickle.loads(KB_PATH.read_bytes()) if KB_PATH.exists() else {"chunks": [], "vecs": None}
if "files" not in st.session_state:
    st.session_state.files = []
if "history" not in st.session_state:
    st.session_state.history = []


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def vision_ocr(img_bytes, ext="png"):
    if not VISION_MODEL:
        return ""
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    b64 = base64.b64encode(img_bytes).decode()
    r = client.chat.completions.create(model=VISION_MODEL, messages=[{"role": "user", "content": [
        {"type": "text", "text": "Transcribe all text in this image exactly. Then briefly describe any tables, diagrams, stamps or handwriting."},
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}]}])
    return r.choices[0].message.content


def extract(name, data):
    """Return list of (source_label, text)."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        import fitz  # PyMuPDF
        out = []
        for i, p in enumerate(fitz.open(stream=data, filetype="pdf")):
            txt = p.get_text().strip()
            if not txt and VISION_MODEL:  # scanned page -> vision model
                txt = vision_ocr(p.get_pixmap(dpi=150).tobytes("png"), "png")
            out.append((f"{name} p.{i + 1}", txt))
        return out
    if ext == "docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        return [(name, "\n".join(p.text for p in d.paragraphs))]
    if ext in ("png", "jpg", "jpeg"):
        return [(name, vision_ocr(data, ext))]
    return [(name, data.decode("utf-8", "ignore"))]


def chunk(text, size=900, overlap=150):
    return [text[i:i + size] for i in range(0, max(len(text), 1), size - overlap) if text[i:i + size].strip()]


def embed(texts):
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in r.data], dtype="float32")


def index_files(files):
    kb = st.session_state.kb
    new = [(src, c) for f in files for src, txt in extract(f.name, f.getvalue()) for c in chunk(txt)]
    if not new:
        return 0
    vecs = np.vstack([embed([t for _, t in new[i:i + 32]]) for i in range(0, len(new), 32)])
    kb["chunks"] += new
    kb["vecs"] = vecs if kb["vecs"] is None else np.vstack([kb["vecs"], vecs])
    KB_PATH.write_bytes(pickle.dumps(kb))
    return len(new)


def search_kb(q, k=6):
    kb = st.session_state.kb
    if not kb["chunks"]:
        return []
    qv = embed([q])[0]
    sims = kb["vecs"] @ qv / (np.linalg.norm(kb["vecs"], axis=1) * np.linalg.norm(qv) + 1e-9)
    return [kb["chunks"][i] for i in np.argsort(-sims)[:k]]


with st.sidebar.expander("Knowledge base", expanded=(mode == "Knowledge Base")):
    ups = st.file_uploader("PDF / DOCX / TXT / screenshots", accept_multiple_files=True,
                           type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg"])
    if st.button("Index files") and ups:
        with st.spinner("Indexing locally..."):
            st.success(f"Indexed {index_files(ups)} chunks")
    st.caption(f"{len(st.session_state.kb['chunks'])} chunks in KB")
    if st.button("Clear KB"):
        st.session_state.kb = {"chunks": [], "vecs": None}
        KB_PATH.unlink(missing_ok=True)

with st.sidebar.expander("Network monitor", expanded=True):
    blocked = sum(1 for e in NETLOG if e[2] == "BLOCKED")
    st.metric("External connections blocked", blocked)
    st.caption(f"{len(NETLOG) - blocked} local connections allowed")
    for t, dest, verdict in NETLOG[-8:][::-1]:
        st.text(f"{t} {verdict} {dest}")


# ------------------------------------------------------------------ tools
def tool_run_python(code):
    """MVP: subprocess with timeout. For production, run in Docker with --network none."""
    f = WORK / "run.py"
    f.write_text(code)
    try:
        p = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=20, cwd=WORK)
        return f"exit={p.returncode}\nstdout:\n{p.stdout[-3000:]}\nstderr:\n{p.stderr[-2000:]}"
    except subprocess.TimeoutExpired:
        return "error: timed out after 20s"


def tool_make_docx(title, body, filename="output.docx"):
    import docx
    d = docx.Document()
    d.add_heading(title, 0)
    for line in body.splitlines():
        s = line.rstrip()
        if s.startswith("### "): d.add_heading(s[4:], 3)
        elif s.startswith("## "): d.add_heading(s[3:], 2)
        elif s.startswith("# "): d.add_heading(s[2:], 1)
        elif s.lstrip().startswith(("- ", "* ")): d.add_paragraph(s.lstrip()[2:], style="List Bullet")
        elif s.strip(): d.add_paragraph(s)
    path = OUT / pathlib.Path(filename).name
    d.save(path)
    if str(path) not in st.session_state.files:
        st.session_state.files.append(str(path))
    return f"saved {path.name}"


def tool_search_kb(query):
    hits = search_kb(query)
    return "\n\n".join(f"[{s}] {t}" for s, t in hits) or "no results"


TOOLS = {
    "General": {"run_python": (tool_run_python, "code"), "make_docx": (tool_make_docx, "title, body, filename")},
    "Knowledge Base": {"search_kb": (tool_search_kb, "query"), "make_docx": (tool_make_docx, "title, body, filename")},
}

PROTOCOL = """You are Servant AI, an offline assistant running fully on the user's own machine.
To use a tool, reply with ONLY one JSON object: {"tool": "<name>", "args": {...}}
When you are done, reply with the final answer in plain text/markdown (no JSON).
Available tools:
{tools}
Use tools only when needed. Plan briefly, act, read the tool result, then continue or finish."""

KB_RULES = """MODE: KNOWLEDGE BASE. Answer ONLY from the CONTEXT below. Cite sources like [file p.3].
If the context does not contain the answer, say exactly: "Not found in the knowledge base." Never use outside knowledge.
If asked to produce a document/summary, call make_docx with content drawn only from the context.

CONTEXT:
{ctx}"""

GEN_RULES = """MODE: GENERAL. Answer helpfully from your own knowledge. If you write code, RUN it with run_python to verify it
works before answering, then explain the code and exactly how to run it in a terminal."""


def chat(messages):
    r = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return strip_think(r.choices[0].message.content or "")


def parse_tool(reply):
    i = reply.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(reply[i:])
        return obj if isinstance(obj, dict) and "tool" in obj else None
    except json.JSONDecodeError:
        return None


def run_agent(user_msg, trace, max_steps=6):
    tools = TOOLS[mode]
    system = PROTOCOL.format(tools="\n".join(f"- {n}({a})" for n, (_, a) in tools.items()))
    if mode == "Knowledge Base":
        hits = search_kb(user_msg, k=8)
        ctx = "\n\n".join(f"[{s}] {t}" for s, t in hits) or "(knowledge base is empty)"
        system += "\n\n" + KB_RULES.format(ctx=ctx)
    else:
        system += "\n\n" + GEN_RULES
    msgs = [{"role": "system", "content": system}] + st.session_state.history[-8:] + [{"role": "user", "content": user_msg}]
    for step in range(max_steps):
        reply = chat(msgs)
        call = parse_tool(reply)
        if not call or call["tool"] not in tools:
            return reply
        name, args = call["tool"], call.get("args", {})
        trace.write(f"**Step {step + 1}: `{name}`**")
        if name == "run_python":
            trace.code(args.get("code", ""), language="python")
        try:
            result = tools[name][0](**args)
        except Exception as e:
            result = f"tool error: {e}"
        trace.code(str(result)[:1500])
        msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"TOOL RESULT ({name}):\n{result}"}]
    return "Stopped: reached max steps."


# --------------------------------------------------------------------- UI
st.title("Servant AI")
st.caption(f"Mode: **{mode}**  |  Model: `{CHAT_MODEL}`  |  Air-gapped: external connections are blocked and logged")

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask Servant AI..." if mode == "General" else "Ask about your documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status("Working locally...", expanded=True) as status:
            answer = run_agent(prompt, status)
            status.update(label="Done", state="complete", expanded=False)
        st.markdown(answer)
    st.session_state.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    st.rerun() if False else None

for f in st.session_state.files:
    p = pathlib.Path(f)
    if p.exists():
        st.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name, key=f)
