# 04. Frontend Specification: Servant AI

**Version:** MVP 0.1  |  **Framework:** Streamlit (single page). The UI must load **no external assets** (no CDN fonts, scripts or images) to keep the sovereignty claim intact.

## 1. UX principles
1. **Mode is always visible.** The user must never be unsure whether answers come from their documents or from the model's own knowledge.
2. **Show the work.** Agent steps, tool calls and results are visible, not hidden.
3. **Sovereignty is a feature.** The network monitor is always on screen.
4. **Fail visibly and kindly.** If the model server is down or the KB is empty, say exactly what to do.

## 2. Layout
```
+--------------------+---------------------------------------------------------------+
| SIDEBAR            | MAIN                                                          |
|                    |                                                               |
| Servant AI         |  Servant AI                                                   |
| ( ) General        |  Mode: Knowledge Base | Model: qwen3.5:4b | AIR-GAPPED  [0 blocked] |
| (o) Knowledge Base |---------------------------------------------------------------|
|                    |  chat history (user / assistant bubbles)                      |
| Connection         |    assistant bubble contains:                                 |
|  Endpoint [.....]  |      - collapsible "Working locally..." step trace            |
|  [Test connection] |      - final answer (markdown, code blocks, citations)        |
|  Chat model  [v]   |      - Download buttons for generated .docx                   |
|  Embed model [v]   |                                                               |
|  Vision model [v]  |---------------------------------------------------------------|
|                    |  [ Ask about your documents...                         ] [send]|
| Knowledge base     +---------------------------------------------------------------+
|  Upload files      |
|  [Index files]     |
|  N chunks / files  |
|  [Clear KB]        |
|                    |
| Network monitor    |
|  0 blocked         |
|  last 8 attempts   |
+--------------------+
```

## 3. Screens and components
### 3.1 Sidebar
| Component | Behavior |
|---|---|
| Mode radio | `General` / `Knowledge Base`. Switching mode keeps chat history but shows a divider line "Switched to <mode>". Help text explains each mode |
| Endpoint input **(to add)** | Text field prefilled from `LLM_BASE_URL`. Non-private hosts show a red warning |
| Test connection **(to add)** | Calls `/v1/models`; shows green "Connected, N models" or red error with fix hints |
| Chat / Embedding / Vision selectors | Populated from `/v1/models`. Embedding defaults to a model whose name contains `embed`, `bge` or `nomic`. Vision has a "(none)" option and an info note: "Natively multimodal chat models can be chosen here too" |
| KB uploader | Accepts pdf, docx, txt, md, png, jpg, jpeg; multiple files |
| Index files button | Shows progress bar per file, then success toast "Indexed N chunks from M files" |
| KB summary | Document list with chunk counts and a delete icon per document **(to add)**; total chunk count; Clear KB with confirm |
| Re-index warning **(to add)** | If the KB was built with a different embedding model: yellow banner with a "Re-index" button |
| Network monitor | Metric "External connections blocked" (green at 0, red if greater), count of local connections allowed, table of last 8 attempts (time, destination, ALLOWED/BLOCKED) |

### 3.2 Main header
One line: mode badge, chat model name, and text "Air-gapped: external connections are blocked and logged". Mode badge colors: General = blue, KB = green.

### 3.3 Chat area
- Standard chat bubbles (Streamlit `st.chat_message`).
- **Step trace** (`st.status`): "Working locally..." expanded while running, collapses to "Done". Each step: tool name, code block for `run_python` input, output block for results (truncated to 1500 chars with "show more").
- **KB answers:** citations rendered inline as `[file p.N]`. If the reply is "Not found in the knowledge base.", show it in a muted info style.
- **General answers:** code in fenced blocks with copy button; "How to run" steps appear as a numbered list.
- **Empty states:**
  - KB mode, empty KB: "No documents indexed yet. Upload files in the sidebar and click Index files."
  - Server down: "Can't reach a local model server at <endpoint>. Start Ollama (`ollama serve`) and click Test connection."
- Streaming of tokens is desirable **(P2)**; MVP shows a spinner and the final reply.

### 3.4 Downloads
Each generated `.docx` appears as a download button under the reply (label: filename). Files persist in the session list for the run.

### 3.5 Suggested prompt chips (P2)
- General: "Explain recursion with a runnable Python script"
- KB: "Summarize this report in one page", "List all findings with severity", "Draft an approval note"

## 4. Interaction flows
### Flow A: KB question
Select KB mode -> upload PDF -> Index -> ask question -> spinner -> cited answer -> ask "Generate a Word approval note" -> step trace shows `make_docx` -> download button.
### Flow B: General coding task
Select General mode -> ask for recursion script -> step trace shows `run_python` code and output -> final explanation + how to run in terminal.
### Flow C: Screenshot
Upload PNG in sidebar -> Index -> vision transcript indexed -> ask about its contents in KB mode. (Optional P2: drag an image into the chat input.)
### Flow D: Swap the model
Change the chat model in the sidebar -> next message uses it. No restart needed.

## 5. State model (`st.session_state`)
| Key | Type | Purpose |
|---|---|---|
| `kb` | dict | chunks, vecs, embed_model |
| `history` | list | chat turns (role, content) |
| `files` | list | generated file paths |
| `NETLOG` | list (cached resource) | connection attempts |

## 6. Visual design
- Streamlit default light/dark theme; accent color teal. Configure via `.streamlit/config.toml` (`[theme]`), `gatherUsageStats = false`.
- Clean and utilitarian; no animations. Monospace for code and logs.
- Accessibility: sufficient contrast, mode conveyed by text as well as color, all buttons labeled.

## 7. Responsiveness
Desktop-first (laptop at a demo table). Sidebar collapses on narrow screens (native Streamlit behavior).

## 8. Acceptance criteria
1. Mode is visible in the header at all times and matches the sidebar radio.
2. Changing the model in the sidebar changes the model used on the next message.
3. Every tool call in a run appears in the step trace with its result.
4. The network monitor updates after each interaction and shows 0 blocked during the demo.
5. The page makes no requests to non-local hosts (verify in the browser DevTools Network tab).
6. Empty-KB and server-down states show the specified messages.

## 9. Future UI (post-MVP)
Login screen and role-based views; per-department KB switcher; audit log page; model registry admin panel; toggles for Excel/PPT/image outputs; side-by-side source viewer that opens the cited page.
