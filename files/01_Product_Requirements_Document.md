# 01. Product Requirements Document (PRD): Servant AI

**Version:** MVP 0.1  |  **Status:** Draft for implementation  |  **Audience:** OpenCode (coding agent), developers, judges

## 1. Problem
Refineries, PSUs, defence-linked units and government offices do a large amount of routine but confidential knowledge work: approval notes, review of inspection reports and scanned drawings, engineering calculations, code for internal tools, and answering questions from manuals and SOPs. Company policy forbids sending this data to cloud AI (Claude, ChatGPT, Codex). As a result people either work manually and slowly, or quietly paste confidential data into public tools, which is a security breach.

## 2. Product vision
**Servant AI** is a self-hosted, air-gapped AI workbench that runs entirely on the organization's own machine. Nothing leaves the premises. It behaves like a private Claude: it reads the organization's own documents, plans multi-step work, uses local tools, and produces real deliverables. It is **model-agnostic**: any open-source LLM the user's hardware can run can be plugged in.

## 3. Target users
| User | Need |
|---|---|
| Engineer / officer at a PSU or refinery | Summarize reports, draft approval notes, ask questions of SOPs, write and verify small scripts, all without data leaving the site |
| IT / security admin | Deploy on-prem, prove no external traffic, choose which models are allowed |
| Hackathon judge / inspector | Install on their own machine, connect their own LLM, verify the sovereignty claim |

## 4. MVP scope
**In scope**
- Inputs: PDF, DOCX, TXT/MD, and image screenshots (PNG/JPG). Scanned PDF pages are treated as images.
- Output: Word (`.docx`) documents. PDF export is a stretch goal.
- Two modes: **Knowledge Base (KB) mode** and **General mode**.
- Agentic behavior: plan, call tools, observe results, iterate (bounded steps).
- Model-agnostic connector to any local OpenAI-compatible server (Ollama, llama.cpp, vLLM, LM Studio).
- Sovereignty proof: live network monitor, blocked-connection log, unplug test.
- Deployment: run natively with Python or via Docker Compose.

**Out of scope for MVP (planned for the full product)**
- Excel and PPT generation, image generation/output
- Automatic multi-model routing by task type (manual model selection in MVP; see Risk R3)
- Multi-user login and role-based access, audit dashboards
- Fine-tuning, cloud hosting, mobile apps

## 5. The two modes
### 5.1 Knowledge Base mode (grounded)
- Answers **only** from the user's indexed files. Every claim carries a citation such as `[report.pdf p.3]`.
- If the knowledge base does not contain the answer, the reply is exactly: "Not found in the knowledge base."
- Can generate deliverables from KB content: summaries, approval notes, findings lists, as `.docx`.
- Never uses the model's outside knowledge and never runs code.

### 5.2 General mode
- Behaves like a general assistant using the model's own knowledge.
- Can write code, **run it in a local sandbox to verify**, then explain it and explain how to run it in a terminal.
- Example: "Give me a Python script explaining recursion, run it, and tell me how to run it in the terminal."
- Can also generate `.docx` documents.

## 6. Functional requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Connect to a local OpenAI-compatible endpoint; list available models; user selects chat, embedding and optional vision model | P0 |
| FR-2 | Upload and index PDF, DOCX, TXT, MD, PNG, JPG into a persistent local KB | P0 |
| FR-3 | Extract text from images and scanned pages using the selected vision model, or an OCR fallback if none | P0 |
| FR-4 | KB mode: retrieval-augmented, cited, refuses when not found | P0 |
| FR-5 | General mode: free answers, code generation, code execution with observed output | P0 |
| FR-6 | Agent loop with tools (`search_kb`, `run_python`, `make_docx`), max-step cap, visible step trace | P0 |
| FR-7 | Generate downloadable `.docx` (headings, bullets, paragraphs, tables) | P0 |
| FR-8 | Network monitor showing every connection attempt and blocked external attempts | P0 |
| FR-9 | Detect embedding-model change and require re-indexing | P1 |
| FR-10 | Whole-document summarization (map-reduce) for long files | P1 |
| FR-11 | Manage KB: list, delete, re-index documents | P1 |
| FR-12 | Rule-based router: image attached goes to the vision model, else chat model | P2 |
| FR-13 | PDF export of generated documents | P2 |
| FR-14 | Sandboxed code execution in Docker with `--network none` | P2 (production must) |

## 7. Non-functional requirements
- **Sovereignty:** zero external network calls in every code path. No telemetry, no hosted APIs, no CDN fonts or scripts in the UI.
- **Hardware-agnostic:** no code or config assumes a specific GPU or model. Works with a 4B model on a laptop or a 120B model on a server.
- **Reliability with small models:** tool calling must degrade gracefully when the model emits malformed JSON.
- **Setup time:** a judge can run it in under 10 minutes given Ollama and one model.
- **Transparency:** every agent step and tool result is visible to the user.

## 8. Success metrics (demo day)
1. KB mode answers 5 of 5 questions from a demo report with correct citations, and correctly says "Not found" on 2 out-of-scope questions.
2. General mode writes, runs and explains the recursion script end to end.
3. A screenshot of a table or inspection page is read and used in an answer.
4. A `.docx` approval note is generated from a scanned or PDF report.
5. Network monitor shows 0 external connections; the demo also works with Wi-Fi/Ethernet turned off.
6. A judge switches to a different local model and it works with no code change.

## 9. Assumptions
- The host machine runs a local model server that exposes `/v1/chat/completions`, `/v1/embeddings` and `/v1/models`.
- The reference dev model is `qwen3.5:4b` via Ollama, which is natively multimodal. An embedding model such as `nomic-embed-text` is also pulled.

## 10. Risks
| ID | Risk | Mitigation |
|---|---|---|
| R1 | Small models produce bad tool-call JSON | Tolerant parser, plain-text fallback, low temperature, retry once |
| R2 | Thinking-mode models are slow and emit reasoning text | Strip `<think>` blocks, allow disabling thinking, show progress |
| R3 | Original problem statement expects automatic model selection | Add a simple rule-based router (FR-12) so the demo shows selection across two task types |
| R4 | Subprocess code execution is not a true sandbox | Demo with network off; Docker `--network none` sandbox is a required post-MVP ticket |
| R5 | Prompt injection through uploaded documents | KB mode has no code execution; documents are treated as data (see Security doc) |

## 11. Open questions
- Which submission portal fields are required (repo link, video, slides)?
- Will judges bring their own hardware or use the team's laptop?
