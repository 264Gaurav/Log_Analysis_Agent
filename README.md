# BAT.AI - Log Analysis with Self-Corrective RAG

BAT.AI (Bug Automation Tool) is a command-line log-analysis assistant. Give it a text log and a question; it retrieves the most relevant log sections, reranks them, and asks an NVIDIA-hosted LLM to produce a structured, actionable answer.

## What it uses

- **LangGraph** to orchestrate retrieval, reranking, grading, response generation, and query rewriting.
- **Hybrid retrieval**: BM25 keyword search plus FAISS vector similarity search.
- **NVIDIA AI Endpoints** for embeddings (`nvidia/nv-embedqa-e5-v5`) and generation (`meta/llama-3.1-70b-instruct`).
- **FlashRank** (`ms-marco-MiniLM-L-12-v2`) as a local cross-encoder reranker. The model may be downloaded on its first run.
- Prompts in [prompt.json](prompt.json), tailored for summaries, error details, and debugging recommendations.

![BAT.AI architecture](<BAT.AI SW Architecture Diagram.drawio.png>)

## Prerequisites

- Python 3.12 (the project is currently set up with this version).
- An NVIDIA API key with access to the models above. See the [NVIDIA API-key guide](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#generate-an-api-key).
- Internet access on the first run for NVIDIA endpoints and, if not already cached, the FlashRank model.

## Setup

From the repository root, create and activate a virtual environment:

```powershell
uv venv --python 3.12 venv
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

`pip` can be used instead of `uv`:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```dotenv
API_KEY=your_nvidia_api_key
```

`.env` is ignored by Git. Do not commit an API key.

## Run an analysis

```powershell
python example.py data/log-data.txt --question "What are the critical errors in the log file?"
```

The CLI accepts:

```text
python example.py <log_path> [--question "your question"]
```

If `--question` is omitted, BAT.AI uses: `Analyze the log file and find the failure messages from the same`.

The final answer is printed after the graph has completed. Input files must be plain-text logs readable by LangChain's `TextLoader`.

## Observability

Each invocation writes to a timestamped file under `app_logs/` and to the console. Every record includes a request-scoped `run_id`, stage name, event, and (where applicable) elapsed time. Run lifecycle records include input file size and a short SHA-256 fingerprint, while question and rewritten-query fingerprints are logged without recording their contents.

Useful events include `run_started`, `input_valid` or `input_invalid`, stage `start`/`complete`/`failed`, graph node updates, routing decisions, document counts, response length, and `run_completed` or `run_failed`. Set `LOG_LEVEL=DEBUG` or another standard Python logging level to change verbosity.

Retrieval diagnostics also record the configured `RETRIEVAL_K`, BM25 and FAISS candidates, native component scores, chunk fingerprints and sizes, weighted RRF contributions, and the final fused ranking. The defaults are `RETRIEVAL_K=4`, `FAISS_SCORE_THRESHOLD=0.8`, `RRF_RANK_CONSTANT=60`, and equal BM25/FAISS weights. Set `LOG_QUERY_CONTENT=true` to log full questions, or set `LOG_CONTENT_PREVIEW_CHARS` to a positive value to include bounded chunk previews; both are disabled by default.

Every LLM call emits `invoke_start`, `invoke_complete`, or `invoke_failed` with its operation name, elapsed time, input fields, and output. Inputs include the question/query, documents passed to the model, and generated answer when available. Content is fingerprinted by default. For a controlled debugging run, enable bounded payload visibility in PowerShell:

```powershell
$env:LOG_LLM_PAYLOADS="true"
$env:LOG_LLM_PREVIEW_CHARS="1000"
$env:LOG_QUERY_CONTENT="true"
python example.py data/log-data.txt --question "What are the critical errors and warnings in the log file?"
```

The CLI also emits `graph_step` records with the node, question fingerprint, document count, and transform count, making query-rewrite loops and routing decisions easy to follow by `run_id`.

## Workflow

```text
log file + question
        |
        v
retrieve (BM25 + FAISS) -> rerank (FlashRank) -> grade documents
                                                   |
                         +-------------------------+-----------------------+
                         |                                                 |
                         v                                                 v
                  rewrite question                                  generate answer
                         |                                                 |
                         +--------------> retrieve again <----------------+
                                                                           |
                                                                    grade answer quality
```

The prompts instruct the model to base its answer only on supplied log context and organize it around a summary, issues, error details, and recommendations.

## Repository layout

| File | Purpose |
| --- | --- |
| [example.py](example.py) | CLI entry point and `process_input()` helper. |
| [bat_ai.py](bat_ai.py) | Defines and compiles the LangGraph workflow. |
| [graphnodes.py](graphnodes.py) | Retrieval, reranking, grading, generation, and query-rewrite nodes. |
| [graphedges.py](graphedges.py) | Routing decisions between graph nodes. |
| [multiagent.py](multiagent.py) | `HybridRetriever`: loader, chunking, BM25, and FAISS setup. |
| [utils.py](utils.py) | NVIDIA LLM chains and prompt loading. |
| [prompt.json](prompt.json) | QA, rewrite, relevance, hallucination, and answer-quality prompts. |
| [data/log-data.txt](data/log-data.txt) | Sample log file. |

## Current limitation

`Nodes.grade_documents()` currently does not return the filtered documents after grading. As written, that node cannot provide the graph state update the following routing step expects, so an end-to-end CLI run may fail at that point. The README reflects the intended workflow, but the method needs to return a state update such as `{"documents": filtered_docs, "question": question}` before the complete pipeline is operational.

## Troubleshooting

- **Authentication or model-access error:** confirm `API_KEY` is set in `.env`, has no surrounding quotes or placeholder text, and is authorized for the configured NVIDIA models.
- **No useful matches:** use a more specific question with an error code, timestamp, service, or exception name. The workflow can rewrite questions for another retrieval attempt.
- **First run is slow:** FAISS embeddings are created from the supplied log each run, and FlashRank may download its model the first time.
