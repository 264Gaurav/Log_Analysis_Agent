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
