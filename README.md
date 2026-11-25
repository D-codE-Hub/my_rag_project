# Local RAG Chatbot (ChromaDB + Sentence Transformers + LM Studio)

A simple, fully **local** Retrieval-Augmented Generation (RAG) project. Put your own `.txt` or `.pdf` files in a folder, index them into a vector database, then ask questions in your terminal. A local LLM answers **only from your documents** and lists the files it used.

No cloud API keys are needed. Everything runs on your own machine.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [How this project works](#2-how-this-project-works)
3. [Project structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Step 1 – Get the project](#step-1--get-the-project)
6. [Step 2 – Create a virtual environment](#step-2--create-a-virtual-environment)
7. [Step 3 – Install dependencies](#step-3--install-dependencies)
8. [Step 4 – (Optional) Test the embedding model](#step-4--optional-test-the-embedding-model)
9. [Step 5 – Add your documents](#step-5--add-your-documents)
10. [Step 6 – Ingest documents into ChromaDB](#step-6--ingest-documents-into-chromadb)
11. [Step 7 – Set up LM Studio (local LLM)](#step-7--set-up-lm-studio-local-llm)
12. [Step 8 – Start chatting](#step-8--start-chatting)
13. [Configuration & customization](#configuration--customization)
14. [Troubleshooting](#troubleshooting)
15. [Next steps / ideas](#next-steps--ideas)

---

## 1. What is RAG?

**Retrieval-Augmented Generation** combines two things:

- **Retrieval** – find the pieces of your documents most relevant to a question.
- **Generation** – give those pieces to an LLM so it can write an answer grounded in them.

This lets an LLM answer questions about *your* private data (HR policies, product lists, manuals…) without retraining it, and reduces made-up answers.

---

## 2. How this project works

```
                 ┌──────────────── INGESTION (ingest.py) ────────────────┐
 documents/      │                                                       │
 *.txt, *.pdf ──►│ read file ─► split into chunks ─► embed each chunk ─► │──► chroma_db/
                 │            (500 chars, 100 overlap)  (MiniLM-L6-v2)   │    (vector store)
                 └───────────────────────────────────────────────────────┘

                 ┌──────────────── QUERY (chat.py) ──────────────────────┐
 Your question ─►│ embed question ─► find top 3 similar chunks in Chroma │
                 │        ─► build prompt (context + question)           │
                 │        ─► send to local LLM in LM Studio ─► answer    │
                 └───────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              ANSWER + SOURCES (file names)
```

| Component | Tool used | Purpose |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Turns text into 384-dimension vectors |
| Vector database | ChromaDB (persistent, stored in `./chroma_db`) | Stores chunks + vectors, does similarity search |
| PDF reader | PyMuPDF (`fitz`) | Extracts text from PDFs |
| LLM | Any model in **LM Studio** (default: Qwen 2.5) | Generates the final answer |
| LLM client | `openai` Python package | Talks to LM Studio's OpenAI-compatible API |

---

## 3. Project structure

```
my_rag_project/
├── documents/              # Your knowledge base (put .txt / .pdf files here)
│   ├── company_details.txt #   sample data
│   ├── hr_policy.txt       #   sample data
│   └── products.txt        #   sample data
├── chroma_db/              # Vector database (auto-created by ingest.py)
├── ingest.py               # Reads, chunks, embeds and stores documents
├── chat.py                 # Interactive question-answer loop
├── test_embedding.py       # Small demo to verify embeddings work
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 4. Prerequisites

- **Python 3.9 – 3.12** (check with `python3 --version`)
- **pip**
- **LM Studio** – free desktop app for running LLMs locally: <https://lmstudio.ai>
- ~8 GB RAM minimum (16 GB recommended for a 7B model)
- Internet connection **once**, to download the embedding model and the LLM

---

## Step 1 – Get the project

```bash
git clone <your-repo-url> my_rag_project
cd my_rag_project
```

(Or simply download/copy the folder and `cd` into it.)

---

## Step 2 – Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your system.

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`. Run the activate command again every time you open a new terminal.

---

## Step 3 – Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- `sentence-transformers` – embedding model (also installs PyTorch)
- `chromadb` – vector database
- `openai` – client used to call LM Studio
- `pymupdf` – PDF text extraction

> The first install can take a few minutes because PyTorch is large.

---

## Step 4 – (Optional) Test the embedding model

```bash
python test_embedding.py
```

The first run downloads the `all-MiniLM-L6-v2` model (~90 MB). Expected output:

```
Number of sentences: 4
Vector dimensions: (4, 384)

Sentence: ERPNext is an ERP system
Vector: [-0.07 0.01 ...]
...
```

If you see `(4, 384)`, embeddings are working.

---

## Step 5 – Add your documents

Put your files into the `documents/` folder.

- Supported formats: **`.txt`** and **`.pdf`** (other files are skipped).
- The project ships with three sample files about a fictional company ("DSNK Technologies") so you can try it immediately.
- To use your own data, add your files and (optionally) delete the samples.

> PDFs must contain real text. Scanned image-only PDFs will produce no text (they need OCR first).

---

## Step 6 – Ingest documents into ChromaDB

```bash
python ingest.py
```

What happens:

1. Every `.txt` / `.pdf` in `documents/` is read.
2. Text is split into **500-character chunks with 100-character overlap**, so sentences on chunk borders aren't lost.
3. Each chunk is converted into a vector with the embedding model.
4. Chunks, vectors and metadata (`source` filename, `chunk` number) are saved to the `company_documents` collection in `./chroma_db`.

Example output:

```
Processing: hr_policy.txt
Stored chunks: 5
Processing: products.txt
Stored chunks: 4
Processing: company_details.txt
Stored chunks: 3

Document ingestion completed!
```

**Re-running is safe.** IDs are a hash of `filename:chunk_number`, and `upsert` is used, so running again updates chunks instead of duplicating them.

> ⚠️ If you **delete** a document or make it **shorter**, its old chunks stay in the database. To fully reset, delete the database and ingest again:
> ```bash
> rm -rf chroma_db
> python ingest.py
> ```

---

## Step 7 – Set up LM Studio (local LLM)

1. Download and install **LM Studio** from <https://lmstudio.ai>.
2. Open it, go to the **Search / Discover** tab, and download a chat model.
   - Default in this project: **Qwen 2.5 VL 7B** (`qwen/qwen2.5-vl-7b`).
   - Any instruct model works, e.g. `qwen2.5-7b-instruct`, `llama-3.1-8b-instruct`, `mistral-7b-instruct`. Smaller models (1.5B–3B) are faster on low-RAM machines.
3. Go to the **Developer** (or **Local Server**) tab.
4. Load your model and click **Start Server**. It should say it is running on `http://localhost:1234`.
5. Copy the exact **model identifier** shown in LM Studio.
6. Open `chat.py` and set it:

   ```python
   MODEL_NAME = "qwen/qwen2.5-vl-7b"   # <- replace with your model ID
   ```

Quick check that the server is up:

```bash
curl http://localhost:1234/v1/models
```

You should see JSON listing your loaded model.

---

## Step 8 – Start chatting

Make sure the venv is active, documents are ingested and LM Studio's server is running, then:

```bash
python chat.py
```

Example session (using the sample documents):

```
Ask a question: type 'exit' or 'quit' to stop
> How many sick leaves do employees get?

ANSWER:
Employees get 10 days of sick leave per year. A medical certificate is
needed for more than 2 consecutive days.

SOURCES:
- hr_policy.txt
- hr_policy.txt
- company_details.txt
```

Type `exit` or `quit` to stop.

More questions to try:

- "What is the price of DSNK SmartHR?"
- "Who is the CEO and where is the headquarters?"
- "How many days can I work from home?"
- "What is the capital of France?" → the bot should say it doesn't know (not in the documents).

### What chat.py does on each question

1. Embeds your question with the same embedding model used during ingestion.
2. Queries ChromaDB for the **3 most similar chunks** (`n_results=3`).
3. Builds a prompt: *"Answer using only the context… if not available, say you don't know"* + the chunks (with their source file) + your question.
4. Sends it to the LLM with `temperature=0.2` (low = more factual, less creative).
5. Prints the answer and the source file of each retrieved chunk.

---

## Configuration & customization

| What | Where | Default | Notes |
|---|---|---|---|
| Documents folder | `ingest.py` → `folder` | `./documents` | |
| Chunk size / overlap | `ingest.py` → `split_text()` | `500` / `100` chars | Bigger chunks = more context per hit; smaller = more precise |
| Embedding model | both `ingest.py` and `chat.py` | `all-MiniLM-L6-v2` | **Must be the same in both files.** Re-ingest after changing |
| DB path | both files → `PersistentClient(path=...)` | `./chroma_db` | |
| Collection name | both files | `company_documents` | |
| Number of chunks retrieved | `chat.py` → `n_results` | `3` | Increase for broader questions |
| LLM server URL | `chat.py` → `base_url` | `http://localhost:1234/v1` | Any OpenAI-compatible server works (Ollama: `http://localhost:11434/v1`) |
| LLM model | `chat.py` → `MODEL_NAME` | `qwen/qwen2.5-vl-7b` | Must match the loaded model's ID |
| Temperature | `chat.py` | `0.2` | |
| System prompt / instructions | `chat.py` → `messages` / `prompt` | company assistant | Change it to fit your domain |

---

## Troubleshooting

| Problem | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'chromadb'` (or similar) | Virtual env not active or deps not installed. Run `source venv/bin/activate` then `pip install -r requirements.txt`. |
| `Collection company_documents does not exist` when running `chat.py` | You haven't ingested yet. Run `python ingest.py` first. |
| `openai.APIConnectionError: Connection error` | LM Studio server isn't running, or wrong port. Start the server in LM Studio and check `curl http://localhost:1234/v1/models`. |
| Error about model not found / 404 | `MODEL_NAME` in `chat.py` doesn't match the model loaded in LM Studio. Copy the exact ID. |
| Answers are "I don't know" for things that are in the docs | Re-run `python ingest.py`; try increasing `n_results`; rephrase the question; make sure the PDF has extractable text. |
| Old/deleted content still shows up | Delete `chroma_db/` and re-ingest. |
| Very slow answers | Use a smaller model in LM Studio, or enable GPU offload in LM Studio settings. |
| Embedding model download fails | Needs internet on first run. Check your connection/proxy, then retry. |

---

## Next steps / ideas

- Support more formats (`.docx`, `.md`, `.csv`).
- Split by sentences/paragraphs instead of fixed character counts.
- Show the retrieved chunk text (not just file names) for debugging.
- Keep chat history so follow-up questions work.
- Add a web UI with Streamlit or Gradio.
- Add a `.gitignore` for `venv/`, `chroma_db/` and `.DS_Store`.
