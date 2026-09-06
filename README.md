# AskTube — YouTube RAG Chatbot

Ask questions about any YouTube video and get grounded, cited answers — powered by Retrieval-Augmented Generation (RAG). Available as a REST API and a Chrome extension that works directly on any YouTube video page.

## How it works

1. **Transcript ingestion** — fetches the video's transcript via the YouTube Transcript API
2. **Semantic chunking** — splits the transcript by *meaning* rather than character count using LangChain's `SemanticChunker` (percentile breakpoints, `breakpoint_threshold_amount=90`, `min_chunk_size=300`). Adjacent sentences are embedded and compared; a new chunk starts wherever the semantic distance spikes, so each chunk maps to a coherent topic segment of the video
3. **Embedding** — encodes chunks with `sentence-transformers/all-MiniLM-L6-v2` and stores them in a per-video Chroma collection
4. **Retrieval** — uses **MMR (Maximal Marginal Relevance)** instead of plain similarity search, since spoken transcripts are repetitive; MMR retrieves relevant *and* diverse chunks (`k=4, lambda_mult=0.65`)
5. **Generation** — a Groq-hosted LLM (`openai/gpt-oss-120b`) answers strictly from retrieved context, citing the timestamp of the source chunk, with an explicit fallback ("I don't know based on the provided transcript") to reduce hallucination
6. **Delivery** — served via a FastAPI backend, containerized with Docker, and consumed by a Chrome extension frontend

## Architecture

```
YouTube video → Chrome Extension (popup.js)
                        ↓ POST /ask {video_id, question}
                 FastAPI (main.py)
                        ↓
              rag_pipeline.py
   ┌─────────────────────────────────────────┐
   │ Transcript fetch → Semantic chunking     │
   │ → Embed → Chroma (per-video)             │
   │ → MMR Retrieve → Prompt → Groq LLM       │
   │ → Timestamped answer                     │
   └─────────────────────────────────────────┘
                        ↓
              schema.py (Pydantic I/O validation)
```

## Chrome Extension

A Manifest V3 extension with a dark, YouTube-native popup UI:

- **Auto-detects** the video ID from the active tab
- **Timestamp chips** — citations like `[4:07]` in answers are rendered as styled, scannable chips
- **Player-style status bar** showing the connected video, with clear error states when not on a video page
- **Polished UX** — empty-state hint, animated loading indicator, reduced-motion support, and full icon set (16/48/128 px)

## Evaluation

Measured using **RAGAS** with LLM-as-a-judge (Groq-hosted, temperature=0, separate from the answer-generation model) across a test set of transcript-grounded questions:

| Metric | Score |
|---|---|
| Faithfulness | 0.97 |
| Answer Relevancy | 0.91 |

**Faithfulness** measures whether generated answers are actually grounded in retrieved context (not hallucinated). **Answer Relevancy** measures whether answers actually address the question asked.

*Limitation: the same model family is used for both generation and judging, which can introduce mild self-evaluation bias. A stronger/independent judge model would be a natural next step for more rigorous evaluation.*

## Tech Stack

- **Orchestration:** LangChain (LCEL / Runnable chains)
- **LLM:** Groq (`openai/gpt-oss-120b`)
- **Chunking:** LangChain Experimental `SemanticChunker` (embedding-based breakpoints)
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** ChromaDB (per-video collections)
- **Evaluation:** RAGAS (faithfulness, answer relevancy)
- **Backend:** FastAPI + Pydantic
- **Deployment:** Docker
- **Frontend:** Chrome Extension (Manifest V3, vanilla JS)

## Project Structure

```
Youtube_chatbot/
├── main.py              # FastAPI app and /ask endpoint
├── rag_pipeline.py      # Reusable RAG chain builder (per video_id)
├── schema.py            # Pydantic input/output schemas
├── evaluate.py          # RAGAS evaluation script
├── Dockerfile
├── requirements.txt     # Production dependencies
├── requirements_dev.txt # Eval-only dependencies (kept out of the Docker image)
├── .dockerignore
└── extension/
    ├── manifest.json
    ├── popup.html
    ├── popup.js
    ├── icon48.png
```

## Setup

### Backend (API)

```bash
git clone https://github.com/sanket012pandey/youtube-rag-chatbot.git
cd youtube-rag-chatbot

pip install -r requirements.txt

# Create a .env file with:
# GROQ_API_KEY=your_key_here
# HUGGINGFACEHUB_API_TOKEN=your_token_here

uvicorn main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`

### Docker

```bash
docker build -t youtube-rag-bot .
docker run -p 8000:8000 --env-file .env youtube-rag-bot
```

### Chrome Extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder
4. Open any YouTube video and click the AskTube icon

### Evaluation

```bash
pip install -r requirements_dev.txt
python evaluate.py
```

## Design Decisions

- **Semantic chunking over fixed-size splitting** — a fixed 1000-char window can cut a spoken explanation mid-thought; `SemanticChunker` places boundaries where the topic actually shifts, adapting chunk count to video length automatically. Percentile mode (`threshold=90`) balances granularity and context, and `min_chunk_size=300` prevents fragmentary chunks from filler speech.
- **MMR over plain similarity search** — spoken transcripts repeat ideas; MMR reduces redundant retrieved chunks and improves context diversity.
- **Per-video Chroma collections** — keeps retrieval scoped to a single video instead of mixing content across videos.
- **Separate `requirements.txt` / `requirements_dev.txt`** — keeps evaluation tooling (RAGAS, datasets) out of the production Docker image.
- **CPU-only PyTorch build** — the embedding model doesn't need GPU acceleration; using the CPU-only wheel keeps the Docker image significantly smaller.

## Author

**Sanket Pandey** — [LinkedIn](https://www.linkedin.com/in/sanket-pandey-821a6634b/) · [GitHub](https://github.com/sanket012pandey)