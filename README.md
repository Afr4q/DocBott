<<<<<<< HEAD
# DocBott
Main Project 
=======
# DocBott — Document-Based Intelligent System (LLM + RAG)

An end-to-end intelligent document system that allows users to upload multiple PDFs, extract text (digital + OCR + tables), chunk and index content using hybrid retrieval, ask questions with direct answers grounded in documents, generate AI summaries, view exact sources/pages/confidence/reasoning, use role-based answers, and compare answers across multiple documents — all through a React frontend.

---

## Architecture

```
Frontend (React + Vite + TailwindCSS)
    │
    ▼
FastAPI Backend (REST API)
    │
    ├─ Auth Module (JWT + bcrypt + RBAC)
    ├─ Ingestion (upload, validate, deduplicate)
    ├─ Extraction (PyMuPDF digital text)
    ├─ OCR (PaddleOCR + Tesseract fallback)
    ├─ Tables (Camelot + regex fallback)
    ├─ Pipeline (orchestration, merge, save)
    ├─ Chunking (sentence-aware, overlap)
    ├─ Indexing (SentenceTransformers → ChromaDB/FAISS)
    ├─ Retrieval (hybrid: BM25 + vector + reranking)
    ├─ RAG Engine (role-based extractive answers)
    ├─ AI Summarization (BART / optional Ollama)
    ├─ Memory (conversation history)
    ├─ Cache (in-memory TTL / optional Redis)
    ├─ Evaluation (BLEU, ROUGE, groundedness)
    └─ Feedback (ratings, analytics)
```

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| Database | SQLAlchemy + SQLite (default) |
| Auth | JWT (python-jose) + bcrypt |
| PDF Extraction | PyMuPDF (fitz) |
| OCR | PaddleOCR + Tesseract (fallback) |
| Table Extraction | Camelot + regex fallback |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (default) / FAISS |
| Keyword Search | rank-bm25 (BM25Okapi) |
| AI Summarization | HuggingFace BART-large-CNN |
| Caching | In-memory (TTL) / Redis (optional) |

### Frontend
| Component | Technology |
|---|---|
| Framework | React 18 + Vite |
| Styling | TailwindCSS |
| Routing | react-router-dom v6 |
| HTTP Client | Axios |
| File Upload | react-dropzone |
| Markdown | react-markdown |
| Icons | lucide-react |
| Notifications | react-hot-toast |

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Tesseract OCR installed (optional, for fallback OCR)
- Poppler (for pdf2image)

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment variables
copy .env.example .env
# Edit .env as needed

# Run the server
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server (proxies API to :8000)
npm run dev
```

The frontend runs at `http://localhost:5173` and proxies `/api` requests to the backend.

---

## Project Structure

```
DocBott/
├── backend/
│   ├── app/
│   │   ├── config.py          # Centralized configuration
│   │   ├── database/          # SQLAlchemy models & session
│   │   ├── auth/              # JWT auth, RBAC
│   │   ├── utils/             # Shared utilities
│   │   ├── ingestion/         # PDF upload & validation
│   │   ├── extraction/        # Digital text extraction
│   │   ├── ocr/               # OCR pipeline
│   │   ├── tables/            # Table extraction
│   │   ├── pipeline/          # Processing orchestrator
│   │   ├── chunking/          # Text chunking
│   │   ├── indexing/          # Embeddings & vector store
│   │   ├── retrieval/         # Hybrid search (BM25+vector)
│   │   ├── rag/               # RAG engine (role-based)
│   │   ├── ai/                # AI summarization
│   │   ├── memory/            # Conversation history
│   │   ├── cache/             # Query caching
│   │   ├── evaluation/        # Answer quality metrics
│   │   ├── feedback/          # User feedback
│   │   ├── main.py            # FastAPI app entry point
│   │   └── routes.py          # API endpoints
│   ├── tests/                 # Pytest test suite
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/        # Navbar
│   │   ├── context/           # AuthContext
│   │   ├── pages/             # Login, Dashboard, Upload, Chat, Summary
│   │   ├── services/          # API client (Axios)
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
└── README.md
```

---

## API Endpoints

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login (returns JWT) |
| GET | `/api/auth/me` | Get current user profile |

### Documents
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/documents/upload` | Upload PDF (auto-processes) |
| GET | `/api/documents/` | List all documents |
| GET | `/api/documents/{id}` | Document details |
| DELETE | `/api/documents/{id}` | Delete document |

### Chat
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat/query` | Ask a question (RAG + AI summary) |
| POST | `/api/chat/compare` | Compare across documents |
| GET | `/api/chat/sessions` | List chat sessions |
| GET | `/api/chat/sessions/{id}/history` | Session message history |
| DELETE | `/api/chat/sessions/{id}` | Delete session |

### Feedback
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/feedback/` | Submit feedback |
| GET | `/api/feedback/stats` | Feedback analytics (admin/teacher) |

---

## Key Features

- **Hybrid Retrieval**: Combines BM25 keyword search with vector similarity (weighted fusion)
- **Role-Based Answers**: Student (simplified), Teacher (key points + page refs), Researcher (inline citations)
- **Explainability**: Every answer includes sources, page numbers, confidence scores, and reasoning
- **AI Summarization**: BART-based local summarization with optional Ollama/LLaMA 3
- **Multi-Document Comparison**: Compare answers across selected documents side-by-side
- **OCR Pipeline**: PaddleOCR primary with Tesseract fallback for scanned documents
- **Table Extraction**: Camelot (lattice + stream) with regex fallback
- **Conversation Memory**: Persistent chat sessions with history
- **Caching**: In-memory TTL cache with optional Redis backend
- **Answer Evaluation**: BLEU, ROUGE, groundedness, and source-overlap metrics

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## Environment Variables

See `.env.example` for all available configuration options including:
- JWT secret & algorithm
- OCR engine selection
- Embedding model name
- Chunk size & overlap
- Retrieval weights
- AI summarization settings
- CORS origins
- Cache TTL

---

## License

This project is for educational/research purposes.
>>>>>>> 0664db2 (First Commit)
