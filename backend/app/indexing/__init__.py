"""
Indexing module - Manages embeddings and vector store operations.
Supports ChromaDB and FAISS backends.
Uses SentenceTransformers (MiniLM) for embedding generation.
"""

from typing import Dict, List, Optional, Tuple
import os

from app.config import (
    EMBEDDING_MODEL, VECTOR_STORE_TYPE, CHROMA_PERSIST_DIR, EMBEDDINGS_DIR
)
from app.chunking import Chunk
from app.utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Embedding Model (lazy-loaded singleton)
# ──────────────────────────────────────────────
_embedding_model = None


def get_embedding_model():
    """Lazy-load the SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embedding vectors for a list of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


# ──────────────────────────────────────────────
# Vector Store Interface
# ──────────────────────────────────────────────
class VectorStore:
    """Abstract interface for vector storage backends."""

    def add(self, ids: List[str], embeddings: List[List[float]],
            documents: List[str], metadatas: List[Dict]) -> None:
        raise NotImplementedError

    def query(self, embedding: List[float], top_k: int = 5,
              filter_dict: Optional[Dict] = None) -> List[Dict]:
        raise NotImplementedError

    def delete(self, ids: List[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class ChromaStore(VectorStore):
    """ChromaDB vector store backend with persistence."""

    def __init__(self):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
            is_persistent=True,
        ))
        self.collection = self.client.get_or_create_collection(
            name="docbott_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized at {CHROMA_PERSIST_DIR}")

    def add(self, ids: List[str], embeddings: List[List[float]],
            documents: List[str], metadatas: List[Dict]) -> None:
        """Add documents with embeddings to the collection."""
        # ChromaDB handles batching internally, but we chunk for safety
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
        logger.info(f"Added {len(ids)} items to ChromaDB")

    def query(self, embedding: List[float], top_k: int = 5,
              filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Query the vector store for similar documents."""
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_dict:
            kwargs["where"] = filter_dict

        results = self.collection.query(**kwargs)

        # Format results
        formatted = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                })
        return formatted

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} items from ChromaDB")

    def delete_by_document(self, document_id: int) -> None:
        """Delete all chunks belonging to a specific document."""
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted all chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete chunks for doc {document_id}: {e}")

    def count(self) -> int:
        return self.collection.count()


class FAISSStore(VectorStore):
    """FAISS-based vector store with metadata sidecar."""

    def __init__(self):
        import faiss
        import numpy as np
        import json

        self.faiss = faiss
        self.np = np
        self.index = None
        self.metadata_store: Dict[int, Dict] = {}
        self.documents: Dict[int, str] = {}
        self.id_map: Dict[str, int] = {}
        self.next_id = 0

        self.index_path = str(EMBEDDINGS_DIR / "faiss.index")
        self.meta_path = str(EMBEDDINGS_DIR / "faiss_meta.json")

        self._load()

    def _load(self):
        """Load existing FAISS index and metadata if available."""
        try:
            if os.path.exists(self.index_path):
                self.index = self.faiss.read_index(self.index_path)
                with open(self.meta_path, "r") as f:
                    data = __import__("json").load(f)
                    self.metadata_store = {int(k): v for k, v in data.get("meta", {}).items()}
                    self.documents = {int(k): v for k, v in data.get("docs", {}).items()}
                    self.id_map = data.get("id_map", {})
                    self.next_id = data.get("next_id", 0)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")

    def _save(self):
        """Persist FAISS index and metadata to disk."""
        if self.index:
            self.faiss.write_index(self.index, self.index_path)
            import json
            with open(self.meta_path, "w") as f:
                json.dump({
                    "meta": self.metadata_store,
                    "docs": self.documents,
                    "id_map": self.id_map,
                    "next_id": self.next_id,
                }, f)

    def add(self, ids: List[str], embeddings: List[List[float]],
            documents: List[str], metadatas: List[Dict]) -> None:
        vectors = self.np.array(embeddings, dtype="float32")
        dim = vectors.shape[1]

        if self.index is None:
            self.index = self.faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)

        self.index.add(vectors)

        for i, str_id in enumerate(ids):
            idx = self.next_id + i
            self.id_map[str_id] = idx
            self.metadata_store[idx] = metadatas[i]
            self.documents[idx] = documents[i]

        self.next_id += len(ids)
        self._save()
        logger.info(f"Added {len(ids)} vectors to FAISS")

    def query(self, embedding: List[float], top_k: int = 5,
              filter_dict: Optional[Dict] = None) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.np.array([embedding], dtype="float32")
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            meta = self.metadata_store.get(int(idx), {})

            # Apply filter
            if filter_dict:
                if not all(meta.get(k) == v for k, v in filter_dict.items()):
                    continue

            results.append({
                "id": str(idx),
                "content": self.documents.get(int(idx), ""),
                "metadata": meta,
                "score": float(scores[0][i]),
            })

        return results

    def delete(self, ids: List[str]) -> None:
        logger.warning("FAISS delete individual vectors not supported efficiently. Rebuild recommended.")

    def count(self) -> int:
        return self.index.ntotal if self.index else 0


# ──────────────────────────────────────────────
# Store Factory (Singleton)
# ──────────────────────────────────────────────
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        if VECTOR_STORE_TYPE == "faiss":
            _vector_store = FAISSStore()
        else:
            _vector_store = ChromaStore()
    return _vector_store


# ──────────────────────────────────────────────
# High-Level Indexing Operations
# ──────────────────────────────────────────────
def index_chunks(chunks: List[Chunk]) -> int:
    """
    Generate embeddings and index a list of chunks.

    Args:
        chunks: List of Chunk objects to index.

    Returns:
        Number of chunks indexed.
    """
    if not chunks:
        return 0

    store = get_vector_store()

    # Prepare data
    ids = [c.chunk_id for c in chunks]
    texts = [c.content for c in chunks]
    metadatas = [
        {
            "document_id": c.document_id,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "source_type": c.source_type,
            "char_count": c.char_count,
        }
        for c in chunks
    ]

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = generate_embeddings(texts)

    # Store
    store.add(ids, embeddings, texts, metadatas)

    logger.info(f"Indexed {len(chunks)} chunks. Total in store: {store.count()}")
    return len(chunks)


def search_vectors(query: str, top_k: int = 5,
                   filter_dict: Optional[Dict] = None) -> List[Dict]:
    """
    Search the vector store for chunks similar to the query.

    Args:
        query: The search query text.
        top_k: Number of results to return.
        filter_dict: Optional metadata filters.

    Returns:
        List of result dicts with content, metadata, and score.
    """
    store = get_vector_store()
    query_embedding = generate_embeddings([query])[0]
    return store.query(query_embedding, top_k=top_k, filter_dict=filter_dict)
