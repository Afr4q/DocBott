"""
Chunking module - Smart document chunking for retrieval.
Splits processed documents into overlapping chunks suitable for
embedding and semantic search.

Strategy: sentence-aware chunking with configurable size and overlap.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.utils import split_into_sentences, clean_text, remove_header_footer, clean_chunk_for_display, get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id: str
    content: str
    document_id: int
    page_number: int
    chunk_index: int
    source_type: str = "text"
    char_count: int = 0
    metadata: Dict = field(default_factory=dict)


def chunk_text(
    text: str,
    document_id: int,
    page_number: int,
    source_type: str = "text",
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Split text into overlapping chunks using sentence boundaries.

    Instead of blindly splitting at character boundaries, this function:
    1. Splits text into sentences
    2. Groups sentences into chunks of ~chunk_size chars
    3. Adds overlap from the previous chunk for context continuity

    Args:
        text: The text to chunk.
        document_id: ID of the source document.
        page_number: Page number of the source.
        source_type: Type of source ("text", "ocr", "table").
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Number of overlapping characters.

    Returns:
        List of Chunk objects.
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    # Clean text before splitting into chunks
    text = clean_text(text)
    text = remove_header_footer(text)
    text = clean_chunk_for_display(text)

    if not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_length = 0
    overlap_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + len(sentence) > chunk_size and current_sentences:
            chunk_text_content = " ".join(current_sentences)
            chunk_id = f"doc{document_id}_p{page_number}_c{len(chunks)}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=chunk_text_content,
                document_id=document_id,
                page_number=page_number,
                chunk_index=len(chunks),
                source_type=source_type,
                char_count=len(chunk_text_content),
                metadata={
                    "page": page_number,
                    "source_type": source_type,
                    "chunk_index": len(chunks),
                }
            ))

            # Calculate overlap: keep last few sentences for context
            overlap_chars = 0
            overlap_sentences = []
            for s in reversed(current_sentences):
                if overlap_chars + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_chars += len(s)

            current_sentences = list(overlap_sentences)
            current_length = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_length += len(sentence)

    # Don't forget the last chunk
    if current_sentences:
        chunk_text_content = " ".join(current_sentences)
        chunk_id = f"doc{document_id}_p{page_number}_c{len(chunks)}"

        chunks.append(Chunk(
            chunk_id=chunk_id,
            content=chunk_text_content,
            document_id=document_id,
            page_number=page_number,
            chunk_index=len(chunks),
            source_type=source_type,
            char_count=len(chunk_text_content),
            metadata={
                "page": page_number,
                "source_type": source_type,
                "chunk_index": len(chunks),
            }
        ))

    logger.debug(f"Created {len(chunks)} chunks from page {page_number}")
    return chunks


def chunk_document(
    pages: List[Dict],
    document_id: int,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Chunk]:
    """
    Chunk an entire processed document.

    Args:
        pages: List of page dicts with 'page_number', 'merged_text', 'source_types'.
        document_id: ID of the document.
        chunk_size: Max chunk size.
        chunk_overlap: Overlap size.

    Returns:
        List of all Chunk objects across all pages.
    """
    all_chunks = []

    for page_data in pages:
        page_num = page_data.get("page_number", 0)
        text = page_data.get("merged_text", "")
        source_types = page_data.get("source_types", ["text"])
        source_type = source_types[0] if source_types else "text"

        if not text.strip():
            continue

        page_chunks = chunk_text(
            text=text,
            document_id=document_id,
            page_number=page_num,
            source_type=source_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    # Re-index all chunks sequentially
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i
        chunk.chunk_id = f"doc{document_id}_c{i}"

    logger.info(f"Document {document_id}: {len(all_chunks)} total chunks")
    return all_chunks


def chunk_table(
    table_text: str,
    document_id: int,
    page_number: int,
) -> List[Chunk]:
    """
    Chunk table content separately - tables should generally stay intact.
    Only split if very large tables exceed chunk_size * 2.
    """
    if len(table_text) <= CHUNK_SIZE * 2:
        # Keep table as single chunk
        chunk_id = f"doc{document_id}_p{page_number}_table"
        return [Chunk(
            chunk_id=chunk_id,
            content=table_text,
            document_id=document_id,
            page_number=page_number,
            chunk_index=0,
            source_type="table",
            char_count=len(table_text),
            metadata={"page": page_number, "source_type": "table"},
        )]

    # Split large tables by rows
    return chunk_text(
        text=table_text,
        document_id=document_id,
        page_number=page_number,
        source_type="table",
    )
