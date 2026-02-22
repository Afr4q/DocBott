"""
Table extraction module - Extract tables from PDFs using Camelot.
Falls back to regex/heuristic extraction when Camelot fails.
Converts tables to structured formats (text, markdown, dicts).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import re

from app.utils import clean_text, get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedTable:
    """Represents a single extracted table."""
    page_number: int
    table_index: int
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    accuracy: float = 0.0
    markdown: str = ""
    raw_text: str = ""

    def to_text(self) -> str:
        """Convert table to plain text representation."""
        if self.markdown:
            return self.markdown
        lines = []
        if self.headers:
            lines.append(" | ".join(self.headers))
            lines.append("-" * len(lines[0]))
        for row in self.rows:
            lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(lines)


def extract_tables_camelot(file_path: str, pages: str = "all") -> List[ExtractedTable]:
    """
    Extract tables from a PDF using Camelot.
    Tries 'lattice' mode first (bordered tables), then 'stream' (borderless).

    Args:
        file_path: Path to the PDF file.
        pages: Page specification (e.g., "1,3,5" or "all").

    Returns:
        List of ExtractedTable objects.
    """
    tables = []

    try:
        import camelot

        # Try lattice mode first (for tables with borders)
        try:
            result = camelot.read_pdf(file_path, pages=pages, flavor="lattice")
            for i, table in enumerate(result):
                tables.append(_camelot_to_extracted(table, i))
        except Exception as e:
            logger.debug(f"Lattice extraction failed: {e}")

        # If no tables found, try stream mode (borderless tables)
        if not tables:
            try:
                result = camelot.read_pdf(file_path, pages=pages, flavor="stream")
                for i, table in enumerate(result):
                    tables.append(_camelot_to_extracted(table, i))
            except Exception as e:
                logger.debug(f"Stream extraction also failed: {e}")

        logger.info(f"Extracted {len(tables)} tables from {file_path}")

    except ImportError:
        logger.warning("Camelot not installed. Table extraction disabled.")
    except Exception as e:
        logger.error(f"Table extraction failed for {file_path}: {e}")

    return tables


def _camelot_to_extracted(table, index: int) -> ExtractedTable:
    """Convert a Camelot table object to our ExtractedTable format."""
    import pandas as pd

    df = table.df
    page = table.page if hasattr(table, "page") else 0
    accuracy = table.accuracy if hasattr(table, "accuracy") else 0.0

    # Use first row as headers if it looks like headers
    headers = [str(cell).strip() for cell in df.iloc[0]]
    rows = []
    for _, row in df.iloc[1:].iterrows():
        row_data = [str(cell).strip() for cell in row]
        # Filter out completely empty rows
        if any(cell for cell in row_data):
            rows.append(row_data)

    # Generate markdown representation
    markdown = _table_to_markdown(headers, rows)

    return ExtractedTable(
        page_number=page,
        table_index=index,
        headers=headers,
        rows=rows,
        accuracy=accuracy,
        markdown=markdown,
        raw_text=markdown,
    )


def _table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Convert table data to Markdown format."""
    if not headers and not rows:
        return ""

    lines = []
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        # Pad row if shorter than headers
        padded = row + [""] * (len(headers) - len(row)) if len(row) < len(headers) else row
        lines.append("| " + " | ".join(padded[:len(headers)]) + " |")

    return "\n".join(lines)


def clean_table_text(table: ExtractedTable) -> ExtractedTable:
    """
    Clean table content: remove noisy rows, empty cells, OCR artifacts.
    """
    cleaned_rows = []
    for row in table.rows:
        # Skip rows that are mostly empty
        non_empty = [cell for cell in row if cell.strip()]
        if len(non_empty) < len(row) * 0.3:
            continue

        # Clean individual cells
        cleaned_row = [clean_text(cell) for cell in row]
        cleaned_rows.append(cleaned_row)

    table.rows = cleaned_rows
    table.markdown = _table_to_markdown(table.headers, cleaned_rows)
    table.raw_text = table.markdown
    return table


def extract_tables_from_text(text: str, page_number: int = 0) -> List[ExtractedTable]:
    """
    Fallback: extract tabular data from plain text using regex patterns.
    Detects rows with consistent column separators.
    """
    tables = []
    lines = text.split("\n")

    # Look for groups of lines with consistent separators
    current_table_lines = []
    for line in lines:
        # Check if line looks tabular (multiple spaces or tabs as separators)
        columns = re.split(r"\s{2,}|\t", line.strip())
        if len(columns) >= 3:
            current_table_lines.append(columns)
        else:
            if len(current_table_lines) >= 3:
                # We found a table-like block
                headers = current_table_lines[0]
                rows = current_table_lines[1:]
                tables.append(ExtractedTable(
                    page_number=page_number,
                    table_index=len(tables),
                    headers=headers,
                    rows=rows,
                    markdown=_table_to_markdown(headers, rows),
                ))
            current_table_lines = []

    # Handle table at end of text
    if len(current_table_lines) >= 3:
        headers = current_table_lines[0]
        rows = current_table_lines[1:]
        tables.append(ExtractedTable(
            page_number=page_number,
            table_index=len(tables),
            headers=headers,
            rows=rows,
            markdown=_table_to_markdown(headers, rows),
        ))

    return tables
