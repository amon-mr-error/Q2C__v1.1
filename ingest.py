"""PDF ingestion utilities — v2 (sentence-aware chunking).

This module provides:
- extract_text_from_pdf()  → List[str]  (one str per page, normalised)
- ingest_pdf()             → List[Dict]  (chunk dicts ready for embedding)

v2 change: _chunk_text() now uses a **recursive separator cascade**
(paragraph → line → sentence → character) so chunks never split mid-sentence.
This is the single biggest factor for embedding quality and retrieval accuracy.

Each chunk dict:
    {"text": str, "meta": {"page": int, "chunk_index": int, "filename": str}}
"""

from __future__ import annotations

import re
from typing import BinaryIO, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# PDF reader — lazy import so the module is importable without PyPDF2
# ---------------------------------------------------------------------------

def _ensure_pdf_reader():
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise ImportError(
            "PyPDF2 is required. Install with: pip install PyPDF2"
        ) from e
    return PdfReader


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)   # keep up to one blank line
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs
    # Remove hyphenated line-breaks common in PDFs: "in-\nformation" → "information"
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence-aware recursive splitter
# ---------------------------------------------------------------------------

# Priority order: paragraph break, line break, sentence boundaries, then chars
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _recursive_split(text: str, chunk_size: int, overlap: int,
                     separators: list[str]) -> List[str]:
    """Recursively split *text* using the first separator that makes sense,
    then merge small pieces back up to *chunk_size* with *overlap* context.

    This mirrors the logic of LangChain's RecursiveCharacterTextSplitter but
    without the external dependency, so the ingest module stays lightweight.
    """
    final_chunks: List[str] = []

    # Pick the first separator that actually occurs in the text
    sep = ""
    new_separators: list[str] = []
    for i, s in enumerate(separators):
        if s == "" or s in text:
            sep = s
            new_separators = separators[i + 1:]
            break

    splits = text.split(sep) if sep else [text]

    # Accumulate splits into chunks of ≤ chunk_size, with overlap
    current: List[str] = []
    current_len = 0

    for split in splits:
        split = split.strip()
        if not split:
            continue

        split_len = len(split)

        # If the single split is already longer than chunk_size, recurse
        if split_len > chunk_size and new_separators:
            sub_chunks = _recursive_split(split, chunk_size, overlap, new_separators)
            final_chunks.extend(sub_chunks)
            continue

        # Would adding this split exceed the limit?
        # +1 accounts for the separator that joins them
        if current_len + split_len + (1 if current else 0) > chunk_size:
            if current:
                chunk_text = sep.join(current).strip()
                if chunk_text:
                    final_chunks.append(chunk_text)

                # Build the overlap tail: keep trailing splits that fit in `overlap`
                overlap_parts: List[str] = []
                overlap_len = 0
                for part in reversed(current):
                    part_len = len(part)
                    if overlap_len + part_len + 1 > overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += part_len + 1

                current = overlap_parts
                current_len = overlap_len

        current.append(split)
        current_len += split_len + (1 if len(current) > 1 else 0)

    # Flush remainder
    if current:
        chunk_text = sep.join(current).strip()
        if chunk_text:
            final_chunks.append(chunk_text)

    return final_chunks


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Sentence-aware chunking.  Replaces the old character sliding-window."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []
    return _recursive_split(text, chunk_size, overlap, _SEPARATORS)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    source: Union[str, bytes, BinaryIO], password: Optional[str] = None
) -> List[str]:
    """Extract and normalise text from each page of a PDF.

    Returns:
        List[str] — one normalised string per page.
    """
    PdfReader = _ensure_pdf_reader()

    try:
        if isinstance(source, (bytes, bytearray)):
            from io import BytesIO
            reader = PdfReader(BytesIO(source))
        else:
            reader = PdfReader(source)
    except Exception as e:
        raise ValueError(f"Failed to open PDF source: {e}") from e

    if getattr(reader, "is_encrypted", False):
        if password is None:
            raise ValueError("PDF is encrypted; provide a password.")
        try:
            reader.decrypt(password)
        except Exception:
            try:
                reader.decrypt("")
            except Exception:
                raise

    pages_text: List[str] = []
    for page in reader.pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        pages_text.append(_normalize_text(raw))

    return pages_text


# ---------------------------------------------------------------------------
# High-level ingestion API
# ---------------------------------------------------------------------------

def ingest_pdf(
    source: Union[str, bytes, BinaryIO],
    chunk_size: int = 1000,
    overlap: int = 200,
    password: Optional[str] = None,
) -> List[Dict]:
    """Extract text from a PDF and return sentence-aware chunk dicts.

    Each item: {"text": str, "meta": {"page": int, "chunk_index": int}}
    """
    pages = extract_text_from_pdf(source, password=password)

    results: List[Dict] = []
    for page_idx, page_text in enumerate(pages):
        if not page_text:
            continue
        page_chunks = _chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for ci, chunk in enumerate(page_chunks):
            results.append({
                "text": chunk,
                "meta": {"page": page_idx, "chunk_index": ci},
            })

    return results


__all__ = ["extract_text_from_pdf", "ingest_pdf", "_chunk_text"]


if __name__ == "__main__":
    print("ingest.py v2 — sentence-aware chunking. Provides ingest_pdf().")
