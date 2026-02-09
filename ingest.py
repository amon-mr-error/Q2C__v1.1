"""PDF ingestion utilities.

This module provides a small data-ingestion layer that:
- reads a PDF from a file path, bytes, or file-like object;
- extracts text per page (with basic normalization);
- splits text into overlapping chunks suitable for passing to a
  HuggingFace transformer or an embedding model.

The functions avoid importing heavy dependencies at module import time
so the module can be imported even when `PyPDF2` isn't installed; a
clear ImportError is raised when PDF functions are actually used.

Primary API:
- ingest_pdf(source, chunk_size=1000, overlap=200) -> list of chunks

Each chunk is a dict: {"text": str, "meta": {"page": int, "chunk_index": int}}
"""

from typing import List, Dict, Union, BinaryIO, Optional
import re


def _ensure_pdf_reader():
	try:
		from PyPDF2 import PdfReader  # imported lazily
	except Exception as e:  # ImportError or other import-time issues
		raise ImportError(
			"PyPDF2 is required to read PDFs. Install with: pip install PyPDF2"
		) from e

	return PdfReader


def _normalize_text(text: Optional[str]) -> str:
	if not text:
		return ""
	# Collapse whitespace, fix newlines
	text = text.replace('\r', '\n')
	text = re.sub(r"\n{2,}", "\n\n", text)
	text = re.sub(r"[ \t]+", " ", text)
	return text.strip()


def extract_text_from_pdf(
	source: Union[str, bytes, BinaryIO], password: Optional[str] = None
) -> List[str]:
	"""Extract text from each page of a PDF.

	Args:
		source: file path (str), bytes object, or file-like binary stream.
		password: optional password for encrypted PDFs.

	Returns:
		A list of strings, one per page (empty string for pages with no text).

	Raises:
		ImportError: if PyPDF2 is not installed.
		ValueError: if the source type is unsupported or PDF cannot be read.
	"""
	PdfReader = _ensure_pdf_reader()

	# Support bytes, file path, or file-like objects
	reader = None
	try:
		if isinstance(source, (bytes, bytearray)):
			from io import BytesIO

			reader = PdfReader(BytesIO(source))
		else:
			# PdfReader accepts a path or a file-like object
			reader = PdfReader(source)
	except Exception as e:
		raise ValueError(f"Failed to open PDF source: {e}") from e

	# If encrypted, attempt to decrypt with provided password
	try:
		if hasattr(reader, "is_encrypted") and reader.is_encrypted:
			if password is None:
				# Some PDFs can be opened but not read without a password
				raise ValueError("PDF is encrypted; provide a password to read it")
			try:
				reader.decrypt(password)
			except Exception:
				# older/newer PyPDF2 versions differ; try alternate call
				try:
					reader.decrypt("")
				except Exception:
					raise
	except Exception:
		# If decryption fails, propagate cleanly
		raise

	pages_text: List[str] = []
	for p in reader.pages:
		try:
			text = p.extract_text()
		except Exception:
			# Extracting text can fail for some PDFs; fallback to empty string
			text = ""
		pages_text.append(_normalize_text(text))

	return pages_text


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
	"""Split a single long string into overlapping character chunks.

	This keeps chunking simple and deterministic for downstream embedding.
	"""
	if chunk_size <= 0:
		raise ValueError("chunk_size must be > 0")
	if overlap < 0:
		raise ValueError("overlap must be >= 0")
	if overlap >= chunk_size:
		raise ValueError("overlap must be smaller than chunk_size")

	chunks: List[str] = []
	start = 0
	text_len = len(text)
	step = chunk_size - overlap
	if text_len == 0:
		return []
	while start < text_len:
		end = start + chunk_size
		chunks.append(text[start:end])
		start += step
	return chunks


def ingest_pdf(
	source: Union[str, bytes, BinaryIO], chunk_size: int = 1000, overlap: int = 200, password: Optional[str] = None
) -> List[Dict]:
	"""High-level PDF ingestion: extract text and return chunk dicts.

	Each returned item is a dict with keys:
	  - "text": the chunk string
	  - "meta": {"page": page_index (0-based), "chunk_index": index_within_page}

	This format is intentionally simple and easy to convert into the
	inputs expected by embedding models or an HF transformer pipeline.
	"""
	pages = extract_text_from_pdf(source, password=password)

	results: List[Dict] = []
	for page_idx, page_text in enumerate(pages):
		if not page_text:
			continue
		page_chunks = _chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
		for ci, chunk in enumerate(page_chunks):
			results.append({"text": chunk, "meta": {"page": page_idx, "chunk_index": ci}})

	return results


__all__ = [
	"extract_text_from_pdf",
	"_chunk_text",
	"ingest_pdf",
]


if __name__ == "__main__":
	print("ingest.py module - provides extract_text_from_pdf() and ingest_pdf()")


