from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: Path) -> list[str]:
    """Return a list of page texts (one entry per page, may be empty string)."""
    pages: list[str] = []

    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
    except Exception as e:
        logger.warning("pypdf failed for %s: %s", path.name, e)
        pages = []

    # Fallback to pymupdf for pages that yielded no text
    empty_indices = [i for i, t in enumerate(pages) if not t.strip()]
    if empty_indices:
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            for i in empty_indices:
                if i < len(doc):
                    text = doc[i].get_text()
                    pages[i] = text or ""
            doc.close()
        except Exception as e:
            logger.warning("pymupdf fallback failed for %s pages %s: %s", path.name, empty_indices, e)

    # If pages is still empty (pypdf failed entirely), try pymupdf from scratch
    if not pages:
        try:
            import fitz
            doc = fitz.open(str(path))
            pages = [page.get_text() or "" for page in doc]
            doc.close()
        except Exception as e:
            logger.warning("All PDF extraction failed for %s: %s", path.name, e)
            return []

    return pages


def chunk_pages(pages: list[str], chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    """
    Split pages into overlapping windows.
    Returns list of (start_page_0indexed, end_page_exclusive, combined_text).
    If chunk_size <= 0, returns a single chunk with all pages.
    """
    if chunk_size <= 0 or len(pages) <= chunk_size:
        text = "\n\n".join(p for p in pages if p.strip())
        return [(0, len(pages), text)]

    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(pages):
        end = min(i + chunk_size, len(pages))
        window = pages[i:end]
        text = "\n\n".join(p for p in window if p.strip())
        chunks.append((i, end, text))
        if end == len(pages):
            break
        i += step
    return chunks
