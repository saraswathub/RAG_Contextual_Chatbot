import sys
import os
from pathlib import Path
import logging
import hashlib
from rag_pipeline import ChatRAG

# Optional PDF support; if PyPDF2 is not installed we'll skip PDFs with a warning.
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

DATA_DIR = Path.cwd() / 'data' / 'sample_docs'

def extract_text_from_pdf(path: Path) -> str:
    if not PDF_AVAILABLE:
        raise RuntimeError("PyPDF2 is not installed; cannot extract PDF text.")
    text_parts = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        text_parts.append(page_text)
    return "\n".join(text_parts)

def load_sample_documents(recursive: bool = False):
    docs = []
    if not DATA_DIR.exists():
        logging.error("DATA_DIR %s does not exist.", DATA_DIR)
        return docs

    walker = DATA_DIR.rglob("*") if recursive else DATA_DIR.iterdir()
    for p in walker:
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in ('.md', '.txt', '.pdf'):
            continue

        try:
            if suffix == '.pdf':
                if not PDF_AVAILABLE:
                    logging.warning("Skipping PDF (PyPDF2 not installed): %s", p)
                    continue
                text = extract_text_from_pdf(p)
            else:
                # read text files, be tolerant of encoding issues
                text = p.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            logging.exception("Failed to read %s: %s", p, e)
            continue

        # stable id: sha1 of absolute path (you can change to uuid if preferred)
        id_hash = hashlib.sha1(str(p.resolve()).encode('utf-8')).hexdigest()
        docs.append({
            'id': id_hash,
            'text': text,
            'source': p.name,
            'path': str(p.resolve()),
        })

    return docs

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    rag = ChatRAG()
    docs = load_sample_documents(recursive=False)
    if not docs:
        logging.info('No sample docs found in %s. Add files and re-run.', DATA_DIR)
        return

    logging.info('Ingesting %d docs...', len(docs))
    try:
        rag.ingest_documents(docs)
    except Exception:
        logging.exception("rag.ingest_documents failed.")
        raise
    logging.info('Ingestion complete. Chroma DB persisted in ./chroma_db/')


if __name__ == '__main__':
    main()
