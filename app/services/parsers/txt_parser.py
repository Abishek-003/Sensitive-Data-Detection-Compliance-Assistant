import re
from app.models.schemas import Page, ParsedDocument


def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_txt(path: str, document_id: str) -> ParsedDocument:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    cleaned = _clean_text(raw)

    pages = [Page(page_no=1, text=cleaned)]
    metadata = {
        "source_path": path,
        "num_pages": 1,
        "filetype": "txt",
    }

    return ParsedDocument(
        document_id=document_id,
        pages=pages,
        metadata=metadata,
    )
