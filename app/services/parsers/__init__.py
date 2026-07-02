from pathlib import Path

from app.models.schemas import ParsedDocument
from .pdf_parser import parse_pdf
from .txt_parser import parse_txt
from .csv_parser import parse_csv


def parse_document(path: str, document_id: str) -> ParsedDocument:
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path, document_id)
    elif suffix == ".txt":
        return parse_txt(path, document_id)
    elif suffix == ".csv":
        return parse_csv(path, document_id)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
