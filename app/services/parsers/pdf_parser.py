from __future__ import annotations

import io
import re

import fitz

from app.models.schemas import Page, ParsedDocument


def _clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _page_text_layout_aware(page: fitz.Page) -> str:
    page_rect = page.rect
    mid_x = page_rect.width / 2

    blocks = page.get_text("blocks")

    left_blocks = []
    right_blocks = []

    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        if not text.strip():
            continue
        mid_block_x = (x0 + x1) / 2
        if mid_block_x < mid_x:
            left_blocks.append(b)
        else:
            right_blocks.append(b)

    has_two_columns = len(left_blocks) > 0 and len(right_blocks) > 0

    def sort_blocks(bs):
        return sorted(bs, key=lambda bb: (bb[1], bb[0]))

    if has_two_columns:
        left_text = "\n".join(b[4] for b in sort_blocks(left_blocks))
        right_text = "\n".join(b[4] for b in sort_blocks(right_blocks))
        full_text = left_text + "\n\n" + right_text
    else:
        full_text = "\n".join(b[4] for b in sort_blocks(blocks))

    return _clean_text(full_text)


def _ocr_page_text(page: fitz.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    matrix = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    try:
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception:
        try:
            image = Image.open(io.BytesIO(pix.tobytes("png")))
        except Exception:
            return ""

    try:
        return _clean_text(pytesseract.image_to_string(image))
    except Exception:
        return ""


def parse_pdf(path: str, document_id: str) -> ParsedDocument:
    doc = fitz.open(path)
    pages: list[Page] = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        text = _page_text_layout_aware(page)
        if len(text) < 20:
            ocr_text = _ocr_page_text(page)
            if len(ocr_text) > len(text):
                text = ocr_text
        pages.append(Page(page_no=i + 1, text=text))

    metadata = {
        "source_path": path,
        "num_pages": len(pages),
        "filetype": "pdf",
    }

    return ParsedDocument(
        document_id=document_id,
        pages=pages,
        metadata=metadata,
    )
