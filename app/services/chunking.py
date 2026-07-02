from __future__ import annotations
from typing import List
from uuid import uuid4

from app.models.schemas import ParsedDocument, Chunk


def chunk_parsed_document(
    parsed: ParsedDocument,
    session_id: str,
    max_chars: int = 1000,
    overlap_paragraphs: int = 1,
) -> List[Chunk]:
    chunks: List[Chunk] = []

    for page in parsed.pages:
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]

        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            if not para:
                continue

            para_len = len(para) + 1

            if current_len + para_len > max_chars and current_parts:
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid4()),
                        session_id=session_id,
                        document_id=parsed.document_id,
                        page_no=page.page_no,
                        text=chunk_text,
                        section=None,
                    )
                )

                if overlap_paragraphs > 0:
                    overlap = current_parts[-overlap_paragraphs:]
                else:
                    overlap = []

                current_parts = overlap + [para]
                current_len = sum(len(p) + 1 for p in current_parts)
            else:
                current_parts.append(para)
                current_len += para_len

        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    session_id=session_id,
                    document_id=parsed.document_id,
                    page_no=page.page_no,
                    text=chunk_text,
                    section=None,
                )
            )

    return chunks
