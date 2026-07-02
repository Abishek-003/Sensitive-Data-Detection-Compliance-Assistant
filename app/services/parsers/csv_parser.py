import pandas as pd
from app.models.schemas import Page, ParsedDocument


def parse_csv(path: str, document_id: str) -> ParsedDocument:
    df = pd.read_csv(path)

    lines = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns]
        lines.append(" ; ".join(parts))

    text = "\n".join(lines)

    pages = [Page(page_no=1, text=text)]
    metadata = {
        "source_path": path,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "filetype": "csv",
    }

    return ParsedDocument(
        document_id=document_id,
        pages=pages,
        metadata=metadata,
    )
