from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SessionInfo(BaseModel):
    id: str
    mode: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentInfo(BaseModel):
    id: str
    session_id: str
    name: str
    path: str
    num_pages: Optional[int] = None
    status: str

    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel):
    page_no: int
    text: str


class ParsedDocument(BaseModel):
    document_id: str
    pages: List[Page]
    metadata: dict = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Chunk(BaseModel):
    chunk_id: str
    session_id: str
    document_id: str
    page_no: int
    text: str
    section: str | None = None


class Finding(BaseModel):
    id: str
    session_id: str
    document_id: str
    page_no: int
    entity_type: str
    source: str
    masked_value: str
    confidence: float
    compliance_tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class RiskSummary(BaseModel):
    document_id: str
    score: float
    bucket: str
    explanation: str
    compliance_tags: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
