from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import uuid4
from datetime import datetime
from datetime import timedelta

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.config import TMP_DIR, SESSION_TTL_MINUTES
from app.models.orm import SessionORM, DocumentORM, FindingORM, ChatMessageORM
from app.models.schemas import SessionInfo, DocumentInfo, Finding, ChatMessage
from app.services.compliance import normalize_entity_type

LEGACY_EXAMPLE_ENTITY_TYPES = {
    "EXAMPLE_DATA",
    "TEST_DATA",
    "EXAMPLE",
    "SAMPLE",
    "DEMO",
}


class SessionManager:
    def __init__(self, tmp_dir: Path = TMP_DIR):
        self.tmp_dir = tmp_dir
        self.session_ttl_minutes = SESSION_TTL_MINUTES

    @staticmethod
    def _finding_key(
        session_id: str,
        document_id: str,
        page_no: int,
        entity_type: str,
        masked_value: str,
    ) -> tuple[str, str, int, str, str]:
        return (
            session_id,
            document_id,
            page_no,
            normalize_entity_type(entity_type),
            (masked_value or "").strip().lower(),
        )

    def create_session(
        self,
        mode: Literal["single", "multi"] = "single",
    ) -> SessionInfo:
        """
        If mode == 'single', create a brand new session.
        You can later add logic to delete previous sessions here.
        """
        db: Session = SessionLocal()
        try:
            session_id = str(uuid4())
            obj = SessionORM(
                id=session_id,
                mode=mode,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(obj)
            db.commit()
            db.refresh(obj)
            return SessionInfo.model_validate(obj)
        finally:
            db.close()

    def get_session(self, session_id: str) -> SessionInfo | None:
        db: Session = SessionLocal()
        try:
            obj = db.get(SessionORM, session_id)
            if not obj:
                return None
            return SessionInfo.model_validate(obj)
        finally:
            db.close()

    def is_session_expired(self, session_id: str) -> bool:
        db: Session = SessionLocal()
        try:
            obj = db.get(SessionORM, session_id)
            if not obj:
                return True
            expiry = obj.updated_at + timedelta(minutes=self.session_ttl_minutes)
            return datetime.utcnow() > expiry
        finally:
            db.close()

    def register_document(
        self,
        session_id: str,
        filename: str,
    ) -> DocumentInfo:
        """
        Create a DocumentORM row and return its info + temp path.
        """
        db: Session = SessionLocal()
        try:
            session_obj = db.get(SessionORM, session_id)
            if session_obj is None:
                raise ValueError(f"Session {session_id} not found")

            document_id = str(uuid4())

            session_dir = self.tmp_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / f"{document_id}_{filename}"

            doc = DocumentORM(
                id=document_id,
                session_id=session_id,
                name=filename,
                path=str(path),
                status="uploaded",
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            session_obj.updated_at = datetime.utcnow()
            db.commit()
            return DocumentInfo.model_validate(doc)
        finally:
            db.close()

    def list_documents(self, session_id: str) -> list[DocumentInfo]:
        db: Session = SessionLocal()
        try:
            docs = (
                db.query(DocumentORM)
                .filter(DocumentORM.session_id == session_id)
                .all()
            )
            return [DocumentInfo.model_validate(d) for d in docs]
        finally:
            db.close()

    def update_document_status(
        self,
        document_id: str,
        *,
        status: str | None = None,
        num_pages: int | None = None,
    ) -> None:
        db: Session = SessionLocal()
        try:
            doc = db.get(DocumentORM, document_id)
            if doc is None:
                return
            if status is not None:
                doc.status = status
            if num_pages is not None:
                doc.num_pages = num_pages
            doc.session.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()

    def terminate_session(self, session_id: str) -> None:
        """
        Delete DB rows + temp files for a session.
        Vector DB and BM25 cleanup will be wired later.
        """
        db: Session = SessionLocal()
        try:
            obj = db.get(SessionORM, session_id)
            if obj:
                db.query(FindingORM).filter(FindingORM.session_id == session_id).delete()
                db.query(ChatMessageORM).filter(ChatMessageORM.session_id == session_id).delete()
                db.delete(obj)
                db.commit()
        finally:
            db.close()

        session_dir = self.tmp_dir / session_id
        if session_dir.exists():
            for p in session_dir.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                session_dir.rmdir()
            except Exception:
                pass

    def save_findings(self, findings: list[Finding]) -> None:
        if not findings:
            return
        db: Session = SessionLocal()
        try:
            existing_rows = {
                self._finding_key(
                    row.session_id,
                    row.document_id,
                    row.page_no,
                    row.entity_type,
                    row.masked_value,
                ): row
                for row in db.query(FindingORM)
                .filter(
                    FindingORM.session_id == findings[0].session_id,
                    FindingORM.document_id == findings[0].document_id,
                )
                .all()
            }
            for finding in findings:
                key = self._finding_key(
                    finding.session_id,
                    finding.document_id,
                    finding.page_no,
                    finding.entity_type,
                    finding.masked_value,
                )
                existing_row = existing_rows.get(key)
                if existing_row is not None:
                    if finding.source == "llm" and existing_row.source != "llm":
                        existing_row.source = "llm"
                        existing_row.confidence = finding.confidence
                    continue
                db.add(
                    FindingORM(
                        id=finding.id,
                        session_id=finding.session_id,
                        document_id=finding.document_id,
                        page_no=finding.page_no,
                        entity_type=normalize_entity_type(finding.entity_type),
                        source=finding.source,
                        masked_value=finding.masked_value,
                        confidence=finding.confidence,
                        compliance_tags="[]",
                        created_at=finding.created_at or datetime.utcnow(),
                    )
                )
            session_obj = db.get(SessionORM, findings[0].session_id)
            if session_obj is not None:
                session_obj.updated_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()

    def list_findings(self, session_id: str, document_id: str | None = None) -> list[Finding]:
        db: Session = SessionLocal()
        try:
            query = db.query(FindingORM).filter(FindingORM.session_id == session_id)
            if document_id:
                query = query.filter(FindingORM.document_id == document_id)
            rows = query.order_by(FindingORM.created_at.asc()).all()
            deduped: dict[tuple[str, str, int, str, str], Finding] = {}
            for row in rows:
                if (row.entity_type or "").strip().upper() in LEGACY_EXAMPLE_ENTITY_TYPES:
                    continue
                finding = Finding(
                    id=row.id,
                    session_id=row.session_id,
                    document_id=row.document_id,
                    page_no=row.page_no,
                    entity_type=normalize_entity_type(row.entity_type),
                    source=row.source,
                    masked_value=row.masked_value,
                    confidence=row.confidence,
                    compliance_tags=[],
                    created_at=row.created_at,
                )
                key = self._finding_key(
                    finding.session_id,
                    finding.document_id,
                    finding.page_no,
                    finding.entity_type,
                    finding.masked_value,
                )
                existing = deduped.get(key)
                if existing is None or existing.source != "llm" and finding.source == "llm":
                    deduped[key] = finding
            return [
                deduped[key]
                for key in sorted(deduped.keys(), key=lambda item: (item[2], item[3], item[4]))
            ]
        finally:
            db.close()

    def save_chat_message(self, session_id: str, role: str, content: str) -> None:
        db: Session = SessionLocal()
        try:
            db.add(
                ChatMessageORM(
                    id=str(uuid4()),
                    session_id=session_id,
                    role=role,
                    content=content,
                    created_at=datetime.utcnow(),
                )
            )
            db.commit()
            session_obj = db.get(SessionORM, session_id)
            if session_obj is not None:
                session_obj.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    def list_chat_messages(self, session_id: str, limit: int = 20) -> list[ChatMessage]:
        db: Session = SessionLocal()
        try:
            rows = (
                db.query(ChatMessageORM)
                .filter(ChatMessageORM.session_id == session_id)
                .order_by(ChatMessageORM.created_at.asc())
                .all()
            )
            rows = rows[-limit:]
            return [
                ChatMessage(
                    id=row.id,
                    session_id=row.session_id,
                    role=row.role,
                    content=row.content,
                    created_at=row.created_at,
                )
                for row in rows
            ]
        finally:
            db.close()
