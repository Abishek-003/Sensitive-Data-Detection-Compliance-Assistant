from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    ForeignKey,
    Float,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship

from app.db import Base


class SessionORM(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)
    mode = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    documents = relationship(
        "DocumentORM",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class DocumentORM(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    num_pages = Column(Integer, nullable=True)
    status = Column(
        String,
        nullable=False,
        default="uploaded",
    )

    session = relationship("SessionORM", back_populates="documents")


class FindingORM(Base):
    __tablename__ = "findings"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = Column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_no = Column(Integer, nullable=False)
    entity_type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    masked_value = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    is_placeholder = Column(Boolean, nullable=False, default=False)
    compliance_tags = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("SessionORM")
    document = relationship("DocumentORM")


class ChatMessageORM(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("SessionORM")
