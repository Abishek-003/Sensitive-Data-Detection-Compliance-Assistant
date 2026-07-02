from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import Base, engine
from app.config import SESSION_CLEANUP_PORT, SQLITE_DB_PATH, TMP_DIR
from app.services.bm25_index import BM25Index
from app.services.browser_cleanup_server import ensure_browser_cleanup_server
from app.services.compliance import score_findings, generate_ai_compliance_summary
from app.services.embeddings import BGEEmbedder
from app.services.llm_detector import detect_document
from app.services.parsers import parse_document
from app.services.qa import answer_question, summarize_conversation
from app.services.rag_indexer import RAGIndexer
from app.services.retrieval import HybridRetriever
from app.services.session_manager import SessionManager
from app.services.vector_store import VectorStore
import streamlit.components.v1 as components


@st.cache_resource
def get_session_manager() -> SessionManager:
    return SessionManager()


@st.cache_resource
def get_embedder() -> BGEEmbedder:
    return BGEEmbedder()


@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource
def get_bm25_index() -> BM25Index:
    return BM25Index()


@st.cache_resource
def get_indexer() -> RAGIndexer:
    embedder = get_embedder()
    return RAGIndexer(embedder, get_vector_store(), get_bm25_index())


@st.cache_resource
def get_retriever() -> HybridRetriever:
    embedder = get_embedder()
    return HybridRetriever(embedder, get_vector_store(), get_bm25_index())


DISPLAY_ENTITY_LABELS = {
    "AADHAAR": "Aadhaar Numbers",
    "PAN": "PAN Numbers",
    "EMAIL": "Email Addresses",
    "PHONE": "Phone Numbers",
    "CREDIT_CARD": "Credit Card Numbers",
    "BANK_ACCOUNT": "Bank Details",
    "API_KEY": "API Keys / Passwords",
    "PASSWORD": "API Keys / Passwords",
    "EMPLOYEE_ID": "Employee IDs",
    "CONFIDENTIAL_BUSINESS_INFO": "Confidential Business Information",
    "SUSPICIOUS_SECRET": "API Keys / Passwords",
    "SUSPICIOUS_TOKEN": "API Keys / Passwords",
    "SUSPICIOUS_IDENTIFIER": "Employee IDs",
    "SUSPICIOUS_DATA": "Confidential Business Information",
}


def display_entity_label(entity_type: str) -> str:
    normalized = (entity_type or "").strip().upper()
    return DISPLAY_ENTITY_LABELS.get(normalized, normalized.replace("_", " ").title())


def display_masked_value(entity_type: str, masked_value: str) -> str:
    normalized_type = (entity_type or "").strip().upper()
    value = (masked_value or "").strip()
    if not value:
        return "***"
    if value.upper().startswith("SUSPICIOUS_"):
        if normalized_type == "EMPLOYEE_ID":
            return "21***1"
        if normalized_type in {"API_KEY", "PASSWORD", "SESSION_TOKEN", "PRIVATE_KEY"}:
            return "***REDACTED***"
        if normalized_type == "CONFIDENTIAL_BUSINESS_INFO":
            return "***CONFIDENTIAL***"
        return "***MASKED***"
    if normalized_type == "EMPLOYEE_ID" and value != "***MASKED***":
        alnum = "".join(ch for ch in value if ch.isalnum())
        if len(alnum) >= 3:
            return f"{alnum[:2]}***{alnum[-1]}"
    return value


def display_source_label(source: str) -> str:
    normalized = (source or "").strip().lower()
    if normalized == "llm":
        return "regex+LLM"
    return "regex"


def _purge_path(path: Path, *, recreate: bool = False) -> None:
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
    if recreate:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _purge_all_storage() -> None:
    try:
        if SQLITE_DB_PATH.exists():
            SQLITE_DB_PATH.unlink()
    except Exception:
        pass
    _purge_path(TMP_DIR, recreate=True)


def clear_session_everywhere(session_id: str) -> None:
    bm25_index = get_bm25_index()
    try:
        session_manager.terminate_session(session_id)
    except Exception:
        pass
    try:
        bm25_index.clear(session_id)
    except Exception:
        pass
    try:
        engine.dispose()
    except Exception:
        pass
    _purge_all_storage()

session_manager = get_session_manager()
Base.metadata.create_all(bind=engine)

st.set_page_config(page_title="Sensitive RAG + Compliance", layout="wide")
st.title("Sensitive RAG + Compliance")

if "session_id" not in st.session_state:
    sess = session_manager.create_session(mode="single")
    st.session_state["session_id"] = sess.id

st.session_state.setdefault("active_document_id", None)
st.session_state.setdefault("last_index_result", None)
st.session_state.setdefault("conversation_summary", "")
st.session_state.setdefault("doc_summaries", {})
st.session_state.setdefault("upload_file_key", 0)
if not st.session_state.get("storage_purged", False):
    _purge_all_storage()
    Base.metadata.create_all(bind=engine)
    st.session_state["storage_purged"] = True

session_id = st.session_state["session_id"]
ensure_browser_cleanup_server(clear_session_everywhere, port=SESSION_CLEANUP_PORT)
if session_manager.is_session_expired(session_id):
    clear_session_everywhere(session_id)
    sess = session_manager.create_session(mode="single")
    st.session_state["session_id"] = sess.id
    st.session_state["active_document_id"] = None
    st.session_state["last_index_result"] = None
    st.session_state["conversation_summary"] = ""
    st.session_state["doc_summaries"] = {}
    session_id = sess.id

documents = session_manager.list_documents(session_id)
document_map = {doc.id: doc for doc in documents}

components.html(
    f"""
<script>
(function() {{
  const sessionId = {session_id!r};
  const endpoint = "http://127.0.0.1:{SESSION_CLEANUP_PORT}/clear-session";
  const payload = JSON.stringify({{session_id: sessionId}});
  const blob = new Blob([payload], {{type: "application/json"}});
  const sendClear = function() {{
    try {{
      navigator.sendBeacon(endpoint, blob);
    }} catch (e) {{}}
  }};
  window.addEventListener("pagehide", sendClear);
  window.addEventListener("beforeunload", sendClear);
}})();
</script>
""",
    height=0,
)


def _display_findings_rows(findings):
    return [
        {
            "page": finding.page_no,
            "entity_type": display_entity_label(finding.entity_type),
            "masked_value": display_masked_value(finding.entity_type, finding.masked_value),
            "source": display_source_label(finding.source),
        }
        for finding in findings
    ]


def _render_document_summary(doc, findings, ai_summary: str | None = None):
    risk = score_findings(findings, document_id=doc.id)
    st.markdown(f"#### {doc.name}")
    cols = st.columns(3)
    cols[0].metric("Findings", len(findings))
    cols[1].metric("Risk", f"{risk.score} / {risk.bucket}")
    cols[2].metric("Pages", doc.num_pages or 0)
    st.write(risk.explanation)
    if ai_summary:
        st.markdown("**AI summary**")
        st.write(ai_summary)
    if findings:
        st.dataframe(_display_findings_rows(findings), use_container_width=True)
    else:
        st.info("No findings stored for this document.")


with st.sidebar:
    st.subheader("Session")
    st.code(session_id)
    st.caption(f"{len(documents)} document(s) in this session")

    if st.button("Clear session now", use_container_width=True):
        clear_session_everywhere(session_id)
        st.session_state.clear()
        st.rerun()


upload_tab, dashboard_tab, qa_tab = st.tabs(
    ["Upload and Index", "Compliance Dashboard", "Search and QA"]
)


with upload_tab:
    st.subheader("Upload a document")
    uploader_key = f"upload_file_{st.session_state['upload_file_key']}"
    uploaded = st.file_uploader(
        "Upload PDF / TXT / CSV",
        type=["pdf", "txt", "csv"],
        key=uploader_key,
    )

    if uploaded is not None:
        st.write(f"Filename: `{uploaded.name}`")

        if st.button("Analyze and index", type="primary"):
            indexer = get_indexer()
            doc_info = session_manager.register_document(session_id, uploaded.name)

            with open(doc_info.path, "wb") as f:
                f.write(uploaded.getvalue())

            parsed = parse_document(doc_info.path, document_id=doc_info.id)
            session_manager.update_document_status(doc_info.id, num_pages=len(parsed.pages))

            with st.spinner("Indexing document..."):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    index_future = executor.submit(
                        indexer.index_parsed_document,
                        session_id=session_id,
                        parsed=parsed,
                    )
                    detection_future = executor.submit(detect_document, parsed, session_id)
                    chunks, n_chunks = index_future.result()
                    detection_result = detection_future.result()
                    findings = detection_result.findings
                    risk = detection_result.risk
                    ai_summary = detection_result.summary or generate_ai_compliance_summary(findings, risk)
                    llm_message = getattr(
                        detection_result,
                        "llm_message",
                        "LLM does not work. Reason: detector result did not include a status message.",
                    )

            session_manager.save_findings(findings)
            session_manager.update_document_status(doc_info.id, status="indexed")

            stored_findings = session_manager.list_findings(session_id, doc_info.id)
            stored_risk = score_findings(stored_findings, document_id=doc_info.id)
            stored_ai_summary = ai_summary or generate_ai_compliance_summary(stored_findings, stored_risk)

            st.session_state["active_document_id"] = doc_info.id
            st.session_state["doc_summaries"][doc_info.id] = stored_ai_summary
            st.session_state["last_index_result"] = {
                "document_id": doc_info.id,
                "document_name": doc_info.name,
                "chunks": chunks,
                "findings": stored_findings,
                "risk": stored_risk,
                "ai_summary": stored_ai_summary,
                "llm_message": llm_message,
            }

            st.success(f"Indexed {n_chunks} chunks from {doc_info.name}")
            st.write(f"Risk bucket: `{stored_risk.bucket}`")
            st.write(f"Risk score: `{stored_risk.score}`")
            st.write(stored_risk.explanation)
            if str(llm_message).lower().startswith("llm works"):
                st.success(llm_message)
            else:
                st.warning(llm_message)
            st.caption("GDPR / DPDP / PCI DSS labels are compliance mappings from detected findings, not a full legal audit.")

            if stored_ai_summary:
                st.subheader("AI compliance summary")
                st.write(stored_ai_summary)

            if stored_findings:
                st.subheader("Detected findings")
                st.dataframe(_display_findings_rows(stored_findings), use_container_width=True)
            else:
                st.info("No sensitive findings were detected in this document.")

            documents = session_manager.list_documents(session_id)
            document_map = {doc.id: doc for doc in documents}

            st.session_state["upload_file_key"] += 1
            st.rerun()

    if st.session_state.get("last_index_result"):
        result = st.session_state["last_index_result"]
        st.divider()
        st.subheader("Last indexed document")
        st.write(f"Document: `{result['document_name']}`")
        st.write(f"Chunks: `{len(result['chunks'])}`")
        st.write(f"Risk: `{result['risk'].bucket}` ({result['risk'].score})")
        if result.get("ai_summary"):
            st.markdown("**AI compliance summary**")
            st.write(result["ai_summary"])
        llm_message = str(result.get("llm_message") or "")
        if llm_message:
            if llm_message.lower().startswith("llm works"):
                st.success(llm_message)
            else:
                st.warning(llm_message)
        if result.get("findings"):
            st.markdown("**Detected findings**")
            st.dataframe(_display_findings_rows(result["findings"]), use_container_width=True)


with dashboard_tab:
    st.subheader("Session dashboard")

    if not documents:
        st.info("Upload and index a document first.")
    else:
        doc_options = ["All documents"] + [doc.id for doc in documents]
        default_index = 0
        active_doc = st.session_state.get("active_document_id")
        if active_doc in document_map:
            default_index = doc_options.index(active_doc)

        selected_doc = st.selectbox(
            "Select document",
            options=doc_options,
            index=default_index,
            format_func=lambda value: "All documents" if value == "All documents" else document_map[value].name,
        )

        all_findings = session_manager.list_findings(session_id)
        findings_by_doc = {
            doc.id: [finding for finding in all_findings if finding.document_id == doc.id]
            for doc in documents
        }
        pdf_count = sum(1 for doc in documents if doc.name.lower().endswith(".pdf"))
        indexed_count = sum(1 for doc in documents if doc.status == "indexed")

        if selected_doc == "All documents":
            cols = st.columns(4)
            cols[0].metric("Documents", len(documents))
            cols[1].metric("PDF uploads", pdf_count)
            cols[2].metric("Indexed", indexed_count)
            cols[3].metric("Findings", len(all_findings))
            st.caption("GDPR / DPDP / PCI DSS labels are compliance mappings from detected findings, not a full legal audit.")

            st.caption("Each document is summarized separately below.")
            for doc in documents:
                with st.expander(doc.name, expanded=(doc.id == active_doc)):
                    _render_document_summary(
                        doc,
                        findings_by_doc.get(doc.id, []),
                        st.session_state["doc_summaries"].get(doc.id),
                    )
        else:
            doc = document_map[selected_doc]
            doc_findings = findings_by_doc.get(doc.id, [])
            cols = st.columns(4)
            cols[0].metric("Documents", len(documents))
            cols[1].metric("PDF uploads", pdf_count)
            cols[2].metric("Indexed", indexed_count)
            cols[3].metric("Findings", len(doc_findings))
            _render_document_summary(
                doc,
                doc_findings,
                st.session_state["doc_summaries"].get(doc.id),
            )


with qa_tab:
    st.subheader("Hybrid retrieval and QA")

    if not documents:
        st.info("Upload and index a document first.")
    else:
        doc_options = ["All documents"] + [doc.id for doc in documents]
        default_index = 0
        active_doc = st.session_state.get("active_document_id")
        if active_doc in document_map:
            default_index = doc_options.index(active_doc)

        selected_doc = st.selectbox(
            "Scope",
            options=doc_options,
            index=default_index,
            key="qa_scope",
            format_func=lambda value: "All documents" if value == "All documents" else document_map[value].name,
        )

        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input("Ask a question about the indexed document(s)")
            asked = st.form_submit_button("Ask")

        if asked and question.strip():
            retriever = get_retriever()
            hits = retriever.search(
                session_id,
                question,
                k=5,
                document_id=None if selected_doc == "All documents" else selected_doc,
            )

            scoped_findings = session_manager.list_findings(
                session_id,
                None if selected_doc == "All documents" else selected_doc,
            )

            history = session_manager.list_chat_messages(session_id, limit=12)
            conversation_summary = summarize_conversation(
                history,
                st.session_state.get("conversation_summary", ""),
            )
            st.session_state["conversation_summary"] = conversation_summary

            response = answer_question(
                question=question,
                hits=hits,
                findings=scoped_findings,
                conversation_summary=conversation_summary,
                document_titles={doc.id: doc.name for doc in documents},
            )

            session_manager.save_chat_message(session_id, "user", question)
            session_manager.save_chat_message(session_id, "assistant", response.answer_text)

            st.markdown("### Answer")
            st.write(response.answer_text)
            st.markdown("### Compliance insight")
            st.write(response.compliance_insights)

            st.markdown("### Citations")
            if response.citations:
                for citation in response.citations:
                    st.write(citation)
            else:
                st.info("No citations available for this answer.")

        st.markdown("### Recent conversation")
        history = session_manager.list_chat_messages(session_id, limit=10)
        if history:
            for msg in history:
                with st.chat_message(msg.role):
                    st.write(msg.content)
        else:
            st.info("No conversation yet.")
