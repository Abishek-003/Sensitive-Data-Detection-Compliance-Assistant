from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import Finding, ChatMessage
from app.services.llm import chat_completion
from app.services.retrieval import SearchHit


@dataclass
class QAResponse:
    answer_text: str
    context_used: list[SearchHit]
    citations: list[str]
    compliance_insights: str


def _summarize_findings(findings: list[Finding]) -> str:
    if not findings:
        return "No compliance findings were associated with this document."
    grouped: dict[str, int] = {}
    for finding in findings:
        grouped[finding.entity_type] = grouped.get(finding.entity_type, 0) + 1
    summary = ", ".join(f"{entity} x{count}" for entity, count in sorted(grouped.items()))
    return f"Findings detected: {summary}."


def summarize_conversation(messages: list[ChatMessage], current_summary: str = "") -> str:
    if not messages:
        return current_summary
    if len(messages) <= 6:
        lines = [f"{msg.role}: {msg.content}" for msg in messages[-6:]]
        return "\n".join(lines)

    prompt = [
        {
            "role": "system",
            "content": (
                "Summarize the conversation in 4-6 bullet points, preserving the user's goals, "
                "document scope, and any important constraints."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(f"{msg.role}: {msg.content}" for msg in messages[-12:]),
        },
    ]
    summary = chat_completion(prompt, temperature=0.1, max_tokens=220)
    return summary or current_summary or "\n".join(f"{msg.role}: {msg.content}" for msg in messages[-6:])


def _query_terms(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}\"'`").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}\"'`")) > 2
    }


ENTITY_KEYWORDS: dict[str, set[str]] = {
    "PHONE": {"phone", "mobile", "contact number", "ph no", "phonenumber"},
    "EMAIL": {"email", "mail", "e-mail"},
    "AADHAAR": {"aadhaar", "aadhar", "uid"},
    "CREDIT_CARD": {"credit card", "card number", "cvv"},
    "BANK_ACCOUNT": {"bank account", "account number", "a/c", "ifsc"},
    "API_KEY": {"api key", "secret key", "token"},
    "PASSWORD": {"password", "passwd", "passcode"},
    "EMPLOYEE_ID": {"employee id", "emp id", "staff id"},
}

SENSITIVE_EXFILTRATION_HINTS = {
    "without encryption",
    "unencrypted",
    "unmasked",
    "raw",
    "bypass",
    "extract",
    "reveal",
    "give me",
    "show me",
    "print",
    "temp_api_token",
}


def _detect_requested_entities(question: str) -> set[str]:
    q = question.lower()
    requested = set()
    for entity, keywords in ENTITY_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            requested.add(entity)
    return requested


def _is_exfiltration_attempt(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in SENSITIVE_EXFILTRATION_HINTS)


def _is_count_question(question: str) -> bool:
    q = question.lower()
    return any(
        phrase in q
        for phrase in (
            "how many",
            "count of",
            "number of",
            "total",
            "how much",
            "present",
            "found",
            "list",
        )
    )


def is_query_relevant(question: str, hits: list[SearchHit], min_overlap: int = 3) -> bool:
    if not question.strip() or not hits:
        return False

    q_terms = _query_terms(question)
    if not q_terms:
        return False

    best_overlap = 0
    for hit in hits[:3]:
        text_terms = _query_terms(hit.chunk.text)
        overlap = len(q_terms & text_terms)
        if overlap > best_overlap:
            best_overlap = overlap

    if best_overlap >= min_overlap:
        return True

    top = hits[0]
    return top.rerank_score >= 0.15 or top.rrf_score >= 0.02


def _build_finding_citation(finding: Finding) -> str:
    return f"{finding.document_id}:p{finding.page_no}"


def _format_document_label(document_id: str, document_titles: dict[str, str] | None) -> str:
    if document_titles and document_id in document_titles:
        return document_titles[document_id]
    return document_id


def answer_question(
    question: str,
    hits: list[SearchHit],
    findings: list[Finding],
    conversation_summary: str = "",
    document_titles: dict[str, str] | None = None,
) -> QAResponse:
    compliance_insights = _summarize_findings(findings)

    requested_entities = _detect_requested_entities(question)
    if requested_entities:
        matched = [finding for finding in findings if finding.entity_type in requested_entities]
        if matched:
            count = len(matched)
            pages = sorted({finding.page_no for finding in matched})
            page_text = ", ".join(f"page {page}" for page in pages)
            entity_label = matched[0].entity_type.lower().replace("_", " ")
            citations = [
                f"{_format_document_label(finding.document_id, document_titles)}, page {finding.page_no}"
                for finding in matched[:5]
            ]

            if _is_exfiltration_attempt(question):
                return QAResponse(
                    answer_text=(
                        f"I can't provide the unmasked {entity_label} value. "
                        f"I found {count} matching {entity_label}{'s' if count != 1 else ''} across {page_text}. "
                        f"Example masked value: {matched[0].masked_value}."
                    ),
                    context_used=[],
                    citations=citations,
                    compliance_insights=compliance_insights,
                )

            if _is_count_question(question):
                return QAResponse(
                    answer_text=(
                        f"I found {count} {entity_label}{'s' if count != 1 else ''} in the document. "
                        f"They appear on {page_text}. "
                        f"Example masked value: {matched[0].masked_value}."
                    ),
                    context_used=[],
                    citations=citations,
                    compliance_insights=compliance_insights,
                )

            return QAResponse(
                answer_text=(
                    f"I found {count} {entity_label}{'s' if count != 1 else ''} in the document, "
                    f"mainly on {page_text}. "
                    f"Example masked value: {matched[0].masked_value}."
                ),
                context_used=[],
                citations=citations,
                compliance_insights=compliance_insights,
            )

    if not is_query_relevant(question, hits):
        return QAResponse(
            answer_text=(
                "I could not find relevant information in the uploaded documents for that question. "
                "Please rephrase it or make it more specific to the document content."
            ),
            context_used=[],
            citations=[],
            compliance_insights=compliance_insights,
        )

    citations = [
        f"{_format_document_label(hit.chunk.document_id, document_titles)}, page {hit.chunk.page_no}"
        for hit in hits
    ]

    context_blob = "\n\n".join(
        f"[{i}] doc={hit.chunk.document_id} page={hit.chunk.page_no}\n{hit.chunk.text[:1200]}"
        for i, hit in enumerate(hits, start=1)
    )
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant answering questions about uploaded documents. "
                "Use only the provided context and keep answers concise, specific, and cited."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Conversation summary: {conversation_summary or 'None'}\n\n"
                f"Compliance findings summary: {compliance_insights}\n\n"
                f"Context:\n{context_blob or 'No retrieved context'}\n\n"
                "Return a short answer, then a brief supporting explanation."
            ),
        },
    ]

    answer = chat_completion(prompt, temperature=0.2, max_tokens=500)
    if not answer:
        if not hits:
            answer = "I could not find relevant context in the indexed session."
        else:
            top_context = hits[0].chunk.text[:700]
            answer = (
                f"Question: {question}\n\n"
                f"Best matching context: {top_context}\n\n"
                f"Relevant documents/pages were retrieved and ranked using hybrid search."
            )

        if conversation_summary:
            answer = f"{answer}\n\nConversation summary: {conversation_summary}"

    return QAResponse(
        answer_text=answer,
        context_used=hits,
        citations=citations,
        compliance_insights=compliance_insights,
    )
