from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from app.models.schemas import Finding, ParsedDocument, RiskSummary
from app.services.compliance import detect_findings, score_findings
from app.services.llm import chat_completion, llm_enabled, llm_last_error


ALLOWED_ENTITY_TYPES = {
    "AADHAAR",
    "PAN",
    "EMAIL",
    "PHONE",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "API_KEY",
    "PASSWORD",
    "EMPLOYEE_ID",
    "SESSION_TOKEN",
    "PRIVATE_KEY",
    "CONFIDENTIAL_BUSINESS_INFO",
    "OTHER",
}


@dataclass(frozen=True)
class LLMDetectorResult:
    findings: list[Finding]
    risk: RiskSummary
    summary: str
    llm_message: str


@dataclass(frozen=True)
class FindingReviewItem:
    review_id: str
    page_no: int
    entity_type: str
    masked_value: str
    page_context: str


AADHAAR_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?ix)"
    r"(?<!\d)"
    r"(?:\+?91[\s\-()]*)?"
    r"(?:0[\s\-()]*)?"
    r"(?:"
    r"\d{10}"
    r"|(?:\d[\s\-()]*){10}"
    r")"
    r"(?!\d)"
)
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
BANK_ACCOUNT_RE = re.compile(r"\b\d{9,18}\b")
API_KEY_RE = re.compile(
    r"(?i)\b(?:sk-or-v1-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|rk-[A-Za-z0-9]{20,}|"
    r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}|"
    r"(?:api[_-]?key|token|secret|access[_-]?token|auth[_-]?token|client[_-]?secret)\b\s*[:=]\s*[^\s,;\"']{8,})"
)
PASSWORD_RE = re.compile(
    r"(?i)\b(?:password|passwd|passcode|secret|db[_-]?password|secret[_-]?password)\b\s*[:=]\s*([^\s,;]+)"
)
EMPLOYEE_ID_RE = re.compile(
    r"(?i)\b(?:employee\s*id|emp\s*id|staff\s*id|member\s*id|user\s*id|eid|empid|employee\s*number|emp[-_]?\d{3,})\b(?:\s*[:#-]?\s*([A-Z0-9-]{4,}))?"
)
SESSION_TOKEN_RE = re.compile(
    r"(?i)\b(?:session[_-]?token|refresh[_-]?token|bearer[_-]?token)\b\s*[:=]\s*([A-Za-z0-9._\-\/+=]{12,})"
)
PRIVATE_KEY_RE = re.compile(
    r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)
CONFIDENTIAL_BUSINESS_RE = re.compile(
    r"(?i)\b(?:confidential|proprietary|internal only|do not share|restricted|trade secret|non-public|pricing|client list|roadmap|forecast|revenue|strategy|merger|acquisition|source code|architecture|blueprint)\b"
)

def _parse_json(payload: str | None) -> dict:
    if not payload:
        return {}
    text = payload.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


def _trim_context(text: str, limit: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _paragraph_spans(text: str) -> list[tuple[int, int, str]]:
    blocks = [block for block in re.split(r"\n\s*\n+", text) if block.strip()]
    if not blocks:
        return []

    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for block in blocks:
        idx = text.find(block, cursor)
        if idx < 0:
            continue
        end = idx + len(block)
        spans.append((idx, end, block))
        cursor = end
    if not spans:
        return [(0, len(text), text)]
    return spans


def _paragraph_triplet(text: str, match_start: int, match_end: int) -> tuple[str, str, str]:
    spans = _paragraph_spans(text)
    if not spans:
        return "", _trim_context(text), ""

    current_index = 0
    for index, (start, end, _) in enumerate(spans):
        if start <= match_start <= end or start <= match_end <= end:
            current_index = index
            break
        if match_start < start:
            current_index = max(0, index - 1)
            break

    previous_chunk = spans[current_index - 1][2] if current_index - 1 >= 0 else ""
    paragraph_chunk = spans[current_index][2] if current_index < len(spans) else text[match_start:match_end]
    next_chunk = spans[current_index + 1][2] if current_index + 1 < len(spans) else ""
    return _trim_context(previous_chunk), _trim_context(paragraph_chunk), _trim_context(next_chunk)


def _evidence_snippet(text: str, match_start: int, match_end: int, window: int = 90) -> str:
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    return _trim_context(text[start:end], limit=140)


def _normalize_entity_type(entity_type: str, hint: str | None = None) -> str:
    value = (entity_type or "").strip().upper()
    alias_map = {
        "AADHAAR NUMBER": "AADHAAR",
        "AADHAAR NUMBERS": "AADHAAR",
        "AADHAR": "AADHAAR",
        "AADHAR NUMBER": "AADHAAR",
        "AADHAR NUMBERS": "AADHAAR",
        "EMAIL ADDRESS": "EMAIL",
        "EMAIL ADDRESSES": "EMAIL",
        "PHONE NUMBER": "PHONE",
        "PHONE NUMBERS": "PHONE",
        "MOBILE NUMBER": "PHONE",
        "MOBILE NUMBERS": "PHONE",
        "BANK DETAILS": "BANK_ACCOUNT",
        "BANK ACCOUNT NUMBER": "BANK_ACCOUNT",
        "BANK ACCOUNT NUMBERS": "BANK_ACCOUNT",
        "API KEYS": "API_KEY",
        "PASSWORDS": "PASSWORD",
        "EMPLOYEE IDS": "EMPLOYEE_ID",
        "EMPLOYEE NUMBER": "EMPLOYEE_ID",
        "EMPLOYEE NUMBERS": "EMPLOYEE_ID",
        "CONFIDENTIAL BUSINESS INFORMATION": "CONFIDENTIAL_BUSINESS_INFO",
        "SUSPICIOUS_SECRET": "API_KEY",
        "SUSPICIOUS_TOKEN": "API_KEY",
        "SUSPICIOUS_IDENTIFIER": "EMPLOYEE_ID",
        "SUSPICIOUS_DATA": "CONFIDENTIAL_BUSINESS_INFO",
        "SECRET": "API_KEY",
        "TOKEN": "API_KEY",
        "CREDENTIAL": "API_KEY",
        "CREDENTIALS": "API_KEY",
        "PASSCODE": "PASSWORD",
        "PASSWD": "PASSWORD",
        "EMP ID": "EMPLOYEE_ID",
        "STAFF ID": "EMPLOYEE_ID",
        "SENSITIVE DATA": "CONFIDENTIAL_BUSINESS_INFO",
        "CONFIDENTIAL": "CONFIDENTIAL_BUSINESS_INFO",
        "CONFIDENTIAL INFO": "CONFIDENTIAL_BUSINESS_INFO",
        "CONFIDENTIAL DATA": "CONFIDENTIAL_BUSINESS_INFO",
    }
    if value in alias_map:
        return alias_map[value]
    if value in {"", "OTHER"}:
        return "OTHER"
    if value in ALLOWED_ENTITY_TYPES:
        return value
    if hint in ALLOWED_ENTITY_TYPES:
        return hint
    if hint == "CONFIDENTIAL_BUSINESS_INFO":
        return "CONFIDENTIAL_BUSINESS_INFO"
    return value if value in {"PERSON", "ORG"} else "OTHER"


def _page_context(parsed: ParsedDocument, page_no: int) -> str:
    for page in parsed.pages:
        if page.page_no == page_no:
            return _trim_context(page.text, limit=1100)
    return ""


def _build_review_items(parsed: ParsedDocument, findings: list[Finding]) -> list[FindingReviewItem]:
    items: list[FindingReviewItem] = []
    for index, finding in enumerate(findings, start=1):
        items.append(
            FindingReviewItem(
                review_id=f"r{index}",
                page_no=finding.page_no,
                entity_type=finding.entity_type,
                masked_value=finding.masked_value,
                page_context=_page_context(parsed, finding.page_no),
            )
        )
    return items


def _extract_review_response(payload: str | None) -> dict:
    data = _parse_json(payload)
    if isinstance(data, dict) and data:
        return data

    text = (payload or "").strip()
    if not text:
        return {}

    findings: list[dict[str, object]] = []
    review_ids = re.findall(r"r\d+", text, flags=re.IGNORECASE)
    labels = {
        "AADHAAR",
        "PAN",
        "EMAIL",
        "PHONE",
        "CREDIT_CARD",
        "BANK_ACCOUNT",
        "API_KEY",
        "PASSWORD",
        "EMPLOYEE_ID",
        "SESSION_TOKEN",
        "PRIVATE_KEY",
        "CONFIDENTIAL_BUSINESS_INFO",
        "OTHER",
    }
    for review_id in review_ids:
        window_start = max(0, text.lower().find(review_id.lower()) - 80)
        window_end = min(len(text), text.lower().find(review_id.lower()) + 140)
        window = text[window_start:window_end].upper()
        entity_type = next((label for label in labels if label in window), "OTHER")
        if entity_type == "OTHER":
            continue
        findings.append(
            {
                "review_id": review_id,
                "entity_type": entity_type,
                "confidence": 0.7,
            }
        )

    return {"findings": findings} if findings else {}


def _chunked(items: list[FindingReviewItem], size: int) -> list[list[FindingReviewItem]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _build_review_prompt(parsed: ParsedDocument, items: list[FindingReviewItem]) -> list[dict[str, str]]:
    blocks = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"review_id: {item.review_id}",
                    f"page_no: {item.page_no}",
                    f"entity_type: {item.entity_type}",
                    f"masked_value: {item.masked_value}",
                    f"page_context: {item.page_context or '[EMPTY]'}",
                ]
            )
        )

    return [
        {
            "role": "system",
            "content": (
                "You are a strict security and compliance reviewer.\n"
                "Review every provided finding using its page context.\n"
                "If a finding is an example, sample, demo, test, dummy, mock, fake, or placeholder value, label it OTHER.\n"
                "Be specific: never output vague labels like secret, token, credential, sensitive data, or confidential info.\n"
                "Classify findings using these exact labels only:\n"
                "AADHAAR, PAN, EMAIL, PHONE, CREDIT_CARD, BANK_ACCOUNT, API_KEY, PASSWORD, "
                "EMPLOYEE_ID, SESSION_TOKEN, PRIVATE_KEY, CONFIDENTIAL_BUSINESS_INFO, OTHER.\n"
                "If the snippet is confidential, proprietary, internal-only, non-public, pricing, roadmap, client-list, source code, architecture, or strategy information, label it CONFIDENTIAL_BUSINESS_INFO.\n"
                "If a finding looks like a password, use PASSWORD.\n"
                "If it looks like an API key, access token, auth token, or secret key, use API_KEY.\n"
                "If it looks like a session token or refresh token, use SESSION_TOKEN.\n"
                "If it looks like an employee or staff identifier, use EMPLOYEE_ID.\n"
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document ID: {parsed.document_id}\n"
                f"Pages: {len(parsed.pages)}\n"
                f"Findings to review: {len(items)}\n\n"
                "Return JSON in this shape:\n"
                "{\n"
                '  "summary": "short document-level compliance summary",\n'
                '  "findings": [\n'
                "    {\n"
                '      "review_id": "r1",\n'
                '      "page_no": 1,\n'
                '      "entity_type": "AADHAAR | PAN | EMAIL | PHONE | CREDIT_CARD | BANK_ACCOUNT | API_KEY | PASSWORD | EMPLOYEE_ID | SESSION_TOKEN | PRIVATE_KEY | CONFIDENTIAL_BUSINESS_INFO | OTHER",\n'
                '      "masked_value": "masked version only",\n'
                '      "confidence": 0.0\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Review these findings:\n"
                + "\n\n".join(blocks)
            ),
        },
    ]


def _llm_review_findings(parsed: ParsedDocument, session_id: str, base_findings: list[Finding]) -> list[Finding]:
    if not base_findings:
        return []

    items = _build_review_items(parsed, base_findings)
    if not items:
        return []

    findings: list[Finding] = []
    for batch in _chunked(items, 40):
        prompt = _build_review_prompt(parsed, batch)
        raw = chat_completion(prompt, temperature=0.0, max_tokens=1800)
        data = _extract_review_response(raw)
        raw_findings = data.get("findings", []) if isinstance(data, dict) else []
        review_map = {item.review_id: item for item in batch}

        if not isinstance(raw_findings, list):
            continue

        for item in raw_findings:
            if not isinstance(item, dict):
                continue

            review_id = str(item.get("review_id", "") or "")
            review_item = review_map.get(review_id)
            if review_item is None:
                continue

            raw_entity = str(item.get("entity_type", "OTHER") or "OTHER")
            entity_type = _normalize_entity_type(raw_entity, review_item.entity_type)
            if entity_type == "OTHER":
                continue

            try:
                confidence = float(item.get("confidence", 0.5) or 0.5)
            except Exception:
                confidence = 0.5

            findings.append(
                Finding(
                    id=str(uuid4()),
                    session_id=session_id,
                    document_id=parsed.document_id,
                    page_no=review_item.page_no,
                    entity_type=entity_type,
                    source="llm",
                    masked_value=review_item.masked_value,
                    confidence=max(0.0, min(confidence, 1.0)),
                    compliance_tags=[],
                    created_at=datetime.utcnow(),
                )
            )

    return findings


def _merge_findings(base: list[Finding], overlay: list[Finding]) -> list[Finding]:
    merged: dict[tuple[int, str, str], Finding] = {}
    for finding in base:
        key = (finding.page_no, finding.entity_type, finding.masked_value)
        merged[key] = finding

    for finding in overlay:
        key = (finding.page_no, finding.entity_type, finding.masked_value)
        existing = merged.get(key)
        if existing is None or existing.source != "llm":
            merged[key] = finding

    return sorted(merged.values(), key=lambda item: (item.page_no, item.entity_type, item.masked_value))


def detect_document(parsed: ParsedDocument, session_id: str) -> LLMDetectorResult:
    findings = detect_findings(parsed, session_id)
    llm_used = False
    if llm_enabled():
        llm_used = True
        llm_findings = _llm_review_findings(parsed, session_id, findings)
        findings = _merge_findings(findings, llm_findings)
        findings = [
            finding.model_copy(update={"source": "llm"})
            for finding in findings
        ]

    risk = score_findings(findings, document_id=parsed.document_id)

    if not llm_enabled():
        reason = llm_last_error() or "LLM is disabled or unavailable."
        return LLMDetectorResult(
            findings=findings,
            risk=risk,
            summary=f"LLM is offline; using direct regex/NER detection only. Reason: {reason}",
            llm_message=f"LLM does not work. Reason: {reason}",
        )

    findings_text = "\n".join(
        f"- page {f.page_no}: {f.entity_type} ({f.masked_value})"
        for f in findings[:20]
    )
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a security and compliance analyst. "
                "Write a short, general compliance summary from the findings only. "
                "Do not repeat raw sensitive values."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Risk score: {risk.score} ({risk.bucket})\n"
                f"Risk explanation: {risk.explanation}\n\n"
                f"Findings:\n{findings_text or 'None'}\n\n"
                "Return a concise 3-4 sentence summary with overall compliance impact and next steps."
            ),
        },
    ]
    raw = chat_completion(prompt, temperature=0.1, max_tokens=220)
    if raw is None or not raw.strip():
        reason = llm_last_error() or "LLM call returned no usable response."
        return LLMDetectorResult(
            findings=findings,
            risk=risk,
            summary=f"LLM is offline; fell back to full regex/NER detection. Reason: {reason}",
            llm_message=f"LLM does not work. Reason: {reason}",
        )

    summary = raw.strip()
    if not summary:
        summary = risk.explanation

    return LLMDetectorResult(
        findings=findings,
        risk=risk,
        summary=summary,
        llm_message="LLM works." if llm_used else "LLM does not work. Reason: LLM review was not used.",
    )
