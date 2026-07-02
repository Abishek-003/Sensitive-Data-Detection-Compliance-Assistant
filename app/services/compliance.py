from __future__ import annotations

import re
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from uuid import uuid4

from app.models.schemas import Finding, ParsedDocument, RiskSummary
from app.services.llm import chat_completion


@dataclass(frozen=True)
class PatternRule:
    entity_type: str
    pattern: re.Pattern[str]
    source: str = "regex"
    confidence: float = 0.95


DISPLAY_ENTITY_LABELS: dict[str, str] = {
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
    "PERSON": "Person Name",
    "ORG": "Organization Name",
    "IP_ADDRESS": "IP Address",
}

ALLOWED_ENTITY_TYPES: set[str] = {
    "AADHAAR",
    "PAN",
    "EMAIL",
    "PHONE",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "EMPLOYEE_ID",
    "API_KEY",
    "PASSWORD",
    "SESSION_TOKEN",
    "PRIVATE_KEY",
    "CONFIDENTIAL_BUSINESS_INFO",
    "PERSON",
    "ORG",
    "IP_ADDRESS",
    "OTHER",
}

RISK_WEIGHTS: dict[str, float] = {
    "PASSWORD": 10.0,
    "API_KEY": 9.0,
    "BANK_ACCOUNT": 7.5,
    "CREDIT_CARD": 8.0,
    "AADHAAR": 7.0,
    "PAN": 5.0,
    "EMPLOYEE_ID": 5.0,
    "SESSION_TOKEN": 9.0,
    "PRIVATE_KEY": 10.0,
    "CONFIDENTIAL_BUSINESS_INFO": 6.0,
    "SUSPICIOUS_SECRET": 8.0,
    "SUSPICIOUS_TOKEN": 8.5,
    "SUSPICIOUS_IDENTIFIER": 4.0,
    "SUSPICIOUS_DATA": 5.0,
    "EMAIL": 2.0,
    "PHONE": 2.0,
    "PERSON": 1.5,
    "ORG": 1.0,
    "IP_ADDRESS": 1.0,
}

PATTERN_RULES: list[PatternRule] = [
    PatternRule("EMAIL", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")),
    PatternRule("AADHAAR", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    PatternRule("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), confidence=0.96),
    PatternRule(
        "PHONE",
        re.compile(
            r"(?ix)"
            r"(?<!\d)"
            r"(?:\+?91[\s\-()]*)?"
            r"(?:0[\s\-()]*)?"
            r"(?:"
            r"\d{10}"
            r"|(?:\d[\s\-()]*){10}"
            r")"
            r"(?!\d)"
        ),
    ),
    PatternRule(
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        confidence=0.9,
    ),
    PatternRule("BANK_ACCOUNT", re.compile(r"\b\d{9,18}\b"), confidence=0.85),
    PatternRule(
        "API_KEY",
        re.compile(r"(?i)\b(?:temp_)?(?:api[_-]?token|secret[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*([^\s,;\"']{8,})"),
        confidence=0.98,
    ),
    PatternRule(
        "API_KEY",
        re.compile(r"(?i)\b(?:api[_-]?key|token|secret)\b\s*[:=]\s*([A-Za-z0-9._\-\/+=]{10,})"),
        confidence=0.96,
    ),
    PatternRule(
        "API_KEY",
        re.compile(r"(?i)\bsk-or-v1-[A-Za-z0-9]{20,}\b"),
        confidence=0.99,
    ),
    PatternRule(
        "API_KEY",
        re.compile(r"(?i)\b(?:sk|rk|pk)[-_]?[A-Za-z0-9_-]{20,}\b"),
        confidence=0.97,
    ),
    PatternRule(
        "PASSWORD",
        re.compile(r"(?i)\bpassword\s*[:=]\s*([^\s,;]+)"),
        confidence=0.92,
    ),
    PatternRule(
        "PASSWORD",
        re.compile(r"(?i)\b(?:db[_-]?password|secret[_-]?password|passcode)\b\s*[:=]\s*([^\s,;]+)"),
        confidence=0.93,
    ),
    PatternRule("EMPLOYEE_ID", re.compile(r"\bEMP[- ]?\d{3,}\b"), confidence=0.88),
    PatternRule(
        "EMPLOYEE_ID",
        re.compile(r"(?i)\b(?:employee\s*id|emp\s*id|staff\s*id)\b\s*[:#-]?\s*([A-Z0-9-]{4,})"),
        confidence=0.88,
    ),
    PatternRule(
        "EMPLOYEE_ID",
        re.compile(r"(?i)\b(?:emp|eid|empid|staff|member|user)[-_ ]?\d{3,}\b"),
        confidence=0.9,
    ),
    PatternRule("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), confidence=0.7),
    PatternRule(
        "SESSION_TOKEN",
        re.compile(r"(?i)\b(?:session[_-]?token|refresh[_-]?token|bearer[_-]?token)\b\s*[:=]\s*([A-Za-z0-9._\-\/+=]{12,})"),
        confidence=0.93,
    ),
    PatternRule(
        "PRIVATE_KEY",
        re.compile(r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
        confidence=0.99,
    ),
    PatternRule(
        "CONFIDENTIAL_BUSINESS_INFO",
        re.compile(r"(?i)\b(?:confidential|proprietary|internal only|do not share|restricted|trade secret|non-public|pricing|client list|roadmap|forecast|revenue|strategy|merger|acquisition|source code|architecture|blueprint)\b"),
        confidence=0.8,
    ),
]

BANK_ACCOUNT_HINTS = (
    "account",
    "a/c",
    "acct",
    "bank",
    "savings",
    "current",
    "ifsc",
    "neft",
    "rtgs",
    "imps",
)

ENTITY_CONTEXT_HINTS: dict[str, tuple[str, ...]] = {
    "PHONE": (
        "phone",
        "mobile",
        "contact",
        "phone number",
        "mobile number",
        "contact number",
        "ph no",
        "mob no",
        "telephone",
        "tel",
        "call",
        "whatsapp",
        "example",
        "sample",
        "demo",
        "test",
        "placeholder",
    ),
    "AADHAAR": (
        "aadhaar",
        "aadhar",
        "uid",
        "government id",
    ),
    "PAN": (
        "pan",
        "permanent account number",
        "income tax",
        "tax id",
    ),
    "BANK_ACCOUNT": BANK_ACCOUNT_HINTS,
    "CREDIT_CARD": (
        "credit card",
        "card number",
        "debit card",
        "cvv",
        "expiry",
        "payment",
    ),
    "EMPLOYEE_ID": (
        "employee id",
        "emp id",
        "staff id",
        "employee number",
    ),
    "API_KEY": (
        "api key",
        "token",
        "secret",
        "access token",
        "auth token",
    ),
    "PASSWORD": (
        "password",
        "passwd",
        "passcode",
    ),
    "SESSION_TOKEN": (
        "session token",
        "refresh token",
        "bearer token",
    ),
    "PRIVATE_KEY": (
        "private key",
        "begin private key",
        "end private key",
    ),
    "CONFIDENTIAL_BUSINESS_INFO": (
        "confidential",
        "proprietary",
        "internal only",
        "do not share",
        "restricted",
        "trade secret",
        "non-public",
        "pricing",
        "client list",
        "roadmap",
        "forecast",
        "revenue",
        "strategy",
        "merger",
        "acquisition",
    ),
}


def _mask_value(entity_type: str, value: str) -> str:
    cleaned = value.strip()
    if entity_type in {"EMAIL"}:
        name, _, domain = cleaned.partition("@")
        if not domain:
            return "***"
        return f"{name[:2]}***@{domain}"
    if entity_type == "PHONE":
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) >= 4:
            return f"{digits[:2]}***{digits[-2:]}"
        return "***MASKED***"
    if entity_type == "AADHAAR":
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) >= 8:
            return f"{digits[:4]}****{digits[-4:]}"
        return "***MASKED***"
    if entity_type == "PAN":
        letters = "".join(ch for ch in cleaned if ch.isalnum()).upper()
        if len(letters) >= 6:
            return f"{letters[:5]}****{letters[-1]}"
        return "***MASKED***"
    if entity_type == "CREDIT_CARD":
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) >= 8:
            return f"{digits[:4]}********{digits[-4:]}"
        return "***MASKED***"
    if entity_type == "BANK_ACCOUNT":
        digits = re.sub(r"\D", "", cleaned)
        if len(digits) >= 8:
            return f"{digits[:4]}******{digits[-4:]}"
        if len(digits) >= 3:
            return f"{digits[:2]}***{digits[-1]}"
        return "***MASKED***"
    if entity_type in {"PASSWORD", "API_KEY"}:
        lowered = cleaned.lower()
        if entity_type == "API_KEY":
            if lowered.startswith("sk-proj-"):
                return "sk-proj-********"
            if lowered.startswith(("sk-", "rk-", "pk-")):
                return f"{cleaned[:3]}********"
            if lowered.startswith(("ghp_", "gho_", "ghu_", "ghs_", "ghr_")):
                return f"{cleaned[:4]}********"
            if lowered.startswith("akia") or lowered.startswith("asia"):
                return f"{cleaned[:4]}********"
            if lowered.startswith("eyj"):
                return "eyJ********"
        return "***REDACTED***"
    if entity_type in {"SESSION_TOKEN", "PRIVATE_KEY"}:
        return "***SECRET***"
    if entity_type == "CONFIDENTIAL_BUSINESS_INFO":
        return "***CONFIDENTIAL***"
    return cleaned[:2] + "***" if len(cleaned) > 2 else "***"


def _luhn_check(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _is_bank_account_candidate(text: str, match: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    if not (9 <= len(digits) <= 18):
        return False

    if re.fullmatch(r"(?:\+91)?[6-9]\d{9}", digits):
        return False
    if re.fullmatch(r"\d{12}", digits):
        return False

    window_start = max(0, match.start() - 40)
    window_end = min(len(text), match.end() + 40)
    window = text[window_start:window_end].lower()
    return any(hint in window for hint in BANK_ACCOUNT_HINTS)


def _has_context_hint(text: str, match: re.Match[str], entity_type: str) -> bool:
    hints = ENTITY_CONTEXT_HINTS.get(entity_type)
    if not hints:
        return True

    line_start = text.rfind("\n", 0, match.start())
    line_end = text.find("\n", match.end())
    if line_start < 0:
        line_start = 0
    else:
        line_start += 1
    if line_end < 0:
        line_end = len(text)

    window_start = max(0, line_start - 60)
    window_end = min(len(text), line_end + 60)
    window = text[window_start:window_end].lower()
    return any(hint in window for hint in hints)


def _is_phone_candidate(text: str, match: re.Match[str]) -> bool:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)

    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    if len(digits) != 10:
        return False
    window = text[max(0, match.start() - 20) : min(len(text), match.end() + 20)].lower()
    if any(hint in window for hint in ("account", "ifsc", "acct", "bank")):
        return False
    return any(hint in window for hint in ENTITY_CONTEXT_HINTS["PHONE"])


def _looks_like_pan(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value.strip().upper()))


def _normalize_entity_type(entity_type: str) -> str:
    value = (entity_type or "").strip().upper()
    alias_map = {
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
    return "OTHER"


def display_entity_label(entity_type: str) -> str:
    normalized = _normalize_entity_type(entity_type)
    return DISPLAY_ENTITY_LABELS.get(normalized, normalized.replace("_", " ").title())


def normalize_entity_type(entity_type: str) -> str:
    return _normalize_entity_type(entity_type)


def _normalize_detection_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"(\w)\s*\n\s*(\w)", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _candidate_texts(text: str) -> list[str]:
    normalized = _normalize_detection_text(text)
    if normalized == text:
        return [text]
    return [text, normalized]


@lru_cache(maxsize=1)
def _get_spacy_nlp():
    try:
        import spacy
    except Exception:
        return None

    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _yield_regex_findings(
    session_id: str,
    document_id: str,
    page_no: int,
    text: str,
) -> Iterable[Finding]:
    seen: set[tuple[str, str]] = set()
    for candidate_text in _candidate_texts(text):
        for rule in PATTERN_RULES:
            for match in rule.pattern.finditer(candidate_text):
                raw = match.group(0)
                if rule.entity_type in {"PASSWORD", "API_KEY", "EMPLOYEE_ID", "SESSION_TOKEN"} and match.groups():
                    raw = match.group(1)
                if rule.entity_type == "CONFIDENTIAL_BUSINESS_INFO" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                if rule.entity_type == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", raw)
                    if not _luhn_check(digits):
                        continue
                    if not _has_context_hint(candidate_text, match, rule.entity_type):
                        continue
                if rule.entity_type == "PAN" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                if rule.entity_type == "PHONE" and not _is_phone_candidate(candidate_text, match):
                    continue
                if rule.entity_type == "AADHAAR" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                if rule.entity_type == "BANK_ACCOUNT" and not _is_bank_account_candidate(candidate_text, match):
                    continue
                if rule.entity_type == "EMPLOYEE_ID" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                if rule.entity_type == "API_KEY" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                if rule.entity_type == "PASSWORD" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                if rule.entity_type == "SESSION_TOKEN" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                if rule.entity_type == "PRIVATE_KEY" and not _has_context_hint(candidate_text, match, rule.entity_type):
                    continue
                normalized_type = _normalize_entity_type(rule.entity_type)
                masked = _mask_value(normalized_type, raw)
                key = (normalized_type, masked)
                if key in seen:
                    continue
                seen.add(key)
                yield Finding(
                    id=str(uuid4()),
                    session_id=session_id,
                    document_id=document_id,
                    page_no=page_no,
                    entity_type=normalized_type,
                    source=rule.source,
                    masked_value=masked,
                    confidence=rule.confidence,
                    compliance_tags=[],
                    created_at=datetime.utcnow(),
                )


def _yield_ner_findings(
    session_id: str,
    document_id: str,
    page_no: int,
    text: str,
) -> Iterable[Finding]:
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    seen: set[tuple[str, str]] = set()
    for ent in doc.ents:
        entity_type = ent.label_
        if entity_type not in {"PERSON", "ORG", "GPE"}:
            continue
        normalized_type = _normalize_entity_type(entity_type)
        masked = _mask_value(normalized_type, ent.text)
        key = (normalized_type, masked)
        if key in seen:
            continue
        seen.add(key)
        yield Finding(
            id=str(uuid4()),
            session_id=session_id,
            document_id=document_id,
            page_no=page_no,
            entity_type=normalized_type,
            source="ner",
            masked_value=masked,
            confidence=0.6,
            compliance_tags=[],
            created_at=datetime.utcnow(),
        )


def detect_findings(parsed: ParsedDocument, session_id: str) -> list[Finding]:
    findings: list[Finding] = []
    dedupe: set[tuple[int, str, str]] = set()

    for page in parsed.pages:
        for finding in _yield_regex_findings(session_id, parsed.document_id, page.page_no, page.text):
            key = (finding.page_no, finding.entity_type, finding.masked_value)
            if key not in dedupe:
                dedupe.add(key)
                findings.append(finding)

    nlp = _get_spacy_nlp()
    if nlp is not None:
        for page, doc in zip(parsed.pages, nlp.pipe((p.text for p in parsed.pages), batch_size=8)):
            for ent in doc.ents:
                entity_type = ent.label_
                if entity_type not in {"PERSON", "ORG", "GPE"}:
                    continue
                normalized_type = _normalize_entity_type(entity_type)
                masked = _mask_value(normalized_type, ent.text)
                key = (page.page_no, normalized_type, masked)
                if key in dedupe:
                    continue
                dedupe.add(key)
                findings.append(
                    Finding(
                        id=str(uuid4()),
                        session_id=session_id,
                        document_id=parsed.document_id,
                        page_no=page.page_no,
                        entity_type=normalized_type,
                        source="ner",
                        masked_value=masked,
                        confidence=0.6,
                        compliance_tags=[],
                        created_at=datetime.utcnow(),
                    )
                )

    return findings


def score_findings(findings: list[Finding], document_id: str) -> RiskSummary:
    if not findings:
        return RiskSummary(
            document_id=document_id,
            score=0.0,
            bucket="low",
            explanation="No sensitive findings were detected.",
            compliance_tags=[],
        )

    score = 0.0
    counts: dict[str, int] = {}
    for finding in findings:
        score += RISK_WEIGHTS.get(finding.entity_type, 1.0)
        counts[finding.entity_type] = counts.get(finding.entity_type, 0) + 1

    if score >= 20:
        bucket = "high"
    elif score >= 8:
        bucket = "medium"
    else:
        bucket = "low"

    top = ", ".join(
        f"{display_entity_label(entity)} x{count}"
        for entity, count in sorted(counts.items(), key=lambda i: (-i[1], i[0]))[:4]
    )
    explanation = f"Detected {top}." if top else "Detected sensitive content."

    return RiskSummary(
        document_id=document_id,
        score=round(score, 2),
        bucket=bucket,
        explanation=explanation,
        compliance_tags=[],
    )


def generate_ai_compliance_summary(
    findings: list[Finding],
    risk: RiskSummary,
) -> str | None:
    if not findings:
        return "No sensitive data findings were detected, so no remediation is needed."

    findings_text = "\n".join(
        f"- page {f.page_no}: {display_entity_label(f.entity_type)} ({f.masked_value})"
        for f in findings[:25]
    )
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a security and compliance analyst. "
                "Write concise, practical compliance guidance based only on the findings provided."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Risk score: {risk.score} ({risk.bucket})\n"
                f"Risk explanation: {risk.explanation}\n"
                f"Findings:\n{findings_text}\n\n"
                "Return 4 short bullets: executive summary, compliance observations, "
                "security risks, remediation suggestions."
            ),
        },
    ]
    return chat_completion(prompt, temperature=0.2, max_tokens=350)
