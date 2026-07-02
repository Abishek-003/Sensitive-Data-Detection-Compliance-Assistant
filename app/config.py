from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH = BASE_DIR / "app.db"
TMP_DIR = BASE_DIR / "tmp"
CHROMA_DIR = BASE_DIR / "chromadb"

DEFAULT_MODE = "single"

load_dotenv(BASE_DIR / ".env")

try:
    import streamlit as st
except Exception:
    st = None


def _setting(name: str, default: str) -> str:
    value = os.getenv(name)
    if value not in {None, ""}:
        return value
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass
    return default


SESSION_TTL_MINUTES = int(_setting("SESSION_TTL_MINUTES", "10"))
SESSION_CLEANUP_PORT = int(_setting("SESSION_CLEANUP_PORT", "8765"))

ENABLE_LLM = _setting("ENABLE_LLM", "1") not in {"0", "false", "False"}
LLM_PROVIDER = _setting("LLM_PROVIDER", "langchain").strip().lower()
LLM_API_KEY = _setting("LLM_API_KEY", "")
LLM_BASE_URL = _setting("LLM_BASE_URL", "")
LLM_MODEL = _setting("LLM_MODEL", "")
LLM_APP_NAME = _setting("LLM_APP_NAME", "")
LLM_APP_URL = _setting("LLM_APP_URL", "")

TMP_DIR.mkdir(parents=True, exist_ok=True)
