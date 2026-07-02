from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_DB_PATH = BASE_DIR / "app.db"
TMP_DIR = BASE_DIR / "tmp"
CHROMA_DIR = BASE_DIR / "chromadb"

DEFAULT_MODE = "single"
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "10"))
SESSION_CLEANUP_PORT = int(os.getenv("SESSION_CLEANUP_PORT", "8765"))

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Sensitive RAG + Compliance")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "http://localhost:8501")
ENABLE_LLM = os.getenv("ENABLE_LLM", "1") not in {"0", "false", "False"}
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "langchain").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENROUTER_API_KEY)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", OPENROUTER_BASE_URL)
LLM_MODEL = os.getenv("LLM_MODEL", OPENROUTER_MODEL)
LLM_APP_NAME = os.getenv("LLM_APP_NAME", OPENROUTER_APP_NAME)
LLM_APP_URL = os.getenv("LLM_APP_URL", OPENROUTER_APP_URL)

TMP_DIR.mkdir(parents=True, exist_ok=True)
