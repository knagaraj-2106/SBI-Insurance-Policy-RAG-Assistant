import os
from pathlib import Path


class Settings:

    # ==================================================
    # Project Paths
    # ==================================================

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"

    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    VECTORSTORE_DIR = BASE_DIR / "chroma_db"

    PROMPTS_DIR = BASE_DIR / "prompts"

    # ==================================================
    # OpenAI Configuration
    # ==================================================

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    LLM_MODEL = "gpt-4o-mini"

    EMBEDDING_MODEL = "text-embedding-3-small"

    # ==================================================
    # Chunking Configuration
    # ==================================================

    CHUNK_SIZE = 1000

    CHUNK_OVERLAP = 150

    # ==================================================
    # Retrieval Configuration
    # ==================================================

    TOP_K = 5

    # ==================================================
    # Application Configuration
    # ==================================================

    APP_TITLE = "SBI Insurance Policy RAG Assistant"

    APP_DESCRIPTION = (
        "AI-powered assistant for querying SBI insurance policy documents."
    )


settings = Settings()