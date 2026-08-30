from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class EmbeddingService:

    def __init__(self):

        if not settings.OPENAI_API_KEY:

            raise ValueError(
                "OPENAI_API_KEY is not available."
            )

        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL
        )

        logger.info(
            f"Embedding service initialized "
            f"with model: {settings.EMBEDDING_MODEL}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        if not texts:

            return []

        logger.info(
            f"Generating embeddings for "
            f"{len(texts)} documents."
        )

        vectors = self.embeddings.embed_documents(
            texts
        )

        logger.info(
            f"Generated {len(vectors)} embeddings."
        )

        return vectors

    def embed_query(self, query: str) -> list[float]:

        if not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        return self.embeddings.embed_query(
            query
        )