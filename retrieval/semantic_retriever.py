from typing import List

from langchain_core.documents import Document

from vectorstore.chroma_store import ChromaVectorStore
from utils.logger import get_logger


logger = get_logger(__name__)


class SemanticRetriever:

    def __init__(self):

        self.vector_store = ChromaVectorStore()

        logger.info(
            "Semantic retriever initialized."
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        policy_type: str | None = None
    ) -> List[Document]:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        logger.info(
            f"Retrieving documents for query: "
            f"{query}"
        )

        if policy_type:

            logger.info(
                f"Policy filter: {policy_type}"
            )

            documents = (
                self.vector_store
                .similarity_search_with_filter(
                    query=query,
                    k=top_k,
                    policy_type=policy_type
                )
            )

        else:

            documents = (
                self.vector_store
                .similarity_search(
                    query=query,
                    k=top_k
                )
            )

        logger.info(
            f"Retrieved {len(documents)} documents."
        )

        return documents