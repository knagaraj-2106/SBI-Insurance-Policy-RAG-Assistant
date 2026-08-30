from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from embeddings.openai_embeddings import EmbeddingService
from utils.logger import get_logger


logger = get_logger(__name__)


class ChromaVectorStore:

    def __init__(self):

        self.embedding_service = EmbeddingService()

        self.vector_store = Chroma(
            collection_name="sbi_insurance_policies",
            embedding_function=(
                self.embedding_service.embeddings
            ),
            persist_directory=str(
                settings.VECTORSTORE_DIR
            )
        )

        logger.info(
            "ChromaDB vector store initialized."
        )

    @staticmethod
    def generate_document_id(
        document: Document,
        chunk_index: int
    ) -> str:

        source = document.metadata.get(
            "source",
            "unknown_source"
        )

        page_number = document.metadata.get(
            "page_number",
            "unknown_page"
        )

        return (
            f"{source}"
            f"_page_{page_number}"
            f"_chunk_{chunk_index}"
        )

    def add_documents(
        self,
        documents: List[Document]
    ):

        if not documents:

            logger.warning(
                "No documents supplied for vectorization."
            )

            return

        ids = []

        for index, document in enumerate(documents):

            document_id = self.generate_document_id(
                document,
                index
            )

            ids.append(document_id)

            document.metadata[
                "document_id"
            ] = document_id

        logger.info(
            f"Indexing {len(documents)} chunks "
            "into ChromaDB."
        )

        self.vector_store.add_documents(
            documents=documents,
            ids=ids
        )

        logger.info(
            "Documents successfully indexed "
            "in ChromaDB."
        )

    def similarity_search(
        self,
        query: str,
        k: int | None = None
    ) -> List[Document]:

        top_k = k or settings.TOP_K

        logger.info(
            f"Performing similarity search. "
            f"Top K: {top_k}"
        )

        results = self.vector_store.similarity_search(
            query,
            k=top_k
        )

        logger.info(
            f"Retrieved {len(results)} documents."
        )

        return results
    def similarity_search_with_filter(
        self,
        query: str,
        k: int = 5,
        policy_type: str | None = None
    ) -> List[Document]:

        logger.info(
            f"Performing filtered similarity search. "
            f"Query: {query}"
        )

        filter_condition = None

        if policy_type:

            filter_condition = {
                "policy_type": policy_type
            }

            logger.info(
                f"Applying policy filter: "
                f"{policy_type}"
            )

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_condition
        )

        logger.info(
            f"Retrieved {len(results)} documents "
            f"after filtering."
        )

        return results

    def get_collection_count(self) -> int:

        count = self.vector_store._collection.count()

        logger.info(
            f"ChromaDB collection contains "
            f"{count} documents."
        )

        return count