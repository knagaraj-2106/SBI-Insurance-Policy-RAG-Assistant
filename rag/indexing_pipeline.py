from ingestion.pdf_loader import PDFLoader
from chunking.text_chunker import TextChunker
from vectorstore.chroma_store import ChromaVectorStore
from utils.logger import get_logger


logger = get_logger(__name__)


class IndexingPipeline:

    def __init__(self):

        self.pdf_loader = PDFLoader()

        self.text_chunker = TextChunker()

        self.vector_store = ChromaVectorStore()

    def run(self):

        logger.info(
            "Starting SBI Insurance indexing pipeline."
        )

        # ------------------------------------------
        # 1. Load PDF documents
        # ------------------------------------------

        documents = (
            self.pdf_loader.load_all_documents()
        )

        logger.info(
            f"Loaded {len(documents)} page documents."
        )

        if not documents:

            raise ValueError(
                "No documents found for indexing."
            )

        # ------------------------------------------
        # 2. Clean + chunk
        # ------------------------------------------

        chunks = (
            self.text_chunker.create_chunks(
                documents
            )
        )

        logger.info(
            f"Created {len(chunks)} chunks."
        )

        if not chunks:

            raise ValueError(
                "No chunks generated."
            )

        # ------------------------------------------
        # 3. Store in ChromaDB
        # ------------------------------------------

        self.vector_store.add_documents(
            chunks
        )

        # ------------------------------------------
        # 4. Verify collection
        # ------------------------------------------

        total_vectors = (
            self.vector_store
            .get_collection_count()
        )

        logger.info(
            f"ChromaDB now contains "
            f"{total_vectors} vectors."
        )

        logger.info(
            "Indexing pipeline completed successfully."
        )

        return {
            "pages": len(documents),
            "chunks": len(chunks),
            "vectors": total_vectors
        }