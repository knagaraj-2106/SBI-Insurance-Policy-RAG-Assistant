from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from chunking.text_cleaner import TextCleaner
from utils.logger import get_logger


logger = get_logger(__name__)


class TextChunker:

    def __init__(self):

        self.cleaner = TextCleaner()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def clean_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:

        cleaned_documents = []

        for document in documents:

            cleaned_text = self.cleaner.clean(
                document.page_content
            )

            if not cleaned_text:
                continue

            cleaned_document = Document(
                page_content=cleaned_text,
                metadata=document.metadata.copy()
            )

            cleaned_documents.append(
                cleaned_document
            )

        logger.info(
            f"Cleaned documents: "
            f"{len(cleaned_documents)}"
        )

        return cleaned_documents

    def create_chunks(
        self,
        documents: List[Document]
    ) -> List[Document]:

        cleaned_documents = self.clean_documents(
            documents
        )

        chunks = self.splitter.split_documents(
            cleaned_documents
        )

        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_id"] = (
                f"chunk_{index:06d}"
            )

        logger.info(
            f"Created {len(chunks)} chunks "
            f"from {len(documents)} page documents."
        )

        return chunks