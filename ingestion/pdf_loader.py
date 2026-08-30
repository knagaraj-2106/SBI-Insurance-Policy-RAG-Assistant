from pathlib import Path
from typing import List

import fitz
from langchain_core.documents import Document

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class PDFLoader:

    def __init__(self, data_directory: Path | None = None):

        self.data_directory = (
            data_directory
            if data_directory
            else settings.RAW_DATA_DIR
        )

    def discover_pdfs(self) -> List[Path]:

        if not self.data_directory.exists():

            logger.warning(
                f"Data directory does not exist: "
                f"{self.data_directory}"
            )

            return []

        pdf_files = list(
            self.data_directory.rglob("*.pdf")
        )

        logger.info(
            f"Discovered {len(pdf_files)} PDF file(s)."
        )

        return pdf_files

    def load_pdf(self, pdf_path: Path) -> List[Document]:

        documents = []

        try:

            logger.info(
                f"Processing PDF: {pdf_path.name}"
            )

            pdf = fitz.open(pdf_path)

            policy_type = pdf_path.parent.name

            for page_number, page in enumerate(
                pdf,
                start=1
            ):

                text = page.get_text("text")

                if not text.strip():

                    logger.warning(
                        f"Empty page detected: "
                        f"{pdf_path.name} - "
                        f"Page {page_number}"
                    )

                    continue

                document = Document(

                    page_content=text,

                    metadata={
                        "source": str(pdf_path),
                        "document_name": pdf_path.name,
                        "policy_type": policy_type,
                        "page_number": page_number,
                    }
                )

                documents.append(document)

            pdf.close()

            logger.info(
                f"Loaded {len(documents)} page(s) "
                f"from {pdf_path.name}"
            )

        except Exception as exc:

            logger.error(
                f"Failed to process "
                f"{pdf_path}: {exc}"
            )

        return documents

    def load_all_documents(self) -> List[Document]:

        all_documents = []

        pdf_files = self.discover_pdfs()

        for pdf_path in pdf_files:

            documents = self.load_pdf(pdf_path)

            all_documents.extend(documents)

        logger.info(
            f"Total LangChain documents created: "
            f"{len(all_documents)}"
        )

        return all_documents