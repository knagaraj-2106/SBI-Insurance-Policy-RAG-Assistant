from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger


logger = get_logger(__name__)


class RetrievalEvaluator:

    def __init__(self):

        logger.info(
            "Retrieval evaluator initialized."
        )

    def evaluate(
        self,
        documents: List[Document],
        expected_keywords: List[str]
    ) -> float:

        if not documents:

            logger.warning(
                "No documents retrieved."
            )

            return 0.0

        if not expected_keywords:

            return 1.0

        combined_content = " ".join(
            document.page_content
            for document in documents
        ).lower()

        matched = 0

        for keyword in expected_keywords:

            if keyword.lower() in combined_content:

                matched += 1

        score = matched / len(
            expected_keywords
        )

        logger.info(
            f"Retrieval evaluation score: "
            f"{score:.2f}"
        )

        return score