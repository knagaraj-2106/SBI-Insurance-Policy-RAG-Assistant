from typing import List

from langchain_core.documents import Document

from utils.logger import get_logger


logger = get_logger(__name__)


class RelevanceGuard:

    def __init__(
        self,
        minimum_score: int = 5,
        minimum_relevant_documents: int = 1,
        minimum_average_score: float = 5.0
    ):

        self.minimum_score = minimum_score
        self.minimum_relevant_documents = minimum_relevant_documents
        self.minimum_average_score = minimum_average_score

        logger.info(
            "Relevance guard initialized. "
            f"Minimum score: {minimum_score}, "
            f"Minimum relevant documents: "
            f"{minimum_relevant_documents}, "
            f"Minimum average score: "
            f"{minimum_average_score}"
        )

    def validate(
        self,
        documents: List[Document]
    ) -> bool:

        # -------------------------------------------------
        # 1. No documents
        # -------------------------------------------------

        if not documents:

            logger.warning(
                "No documents available for relevance validation."
            )

            return False

        # -------------------------------------------------
        # 2. Extract reranking scores
        # -------------------------------------------------

        scores = []

        for document in documents:

            score = document.metadata.get(
                "rerank_score"
            )

            if score is None:
                continue

            try:

                scores.append(
                    float(score)
                )

            except (TypeError, ValueError):

                logger.warning(
                    f"Invalid rerank score found: {score}"
                )

        # -------------------------------------------------
        # 3. No valid scores
        # -------------------------------------------------

        if not scores:

            logger.warning(
                "No valid rerank scores found."
            )

            return False

        # -------------------------------------------------
        # 4. Calculate relevance metrics
        # -------------------------------------------------

        highest_score = max(scores)

        average_score = sum(scores) / len(scores)

        relevant_documents = [
            score
            for score in scores
            if score >= self.minimum_score
        ]

        relevant_document_count = len(
            relevant_documents
        )

        # -------------------------------------------------
        # 5. Logging
        # -------------------------------------------------

        logger.info(
            f"Highest rerank score: {highest_score}"
        )

        logger.info(
            f"Average rerank score: {average_score:.2f}"
        )

        logger.info(
            f"Relevant documents: "
            f"{relevant_document_count}/{len(scores)}"
        )

        # -------------------------------------------------
        # 6. Check highest score
        # -------------------------------------------------

        if highest_score < self.minimum_score:

            logger.warning(
                "Relevance validation failed: "
                "highest score is below threshold."
            )

            return False

        # -------------------------------------------------
        # 7. Check number of relevant documents
        # -------------------------------------------------

        if (
            relevant_document_count
            < self.minimum_relevant_documents
        ):

            logger.warning(
                "Relevance validation failed: "
                "insufficient relevant documents."
            )

            return False

        # -------------------------------------------------
        # 8. Check average relevance
        # -------------------------------------------------

        if average_score < self.minimum_average_score:

            logger.warning(
                "Relevance validation failed: "
                "average relevance score is below threshold."
            )

            return False

        # -------------------------------------------------
        # 9. Passed
        # -------------------------------------------------

        logger.info(
            "Retrieved documents passed "
            "relevance validation."
        )

        return True