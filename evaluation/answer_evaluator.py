from typing import List

from utils.logger import get_logger


logger = get_logger(__name__)


class AnswerEvaluator:

    def __init__(self):

        logger.info(
            "Answer evaluator initialized."
        )

    def keyword_match_score(
        self,
        answer: str,
        expected_keywords: List[str]
    ) -> float:

        if not answer:

            return 0.0

        if not expected_keywords:

            return 1.0

        answer_lower = answer.lower()

        matched = 0

        for keyword in expected_keywords:

            if keyword.lower() in answer_lower:

                matched += 1

        score = matched / len(
            expected_keywords
        )

        logger.info(
            f"Keyword evaluation score: "
            f"{score:.2f}"
        )

        return score