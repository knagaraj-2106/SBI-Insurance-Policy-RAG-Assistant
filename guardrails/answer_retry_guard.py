from utils.logger import get_logger


logger = get_logger(__name__)


class AnswerRetryGuard:

    def __init__(
        self,
        max_retries: int = 1
    ):

        self.max_retries = max_retries

        logger.info(
            f"Answer retry guard initialized. "
            f"Maximum retries: {max_retries}"
        )

    def should_retry(
        self,
        grounding_passed: bool,
        groundedness_passed: bool,
        retry_count: int
    ) -> bool:

        if retry_count >= self.max_retries:

            logger.info(
                "Maximum answer retry limit reached."
            )

            return False

        if grounding_passed and groundedness_passed:

            logger.info(
                "Answer passed all validation. "
                "Retry not required."
            )

            return False

        logger.warning(
            "Answer validation failed. "
            "Retry is required."
        )

        return True