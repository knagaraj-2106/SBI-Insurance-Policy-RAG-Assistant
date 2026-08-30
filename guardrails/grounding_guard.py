import json

from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class GroundingGuard:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0
        )

        logger.info(
            "Grounding guard initialized."
        )

    def validate(
        self,
        answer: str,
        context: str
    ) -> bool:

        if not answer or not answer.strip():

            logger.warning(
                "Grounding validation failed: "
                "empty answer."
            )

            return False

        if not context or not context.strip():

            logger.warning(
                "Grounding validation failed: "
                "empty context."
            )

            return False

        prompt = f"""
You are a grounding validator for an
insurance policy RAG system.

Your task is to determine whether the
GENERATED ANSWER is fully supported by
the PROVIDED CONTEXT.

Rules:

1. The answer must be supported by the context.
2. Do not use outside knowledge.
3. If the answer contains information that
   is not supported by the context, return false.
4. If the answer is fully supported, return true.
5. Return ONLY valid JSON.
6. Do not use markdown.
7. Do not provide explanations.

Expected format:

{{
    "grounded": true
}}

or

{{
    "grounded": false
}}

PROVIDED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""

        logger.info(
            "Validating answer grounding."
        )

        response = self.llm.invoke(prompt)

        raw_result = response.content.strip()

        logger.info(
            f"Raw grounding response: {raw_result}"
        )

        # ---------------------------------------------
        # Remove markdown code fences if LLM adds them
        # ---------------------------------------------

        cleaned_result = raw_result

        if cleaned_result.startswith("```"):

            cleaned_result = (
                cleaned_result
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        try:

            result = json.loads(
                cleaned_result
            )

        except json.JSONDecodeError as exc:

            logger.exception(
                "Failed to parse grounding response."
            )

            raise ValueError(
                "Grounding guard returned invalid JSON."
            ) from exc

        grounded = result.get(
            "grounded"
        )

        if not isinstance(grounded, bool):

            raise ValueError(
                "Grounding response must contain "
                "a boolean 'grounded' field."
            )

        if grounded:

            logger.info(
                "Answer passed grounding validation."
            )

        else:

            logger.warning(
                "Answer failed grounding validation."
            )

        return grounded