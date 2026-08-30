import json

from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class GroundednessGuard:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0
        )

        logger.info(
            "Groundedness guard initialized."
        )

    # =========================================================
    # MAIN VALIDATION METHOD
    # =========================================================

    def validate(
        self,
        answer: str,
        context: str
    ) -> bool:

        # -----------------------------------------------------
        # Validate answer
        # -----------------------------------------------------

        if not answer or not answer.strip():

            logger.warning(
                "Groundedness validation failed: "
                "empty answer."
            )

            return False

        # -----------------------------------------------------
        # Validate context
        # -----------------------------------------------------

        if not context or not context.strip():

            logger.warning(
                "Groundedness validation failed: "
                "empty context."
            )

            return False

        # -----------------------------------------------------
        # Build evaluator prompt
        # -----------------------------------------------------

        prompt = f"""
You are a strict groundedness evaluator for an
insurance policy Retrieval-Augmented Generation system.

Your task is to determine whether the GENERATED ANSWER
is adequately supported by the PROVIDED POLICY CONTEXT.

IMPORTANT RULES:

1. Evaluate ONLY against the provided policy context.

2. Do NOT use outside knowledge.

3. Minor wording differences are acceptable.

4. Paraphrasing is acceptable.

5. The answer does not need to copy the context word-for-word.

6. Information from multiple context sections may be combined.

7. Do not require every sentence to exactly match a
   sentence in the context.

8. The answer must not introduce unsupported:
   - benefits
   - exclusions
   - coverage limits
   - monetary amounts
   - percentages
   - durations
   - conditions
   - eligibility requirements
   - policy rules
   - medical facts
   - geographical restrictions

9. If the main claims of the answer are supported by
   the context, mark it as grounded.

10. If the answer contains a significant unsupported claim,
    mark it as not grounded.

11. If a minor stylistic statement is not explicitly present
    in the context but does not add a factual policy claim,
    do not fail the answer for that reason.

12. Be conservative when evaluating insurance policy claims.

------------------------------------------------------------
EXAMPLE 1
------------------------------------------------------------

POLICY CONTEXT:

The policy covers medically necessary medical expenses
incurred overseas.

GENERATED ANSWER:

The policy covers medically necessary medical expenses
incurred overseas.

RESULT:

{{
    "grounded": true,
    "reason": "The answer is directly supported by the context."
}}

------------------------------------------------------------
EXAMPLE 2
------------------------------------------------------------

POLICY CONTEXT:

The policy covers medically necessary medical expenses
incurred overseas.

GENERATED ANSWER:

The policy covers medically necessary medical expenses
and provides a maximum benefit of Rs. 10 lakh.

RESULT:

{{
    "grounded": false,
    "reason": "The Rs. 10 lakh benefit limit is not supported by the context."
}}

------------------------------------------------------------
EXAMPLE 3
------------------------------------------------------------

POLICY CONTEXT:

The policy provides emergency medical evacuation when
adequate medical treatment is not available within a
reasonable distance.

GENERATED ANSWER:

Emergency medical evacuation may be provided when adequate
medical treatment is not available within a reasonable distance.

RESULT:

{{
    "grounded": true,
    "reason": "The answer is a supported paraphrase of the context."
}}

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Return ONLY valid JSON.

The JSON must contain exactly:

{{
    "grounded": true,
    "reason": "short explanation"
}}

or

{{
    "grounded": false,
    "reason": "short explanation"
}}

Do not return markdown.

Do not return code fences.

Do not return any text outside the JSON.

------------------------------------------------------------
POLICY CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
GENERATED ANSWER
------------------------------------------------------------

{answer}

------------------------------------------------------------
EVALUATION
------------------------------------------------------------
"""

        logger.info(
            "Running groundedness validation."
        )

        # =====================================================
        # CALL LLM
        # =====================================================

        try:

            response = self.llm.invoke(
                prompt
            )

            raw_result = (
                response.content
                .strip()
            )

            logger.info(
                f"Raw groundedness response: "
                f"{raw_result}"
            )

        except Exception:

            logger.exception(
                "Groundedness validation failed "
                "because the LLM call failed."
            )

            return False

        # =====================================================
        # CLEAN RESPONSE
        # =====================================================

        cleaned_result = raw_result

        # Remove markdown code fences if the model
        # unexpectedly returns them.

        if cleaned_result.startswith("```"):

            cleaned_result = (
                cleaned_result
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        # =====================================================
        # PARSE JSON
        # =====================================================

        try:

            result = json.loads(
                cleaned_result
            )

        except json.JSONDecodeError:

            logger.exception(
                "Failed to parse groundedness "
                "guard response as JSON."
            )

            return False

        # =====================================================
        # VALIDATE JSON STRUCTURE
        # =====================================================

        grounded = result.get(
            "grounded"
        )

        reason = result.get(
            "reason",
            ""
        )

        if not isinstance(
            grounded,
            bool
        ):

            logger.warning(
                "Groundedness response contains "
                "an invalid 'grounded' field."
            )

            return False

        logger.info(
            f"Groundedness result: {grounded}"
        )

        logger.info(
            f"Groundedness reason: {reason}"
        )

        # =====================================================
        # FINAL DECISION
        # =====================================================

        if grounded:

            logger.info(
                "Answer passed groundedness validation."
            )

            return True

        logger.warning(
            "Answer failed groundedness validation."
        )

        return False