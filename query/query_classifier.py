import json

from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class QueryClassifier:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0
        )

        logger.info(
            "Query classifier initialized."
        )

    def classify(
        self,
        query: str,
        conversation_history: str = ""
    ) -> dict:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        prompt = f"""
You are a query classification component
for an insurance policy RAG system.

Classify the user's query into exactly one
of the following categories:

1. INSURANCE_QUERY

Use this category when the user asks a
self-contained question about insurance
policies.

Examples:

"What medical expenses are covered?"
"Is dental treatment covered?"
"What is emergency medical evacuation?"
"What are the exclusions?"

The query can be understood without previous
conversation history.

2. FOLLOW_UP

Use this category when the user's query depends
on the previous conversation.

Look carefully at the conversation history.

Typical follow-up indicators include:

"What about it?"
"What about emergency evacuation?"
"What about dental treatment?"
"How about that?"
"Is it covered?"
"Does this apply to me?"
"What about the same condition?"
"How much is it?"
"Does the policy cover that too?"

IMPORTANT:

If the query contains a reference such as:

- it
- this
- that
- the same
- what about
- how about
- does it
- is it
- can I
- what about X

AND the conversation history provides context
for understanding that reference, classify it
as FOLLOW_UP.

A query such as:

"What about emergency evacuation?"

should be classified as FOLLOW_UP when the
previous conversation discusses insurance coverage.

3. OUT_OF_DOMAIN

Use this category when the query is unrelated
to the available insurance policy documents.

Examples:

"What is the weather today?"
"Who won the cricket match?"
"Write Python code."
"What is the capital of France?"

IMPORTANT CLASSIFICATION RULE:

First determine whether the query depends on
conversation history.

If it does, choose FOLLOW_UP even if the
underlying topic is insurance.

Return ONLY valid JSON.

Required format:

{{
    "category": "INSURANCE_QUERY",
    "confidence": 0.95,
    "reason": "Short explanation"
}}

Conversation History:
{conversation_history}

User Query:
{query}
"""

        logger.info(
            f"Classifying query: {query}"
        )

        response = self.llm.invoke(prompt)

        raw_result = response.content.strip()

        logger.info(
            f"Raw classifier response: {raw_result}"
        )

        # Remove markdown code fences if returned
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

            logger.error(
                "Failed to parse query "
                "classifier response."
            )

            raise ValueError(
                "Query classifier returned invalid JSON."
            ) from exc

        category = result.get("category")

        allowed_categories = {
            "INSURANCE_QUERY",
            "FOLLOW_UP",
            "OUT_OF_DOMAIN"
        }

        if category not in allowed_categories:

            raise ValueError(
                f"Invalid query category: {category}"
            )

        confidence = result.get(
            "confidence",
            0.0
        )

        logger.info(
            f"Query category: {category}, "
            f"confidence: {confidence}"
        )

        return result