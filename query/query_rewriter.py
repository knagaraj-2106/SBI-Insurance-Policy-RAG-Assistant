from langchain_openai import ChatOpenAI
from utils.logger import get_logger

from config.settings import settings


logger = get_logger(__name__)


class QueryRewriter:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0
        )

        logger.info(
            "Query rewriter initialized."
        )

    def rewrite(
        self,
        query: str,
        conversation_history: str = "",
        policy_type: str | None = None
    ) -> str:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        policy_context = ""

        if policy_type:

            policy_context = (
                f"\nPolicy Type: {policy_type}"
            )

        prompt = f"""
You are a query rewriting component
for an insurance policy RAG system.

Your task is to rewrite the user's query
into a clear, standalone search query.

Rules:

1. Preserve the original meaning.
2. Do not answer the question.
3. Do not add facts that are not present.
4. Resolve references using conversation history
   when possible.
5. Keep the query concise.
6. Return ONLY the rewritten query.

Conversation History:
{conversation_history}

{policy_context}

User Query:
{query}

Rewritten Query:
"""

        logger.info(
            "Rewriting user query."
        )

        response = self.llm.invoke(prompt)

        rewritten_query = response.content.strip()

        logger.info(
            f"Rewritten query: {rewritten_query}"
        )

        return rewritten_query