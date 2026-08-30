import json
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class LLMReranker:

    """
    LLM-based document reranker.

    Purpose:
        Re-rank semantically retrieved insurance policy
        documents according to their relevance to the
        rewritten user query.

    Input:
        - User query
        - Retrieved documents

    Output:
        - Top-k most relevant documents

    The reranker does NOT generate the final answer.
    It only determines which retrieved documents should
    be passed to the downstream RAG pipeline.
    """

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0
        )

        logger.info(
            "LLM reranker initialized successfully."
        )

    # =========================================================
    # BUILD RERANKING PROMPT
    # =========================================================

    def _build_prompt(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> str:

        """
        Build the LLM reranking prompt.

        The LLM is instructed to:
            - score every candidate
            - rank by relevance
            - return JSON only
            - never answer the user question
        """

        prompt = f"""
You are a relevance-ranking component
for an insurance policy RAG system.

Your task is ONLY to rank the candidate policy
chunks according to their relevance to the
user's query.

Do NOT answer the user's question.

Do NOT rewrite the documents.

Do NOT add information.

Do NOT use outside knowledge.

User Query:
{query}

Candidate Documents:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Instructions:

1. Evaluate every candidate document.
2. Score each candidate from 0 to 10.
3. 10 = highly relevant to the query.
4. 0 = completely irrelevant.
5. Rank candidates from highest relevance to lowest.
6. Preserve the original candidate index.
7. Return ONLY valid JSON.
8. Do not include Markdown.
9. Do not include explanations.
10. Do not omit candidates.

Required JSON format:

[
    {{
        "index": 0,
        "score": 9
    }},
    {{
        "index": 1,
        "score": 7
    }}
]
"""

        return prompt.strip()

    # =========================================================
    # CLEAN LLM RESPONSE
    # =========================================================

    @staticmethod
    def _clean_response(
        raw_result: str
    ) -> str:

        """
        Remove common Markdown code fences from
        the LLM response before JSON parsing.
        """

        if not raw_result:

            return ""

        cleaned = raw_result.strip()

        if cleaned.startswith("```"):

            lines = cleaned.splitlines()

            # Remove first line containing ``` or ```json
            if lines:

                lines = lines[1:]

            # Remove final ```
            if lines and lines[-1].strip() == "```":

                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        return cleaned

    # =========================================================
    # VALIDATE RANKING RESPONSE
    # =========================================================

    @staticmethod
    def _validate_ranking(
        ranking: Any,
        document_count: int
    ) -> List[Dict[str, Any]]:

        """
        Validate and normalize the LLM ranking response.

        Invalid ranking entries are ignored instead of
        crashing the complete RAG pipeline.
        """

        if not isinstance(ranking, list):

            raise ValueError(
                "Reranker response must be a JSON list."
            )

        if not ranking:

            return []

        validated_ranking = []

        seen_indexes = set()

        for item in ranking:

            if not isinstance(item, dict):

                logger.warning(
                    "Ignoring invalid reranker item: "
                    f"{item}"
                )

                continue

            if "index" not in item:

                logger.warning(
                    "Ignoring reranker item without index: "
                    f"{item}"
                )

                continue

            if "score" not in item:

                logger.warning(
                    "Ignoring reranker item without score: "
                    f"{item}"
                )

                continue

            # -------------------------------------------------
            # Validate index
            # -------------------------------------------------

            try:

                index = int(
                    item["index"]
                )

            except (
                TypeError,
                ValueError
            ):

                logger.warning(
                    "Ignoring reranker item with invalid "
                    f"index: {item}"
                )

                continue

            if not (
                0 <= index < document_count
            ):

                logger.warning(
                    "Ignoring reranker item with "
                    f"out-of-range index: {index}"
                )

                continue

            # -------------------------------------------------
            # Avoid duplicate indexes
            # -------------------------------------------------

            if index in seen_indexes:

                logger.warning(
                    f"Ignoring duplicate reranker index: "
                    f"{index}"
                )

                continue

            # -------------------------------------------------
            # Validate score
            # -------------------------------------------------

            try:

                score = float(
                    item["score"]
                )

            except (
                TypeError,
                ValueError
            ):

                logger.warning(
                    "Ignoring reranker item with invalid "
                    f"score: {item}"
                )

                continue

            # -------------------------------------------------
            # Clamp score to 0-10
            # -------------------------------------------------

            score = max(
                0.0,
                min(
                    10.0,
                    score
                )
            )

            seen_indexes.add(
                index
            )

            validated_ranking.append(
                {
                    "index": index,
                    "score": score
                }
            )

        # -----------------------------------------------------
        # Sort by score
        # -----------------------------------------------------

        validated_ranking.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return validated_ranking

    # =========================================================
    # RERANK DOCUMENTS
    # =========================================================

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:

        """
        Re-rank retrieved documents using an LLM.

        Parameters:
            query:
                Rewritten user query.

            documents:
                Documents returned by semantic retrieval.

            top_k:
                Number of documents to return after reranking.

        Returns:
            List[Document]
                Top-k reranked documents.
        """

        # =====================================================
        # INPUT VALIDATION
        # =====================================================

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        if not documents:

            logger.warning(
                "No documents supplied for reranking."
            )

            return []

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        # Prevent requesting more documents than exist
        top_k = min(
            top_k,
            len(documents)
        )

        logger.info(
            f"Starting LLM reranking. "
            f"Documents: {len(documents)}, "
            f"Requested top_k: {top_k}"
        )

        # =====================================================
        # BUILD CANDIDATES
        # =====================================================

        candidates = []

        for index, document in enumerate(
            documents
        ):

            content = (
                document.page_content or ""
            ).strip()

            candidates.append(
                {
                    "index": index,
                    "content": content
                }
            )

        logger.info(
            f"Prepared {len(candidates)} "
            f"candidate documents for reranking."
        )

        # =====================================================
        # BUILD PROMPT
        # =====================================================

        prompt = self._build_prompt(
            query=query,
            candidates=candidates
        )

        logger.debug(
            "Reranker prompt constructed successfully."
        )

        # =====================================================
        # LLM CALL
        # =====================================================

        try:

            response = self.llm.invoke(
                prompt
            )

        except Exception as exc:

            logger.exception(
                "LLM reranker invocation failed."
            )

            raise RuntimeError(
                "LLM reranker invocation failed."
            ) from exc

        # =====================================================
        # EXTRACT RESPONSE
        # =====================================================

        raw_result = getattr(
            response,
            "content",
            ""
        )

        if isinstance(
            raw_result,
            list
        ):

            # Some LangChain/OpenAI response formats
            # may return content blocks.
            raw_result = "".join(
                str(
                    block.get(
                        "text",
                        ""
                    )
                )
                if isinstance(
                    block,
                    dict
                )
                else str(block)
                for block in raw_result
            )

        raw_result = str(
            raw_result
        ).strip()

        logger.info(
            f"Raw reranker response: "
            f"{raw_result}"
        )

        if not raw_result:

            raise ValueError(
                "Reranker returned an empty response."
            )

        # =====================================================
        # CLEAN RESPONSE
        # =====================================================

        cleaned_result = (
            self._clean_response(
                raw_result
            )
        )

        logger.info(
            f"Cleaned reranker response: "
            f"{cleaned_result}"
        )

        if not cleaned_result:

            raise ValueError(
                "Reranker returned an empty "
                "response after cleaning."
            )

        # =====================================================
        # PARSE JSON
        # =====================================================

        try:

            ranking = json.loads(
                cleaned_result
            )

        except json.JSONDecodeError as exc:

            logger.exception(
                "Failed to parse reranker response "
                "as JSON."
            )

            raise ValueError(
                "Reranker returned invalid JSON."
            ) from exc

        # =====================================================
        # VALIDATE RANKING
        # =====================================================

        try:

            validated_ranking = (
                self._validate_ranking(
                    ranking=ranking,
                    document_count=len(documents)
                )
            )

        except ValueError:

            logger.exception(
                "Reranker response structure "
                "validation failed."
            )

            raise

        if not validated_ranking:

            logger.warning(
                "Reranker returned no valid "
                "ranking entries."
            )

            return []

        # =====================================================
        # SELECT TOP-K DOCUMENTS
        # =====================================================

        selected_documents = []

        for item in validated_ranking[:top_k]:

            index = item["index"]

            score = item["score"]

            document = documents[index]

            # -------------------------------------------------
            # Copy score into document metadata
            # -------------------------------------------------

            document.metadata[
                "rerank_score"
            ] = score

            selected_documents.append(
                document
            )

            logger.debug(
                f"Selected document index={index}, "
                f"score={score}"
            )

        # =====================================================
        # FINAL LOGGING
        # =====================================================

        logger.info(
            f"Reranking completed successfully. "
            f"Selected {len(selected_documents)} "
            f"documents from {len(documents)} candidates."
        )

        return selected_documents

