from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):

    source_id: str

    policy_type: Optional[str] = None

    document_name: Optional[str] = None

    page_number: Optional[int] = None

    rerank_score: Optional[float] = None

    source: Optional[str] = None


class RAGResponse(BaseModel):

    # =========================================================
    # QUERY INFORMATION
    # =========================================================

    original_query: str

    rewritten_query: str

    # =========================================================
    # CLASSIFICATION INFORMATION
    # =========================================================

    query_category: Optional[str] = None

    classifier_confidence: Optional[float] = None

    classification_reason: Optional[str] = None

    # =========================================================
    # GENERATED ANSWER
    # =========================================================

    answer: str

    # =========================================================
    # RETRIEVAL INFORMATION
    # =========================================================

    retrieved_document_count: int = 0

    reranked_document_count: int = 0

    # =========================================================
    # GUARDRAIL STATUS
    # =========================================================

    relevance_passed: bool = False

    grounding_passed: bool = False

    groundedness_passed: bool = False

    citation_passed: bool = False

    # =========================================================
    # SOURCES
    # =========================================================

    sources: List[Source] = Field(
        default_factory=list
    )