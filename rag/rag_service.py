from typing import List, Dict, Optional, Tuple

from guardrails.relevance_guard import RelevanceGuard
from guardrails.grounding_guard import GroundingGuard
from guardrails.groundedness_guard import GroundednessGuard
from guardrails.citation_guard import CitationGuard
from guardrails.answer_retry_guard import AnswerRetryGuard

from query.query_rewriter import QueryRewriter
from query.query_classifier import QueryClassifier

from retrieval.semantic_retriever import SemanticRetriever

from reranking.llm_reranker import LLMReranker

from context.context_builder import ContextBuilder

from prompts.prompt_builder import PromptBuilder

from llm.openai_llm import OpenAILLM

from schemas.rag_response import RAGResponse, Source

from conversation.conversation_manager import ConversationManager

from utils.logger import get_logger


logger = get_logger(__name__)


class RAGService:

    def __init__(self):

        logger.info(
            "Initializing SBI Insurance RAG Service."
        )

        # =====================================================
        # STEP 1 - QUERY CLASSIFIER
        # =====================================================

        self.query_classifier = QueryClassifier()

        # =====================================================
        # STEP 2 - QUERY REWRITER
        # =====================================================

        self.query_rewriter = QueryRewriter()

        # =====================================================
        # STEP 3 - RETRIEVER
        # =====================================================

        self.retriever = SemanticRetriever()

        # =====================================================
        # STEP 4 - RERANKER
        # =====================================================

        self.reranker = LLMReranker()

        # =====================================================
        # STEP 5 - RELEVANCE GUARD
        # =====================================================

        self.relevance_guard = RelevanceGuard()

        # =====================================================
        # STEP 6 - GROUNDING GUARD
        # =====================================================

        self.grounding_guard = GroundingGuard()

        # =====================================================
        # STEP 7 - GROUNDEDNESS GUARD
        # =====================================================

        self.groundedness_guard = GroundednessGuard()

        # =====================================================
        # STEP 8 - CITATION GUARD
        # =====================================================

        self.citation_guard = CitationGuard()

        # =====================================================
        # STEP 9 - ANSWER RETRY GUARD
        # =====================================================

        self.answer_retry_guard = AnswerRetryGuard(
            max_retries=1
        )

        # =====================================================
        # STEP 10 - CONTEXT BUILDER
        # =====================================================

        self.context_builder = ContextBuilder()

        # =====================================================
        # STEP 11 - PROMPT BUILDER
        # =====================================================

        self.prompt_builder = PromptBuilder()

        # =====================================================
        # STEP 12 - LLM
        # =====================================================

        self.llm = OpenAILLM()

        # =====================================================
        # STEP 13 - CONVERSATION MANAGER
        # =====================================================

        self.conversation_manager = ConversationManager(
            max_messages=10
        )

        logger.info(
            "SBI Insurance RAG Service initialized successfully."
        )

    # =========================================================
    # BUILD AUTHORITATIVE SOURCES
    # =========================================================

    def _build_sources(
        self,
        ranked_documents
    ) -> List[Source]:

        """
        Build unique authoritative citation sources.

        A source is uniquely identified using:

            document_name + page_number

        Example:

            S1 -> Travel Insurance Policy.pdf -> Page 3
            S2 -> Travel Insurance Policy.pdf -> Page 4
            S3 -> Critical Illness Policy.pdf -> Page 7

        These source IDs are the ONLY citation IDs that
        the generated answer is allowed to use.
        """

        unique_sources = {}

        for document in ranked_documents:

            metadata = document.metadata or {}

            document_name = metadata.get(
                "document_name"
            )

            page_number = metadata.get(
                "page_number"
            )

            key = (
                document_name,
                page_number
            )

            if key in unique_sources:
                continue

            source_id = (
                f"S{len(unique_sources) + 1}"
            )

            try:

                rerank_score = metadata.get(
                    "rerank_score"
                )

                if rerank_score is not None:

                    rerank_score = float(
                        rerank_score
                    )

            except (TypeError, ValueError):

                rerank_score = None

            source = Source(

                source_id=source_id,

                policy_type=metadata.get(
                    "policy_type"
                ),

                document_name=document_name,

                page_number=page_number,

                rerank_score=rerank_score,

                source=metadata.get(
                    "source"
                )
            )

            unique_sources[key] = source

        sources = list(
            unique_sources.values()
        )

        logger.info(
            f"Built {len(sources)} unique citation sources."
        )

        for source in sources:

            logger.info(
                f"Citation mapping: "
                f"{source.source_id} -> "
                f"{source.document_name} "
                f"page {source.page_number}"
            )

        authoritative_ids = [
            source.source_id
            for source in sources
        ]

        logger.info(
            f"Authoritative citation IDs: "
            f"{authoritative_ids}"
        )

        return sources

    # =========================================================
    # BUILD CITATION-AWARE CONTEXT
    # =========================================================

    def _build_citation_context(
        self,
        ranked_documents,
        sources: List[Source]
    ) -> str:

        """
        Build context containing authoritative citation IDs.

        Every document included in the final context must map
        to one of the source IDs generated by _build_sources().
        """

        if not ranked_documents:

            logger.warning(
                "No ranked documents available "
                "for citation context."
            )

            return ""

        if not sources:

            logger.warning(
                "No sources available "
                "for citation context."
            )

            return ""

        # -----------------------------------------------------
        # BUILD SOURCE MAPPING
        # -----------------------------------------------------

        source_mapping = {}

        for source in sources:

            key = (
                source.document_name,
                source.page_number
            )

            source_mapping[key] = source.source_id

        # -----------------------------------------------------
        # BUILD CONTEXT
        # -----------------------------------------------------

        context_parts = []

        context_index = 1

        for document in ranked_documents:

            metadata = document.metadata or {}

            document_name = metadata.get(
                "document_name",
                "Unknown Document"
            )

            page_number = metadata.get(
                "page_number",
                "Unknown Page"
            )

            source_id = source_mapping.get(
                (
                    document_name,
                    page_number
                )
            )

            # -------------------------------------------------
            # NEVER INVENT SOURCE ID
            # -------------------------------------------------

            if not source_id:

                logger.warning(
                    f"No authoritative source ID found "
                    f"for {document_name} "
                    f"page {page_number}. "
                    f"Skipping document."
                )

                continue

            content = (
                document.page_content or ""
            ).strip()

            if not content:

                logger.warning(
                    f"Empty content found for "
                    f"{document_name} "
                    f"page {page_number}. "
                    f"Skipping document."
                )

                continue

            policy_type = metadata.get(
                "policy_type",
                "Unknown Policy"
            )

            source = metadata.get(
                "source",
                "Unknown Source"
            )

            rerank_score = metadata.get(
                "rerank_score",
                "N/A"
            )

            context_part = f"""
--- CONTEXT {context_index} ---

Citation ID:
[{source_id}]

Policy Type:
{policy_type}

Document:
{document_name}

Page:
{page_number}

Relevance Score:
{rerank_score}

Source:
{source}

Content:
{content}
"""

            context_parts.append(
                context_part.strip()
            )

            context_index += 1

        # -----------------------------------------------------
        # FINAL CONTEXT
        # -----------------------------------------------------

        if not context_parts:

            logger.warning(
                "No valid citation-aware context "
                "could be constructed."
            )

            return ""

        final_context = "\n\n".join(
            context_parts
        )

        logger.info(
            f"Citation-aware context created using "
            f"{len(context_parts)} context sections."
        )

        return final_context

    # =========================================================
    # GET AUTHORITATIVE SOURCE IDS
    # =========================================================

    @staticmethod
    def _get_authoritative_source_ids(
        sources: List[Source]
    ) -> List[str]:

        source_ids = []

        for source in sources:

            if source.source_id:

                source_ids.append(
                    source.source_id
                )

        return source_ids

    # =========================================================
    # BUILD STRICT GROUNDING RETRY PROMPT
    # =========================================================

    def _build_retry_prompt(
        self,
        user_query: str,
        context: str,
        sources: List[Source],
        conversation_history: str = ""
    ) -> str:

        """
        Prompt used when grounding or groundedness validation
        fails.

        The answer is regenerated using strict grounding rules.
        """

        source_ids = (
            self._get_authoritative_source_ids(
                sources
            )
        )

        source_id_text = ", ".join(
            source_ids
        )

        retry_prompt = f"""
You are an AI assistant specialized in
SBI insurance policy documents.

The previous generated answer failed
grounding or groundedness validation.

Generate a completely new answer.

============================================================
STRICT GROUNDING RULES
============================================================

1. Use ONLY information explicitly present
   in the Retrieved Policy Context.

2. Do NOT use outside knowledge.

3. Do NOT use general insurance knowledge.

4. Do NOT infer unsupported information.

5. Do NOT assume anything.

6. Do NOT expand policy statements beyond
   what is explicitly supported.

7. Do NOT introduce unsupported:
   - benefits
   - exclusions
   - conditions
   - limitations
   - monetary amounts
   - percentages
   - waiting periods
   - durations
   - eligibility requirements
   - medical facts
   - geographical restrictions
   - policy rules

8. If a concept is not explicitly supported
   by the retrieved context, do not mention it.

9. If the context contains only partial information,
   answer only the supported portion.

10. If the context is insufficient, say:

"The available policy context does not provide
sufficient information to answer this part
of the question."

11. Do not guess.

12. Keep the answer concise.

============================================================
STRICT CITATION RULES
============================================================

The ONLY valid citation IDs are:

{source_id_text}

Use citations exactly in this format:

[S1]

[S2]

[S1][S2]

Do NOT create any new citation IDs.

Do NOT use:

[1]
[2]
[Source 1]
(Source 1)
(Page 3)

Every factual policy claim MUST contain
a citation immediately after the claim.

============================================================
CONVERSATION HISTORY
============================================================

{conversation_history}

============================================================
RETRIEVED POLICY CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{user_query}

============================================================
FINAL ANSWER
============================================================
"""

        logger.info(
            "Strict grounding retry prompt constructed."
        )

        return retry_prompt

    # =========================================================
    # BUILD CITATION RETRY PROMPT
    # =========================================================

    def _build_citation_retry_prompt(
        self,
        user_query: str,
        context: str,
        sources: List[Source],
        conversation_history: str = ""
    ) -> str:

        """
        Prompt used when grounding and groundedness pass,
        but citation validation fails.
        """

        source_ids = (
            self._get_authoritative_source_ids(
                sources
            )
        )

        source_id_text = ", ".join(
            source_ids
        )

        retry_prompt = f"""
You are an AI assistant specialized in
SBI insurance policy documents.

The previous answer was supported by the
retrieved policy context, but its citations
were invalid or incomplete.

Generate the answer again.

============================================================
GROUNDING RULES
============================================================

1. Use ONLY the Retrieved Policy Context.

2. Do NOT use outside knowledge.

3. Do NOT infer unsupported policy information.

4. Do NOT add unsupported:
   - benefits
   - exclusions
   - conditions
   - limitations
   - amounts
   - percentages
   - durations
   - eligibility requirements
   - medical facts
   - geographical restrictions

5. Keep the answer concise.

============================================================
AUTHORITATIVE CITATION IDS
============================================================

The ONLY valid citation IDs are:

{source_id_text}

Use ONLY these IDs.

Valid:

[S1]

[S2]

[S1][S2]

Invalid:

[1]

[2]

[S3]

[Source 1]

(Source 1)

(Page 3)

============================================================
CITATION PLACEMENT
============================================================

Every factual policy claim MUST contain
a citation immediately after the claim.

Example:

The policy covers eligible medical expenses [S1].

Emergency medical transportation is covered [S2].

If multiple sources support a claim:

The benefit is described in both sources [S1][S2].

============================================================
CONVERSATION HISTORY
============================================================

{conversation_history}

============================================================
RETRIEVED POLICY CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{user_query}

============================================================
FINAL ANSWER
============================================================
"""

        logger.info(
            "Strict citation retry prompt constructed."
        )

        return retry_prompt

    # =========================================================
    # VALIDATE GENERATED ANSWER
    # =========================================================

    def _validate_answer(
        self,
        answer: str,
        context: str,
        sources: List[Source]
    ) -> Tuple[bool, bool, bool]:

        """
        Execute the validation chain:

            Grounding
                ↓
            Groundedness
                ↓
            Citation

        Returns:

            grounding_passed
            groundedness_passed
            citation_passed
        """

        grounding_passed = False
        groundedness_passed = False
        citation_passed = False

        if not answer or not answer.strip():

            logger.warning(
                "Answer validation skipped because "
                "answer is empty."
            )

            return (
                False,
                False,
                False
            )

        # =====================================================
        # GROUNDING
        # =====================================================

        try:

            grounding_passed = (
                self.grounding_guard.validate(
                    answer=answer,
                    context=context
                )
            )

        except Exception:

            logger.exception(
                "Grounding validation raised an exception."
            )

            grounding_passed = False

        logger.info(
            f"Grounding validation result: "
            f"{grounding_passed}"
        )

        if not grounding_passed:

            return (
                False,
                False,
                False
            )

        # =====================================================
        # GROUNDEDNESS
        # =====================================================

        try:

            groundedness_passed = (
                self.groundedness_guard.validate(
                    answer=answer,
                    context=context
                )
            )

        except Exception:

            logger.exception(
                "Groundedness validation raised "
                "an exception."
            )

            groundedness_passed = False

        logger.info(
            f"Groundedness validation result: "
            f"{groundedness_passed}"
        )

        if not groundedness_passed:

            return (
                grounding_passed,
                False,
                False
            )

        # =====================================================
        # CITATION
        # =====================================================

        try:

            citation_passed = (
                self.citation_guard.validate(
                    answer=answer,
                    sources=sources
                )
            )

        except Exception:

            logger.exception(
                "Citation validation raised "
                "an exception."
            )

            citation_passed = False

        logger.info(
            f"Citation validation result: "
            f"{citation_passed}"
        )

        return (
            grounding_passed,
            groundedness_passed,
            citation_passed
        )

    # =========================================================
    # GENERATE + VALIDATE ANSWER
    # =========================================================

    def _generate_validated_answer(
        self,
        user_query: str,
        context: str,
        sources: List[Source],
        conversation_history: str = ""
    ) -> Tuple[
        Optional[str],
        bool,
        bool,
        bool,
        int
    ]:

        retry_count = 0

        # =====================================================
        # FIRST ATTEMPT + GROUNDING RETRY
        # =====================================================

        while True:

            attempt_number = (
                retry_count + 1
            )

            logger.info(
                f"Generating answer. "
                f"Attempt: {attempt_number}"
            )

            # -------------------------------------------------
            # FIRST ATTEMPT
            # -------------------------------------------------

            if retry_count == 0:

                try:

                    prompt = (
                        self.prompt_builder.build_prompt(
                            query=user_query,
                            context=context,
                            conversation_history=(
                                conversation_history
                            )
                        )
                    )

                except Exception:

                    logger.exception(
                        "Failed to construct initial prompt."
                    )

                    return (
                        None,
                        False,
                        False,
                        False,
                        retry_count
                    )

            # -------------------------------------------------
            # GROUNDING / GROUNDEDNESS RETRY
            # -------------------------------------------------

            else:

                prompt = self._build_retry_prompt(
                    user_query=user_query,
                    context=context,
                    sources=sources,
                    conversation_history=(
                        conversation_history
                    )
                )

            if not prompt or not prompt.strip():

                logger.warning(
                    "Generated prompt is empty."
                )

                return (
                    None,
                    False,
                    False,
                    False,
                    retry_count
                )

            # =================================================
            # LLM GENERATION
            # =================================================

            try:

                answer = self.llm.generate(
                    prompt
                )

            except Exception:

                logger.exception(
                    "LLM answer generation failed."
                )

                answer = None

            if not answer or not answer.strip():

                logger.warning(
                    "LLM returned an empty answer."
                )

                grounding_passed = False
                groundedness_passed = False
                citation_passed = False

            else:

                answer = answer.strip()

                logger.info(
                    "LLM answer generated successfully."
                )

                (
                    grounding_passed,
                    groundedness_passed,
                    citation_passed
                ) = self._validate_answer(
                    answer=answer,
                    context=context,
                    sources=sources
                )

            # =================================================
            # ALL VALIDATIONS PASSED
            # =================================================

            if (
                answer
                and grounding_passed
                and groundedness_passed
                and citation_passed
            ):

                logger.info(
                    "Answer passed all guard validations."
                )

                return (
                    answer,
                    True,
                    True,
                    True,
                    retry_count
                )

            # =================================================
            # CITATION FAILURE
            # =================================================

            if (
                answer
                and grounding_passed
                and groundedness_passed
                and not citation_passed
            ):

                logger.warning(
                    "Grounding and groundedness passed, "
                    "but citation validation failed."
                )

                if retry_count >= 1:

                    logger.warning(
                        "Citation validation failed and "
                        "maximum retry limit reached."
                    )

                    return (
                        None,
                        grounding_passed,
                        groundedness_passed,
                        False,
                        retry_count
                    )

                # -------------------------------------------------
                # CITATION RETRY
                # -------------------------------------------------

                retry_count += 1

                logger.warning(
                    f"Retrying answer generation for "
                    f"citation correction. "
                    f"Retry number: {retry_count}"
                )

                citation_retry_prompt = (
                    self._build_citation_retry_prompt(
                        user_query=user_query,
                        context=context,
                        sources=sources,
                        conversation_history=(
                            conversation_history
                        )
                    )
                )

                try:

                    retry_answer = self.llm.generate(
                        citation_retry_prompt
                    )

                except Exception:

                    logger.exception(
                        "Citation retry LLM generation failed."
                    )

                    return (
                        None,
                        grounding_passed,
                        groundedness_passed,
                        False,
                        retry_count
                    )

                if (
                    not retry_answer
                    or not retry_answer.strip()
                ):

                    logger.warning(
                        "Citation retry returned empty answer."
                    )

                    return (
                        None,
                        grounding_passed,
                        groundedness_passed,
                        False,
                        retry_count
                    )

                retry_answer = retry_answer.strip()

                logger.info(
                    "Citation retry answer generated."
                )

                (
                    retry_grounding_passed,
                    retry_groundedness_passed,
                    retry_citation_passed
                ) = self._validate_answer(
                    answer=retry_answer,
                    context=context,
                    sources=sources
                )

                if (
                    retry_grounding_passed
                    and retry_groundedness_passed
                    and retry_citation_passed
                ):

                    logger.info(
                        "Citation retry passed all validations."
                    )

                    return (
                        retry_answer,
                        True,
                        True,
                        True,
                        retry_count
                    )

                logger.warning(
                    "Citation retry failed validation."
                )

                return (
                    None,
                    retry_grounding_passed,
                    retry_groundedness_passed,
                    retry_citation_passed,
                    retry_count
                )

            # =================================================
            # GROUNDING / GROUNDEDNESS FAILURE
            # =================================================

            should_retry = (
                self.answer_retry_guard.should_retry(
                    grounding_passed=(
                        grounding_passed
                    ),
                    groundedness_passed=(
                        groundedness_passed
                    ),
                    retry_count=retry_count
                )
            )

            if not should_retry:

                logger.warning(
                    "Answer failed validation and "
                    "no retry is available."
                )

                return (
                    None,
                    grounding_passed,
                    groundedness_passed,
                    citation_passed,
                    retry_count
                )

            # =================================================
            # RETRY
            # =================================================

            retry_count += 1

            logger.warning(
                f"Retrying answer generation due to "
                f"grounding/groundedness failure. "
                f"Retry number: {retry_count}"
            )

    # =========================================================
    # MAIN QUERY METHOD
    # =========================================================

    def query(
        self,
        user_query: str,
        policy_type: Optional[str] = None,
        conversation_history: str = ""
    ) -> RAGResponse:

        # =====================================================
        # INPUT VALIDATION
        # =====================================================

        if not user_query or not user_query.strip():

            raise ValueError(
                "User query cannot be empty."
            )

        user_query = user_query.strip()

        logger.info(
            f"Processing user query: {user_query}"
        )

        logger.info(
            f"Selected policy: {policy_type}"
        )

        # =====================================================
        # DEFAULT RESPONSE STATE
        # =====================================================

        query_category = None

        classifier_confidence = None

        classification_reason = ""

        retrieved_document_count = 0

        reranked_document_count = 0

        relevance_passed = False

        grounding_passed = False

        groundedness_passed = False

        citation_passed = False

        # =====================================================
        # STEP 1 - QUERY CLASSIFICATION
        # =====================================================

        try:

            classification = (
                self.query_classifier.classify(
                    query=user_query,
                    conversation_history=(
                        conversation_history
                    )
                )
            )

        except Exception:

            logger.exception(
                "Query classification failed."
            )

            classification = {
                "category": "INSURANCE_QUERY",
                "confidence": 0.0,
                "reason": (
                    "Classification failed; "
                    "defaulting to insurance query."
                )
            }

        query_category = classification.get(
            "category",
            "INSURANCE_QUERY"
        )

        classifier_confidence = classification.get(
            "confidence",
            0.0
        )

        classification_reason = classification.get(
            "reason",
            ""
        )

        logger.info(
            f"Query category: {query_category}"
        )

        logger.info(
            f"Classifier confidence: "
            f"{classifier_confidence}"
        )

        logger.info(
            f"Classification reason: "
            f"{classification_reason}"
        )

        # =====================================================
        # STEP 2 - OUT-OF-DOMAIN
        # =====================================================

        if query_category in("OUT_OF_SCOPE", "OUT_OF_DOMAIN"):

            logger.warning(
                f"Query classified as out-of-scope:"
                f"{query_category}"
            )

            return RAGResponse(

                original_query=user_query,

                rewritten_query=user_query,

                query_category=query_category,

                classifier_confidence=(
                    classifier_confidence
                ),

                classification_reason=(
                    classification_reason
                ),

                answer=(
                    "I can only answer questions related "
                    "to the available SBI Insurance Policy "
                    "documents."
                ),

                retrieved_document_count=0,

                reranked_document_count=0,

                relevance_passed=False,

                grounding_passed=False,

                groundedness_passed=False,

                citation_passed=False,

                sources=[]
            )

        # =====================================================
        # STEP 3 - QUERY REWRITING
        # =====================================================

        try:

            rewritten_query = (
                self.query_rewriter.rewrite(
                    query=user_query,
                    conversation_history=(
                        conversation_history
                    ),
                    policy_type=policy_type
                )
            )

        except Exception:

            logger.exception(
                "Query rewriting failed."
            )

            rewritten_query = user_query

        if (
            not rewritten_query
            or not rewritten_query.strip()
        ):

            logger.warning(
                "Query rewriter returned empty query. "
                "Using original query."
            )

            rewritten_query = user_query

        rewritten_query = rewritten_query.strip()

        logger.info(
            f"Rewritten query: {rewritten_query}"
        )

        # =====================================================
        # STEP 4 - RETRIEVAL
        # =====================================================

        logger.info(
            "Starting semantic retrieval."
        )

        try:

            documents = self.retriever.retrieve(
                query=rewritten_query,
                top_k=10,
                policy_type=policy_type
            )

        except Exception:

            logger.exception(
                "Semantic retrieval failed."
            )

            documents = []

        retrieved_document_count = len(
            documents
        )

        logger.info(
            f"Retrieved {retrieved_document_count} "
            f"documents."
        )

        # =====================================================
        # NO RETRIEVAL
        # =====================================================

        if not documents:

            return RAGResponse(

                original_query=user_query,

                rewritten_query=rewritten_query,

                query_category=query_category,

                classifier_confidence=(
                    classifier_confidence
                ),

                classification_reason=(
                    classification_reason
                ),

                answer=(
                    "I could not find relevant information "
                    "in the available insurance policy "
                    "documents."
                ),

                retrieved_document_count=(
                    retrieved_document_count
                ),

                reranked_document_count=0,

                relevance_passed=False,

                grounding_passed=False,

                groundedness_passed=False,

                citation_passed=False,

                sources=[]
            )

        # =====================================================
        # STEP 5 - RERANKING
        # =====================================================

        logger.info(
            "Starting document reranking."
        )

        try:

            ranked_documents = (
                self.reranker.rerank(
                    query=rewritten_query,
                    documents=documents,
                    top_k=5
                )
            )

        except Exception:

            logger.exception(
                "Document reranking failed."
            )

            ranked_documents = []

        reranked_document_count = len(
            ranked_documents
        )

        logger.info(
            f"Reranked documents: "
            f"{reranked_document_count}"
        )

        # =====================================================
        # NO RERANKED DOCUMENTS
        # =====================================================

        if not ranked_documents:

            return RAGResponse(

                original_query=user_query,

                rewritten_query=rewritten_query,

                query_category=query_category,

                classifier_confidence=(
                    classifier_confidence
                ),

                classification_reason=(
                    classification_reason
                ),

                answer=(
                    "I could not identify sufficiently "
                    "relevant information in the available "
                    "insurance policy documents."
                ),

                retrieved_document_count=(
                    retrieved_document_count
                ),

                reranked_document_count=0,

                relevance_passed=False,

                grounding_passed=False,

                groundedness_passed=False,

                citation_passed=False,

                sources=[]
            )

        # =====================================================
        # STEP 6 - RELEVANCE GUARD
        # =====================================================

        logger.info(
            "Running relevance validation."
        )

        try:

            relevance_passed = (
                self.relevance_guard.validate(
                    ranked_documents
                )
            )

        except Exception:

            logger.exception(
                "Relevance validation failed."
            )

            relevance_passed = False

        logger.info(
            f"Relevance validation result: "
            f"{relevance_passed}"
        )

        if not relevance_passed:

            logger.warning(
                "Retrieved documents failed "
                "relevance validation."
            )

            return RAGResponse(

                original_query=user_query,

                rewritten_query=rewritten_query,

                query_category=query_category,

                classifier_confidence=(
                    classifier_confidence
                ),

                classification_reason=(
                    classification_reason
                ),

                answer=(
                    "I couldn't find sufficiently relevant "
                    "information in the SBI Insurance Policy "
                    "documents to answer this question."
                ),

                retrieved_document_count=(
                    retrieved_document_count
                ),

                reranked_document_count=(
                    reranked_document_count
                ),

                relevance_passed=False,

                grounding_passed=False,

                groundedness_passed=False,

                citation_passed=False,

                sources=[]
            )

        # =====================================================
        # STEP 7 - BUILD AUTHORITATIVE SOURCES
        # =====================================================

        sources = self._build_sources(
            ranked_documents
        )

        if not sources:

            logger.warning(
                "No authoritative sources generated."
            )

            return RAGResponse(

                original_query=user_query,

                rewritten_query=rewritten_query,

                query_category=query_category,

                classifier_confidence=(
                    classifier_confidence
                ),

                classification_reason=(
                    classification_reason
                ),

                answer=(
                    "I couldn't generate authoritative "
                    "policy sources for this answer."
                ),

                retrieved_document_count=(
                    retrieved_document_count
                ),

                reranked_document_count=(
                    reranked_document_count
                ),

                relevance_passed=relevance_passed,

                grounding_passed=False,

                groundedness_passed=False,

                citation_passed=False,

                sources=[]
            )

        # =====================================================
        # STEP 8 - BUILD CITATION-AWARE CONTEXT
        # =====================================================

        context = self._build_citation_context(
            ranked_documents=ranked_documents,
            sources=sources
        )

        if not context or not context.strip():

            logger.warning(
                "Citation-aware context is empty."
            )

            return RAGResponse(

                original_query=user_query,

                rewritten_query=rewritten_query,

                query_category=query_category,

                classifier_confidence=(
                    classifier_confidence
                ),

                classification_reason=(
                    classification_reason
                ),

                answer=(
                    "I couldn't build sufficient context "
                    "from the available insurance policy "
                    "documents to answer this question."
                ),

                retrieved_document_count=(
                    retrieved_document_count
                ),

                reranked_document_count=(
                    reranked_document_count
                ),

                relevance_passed=relevance_passed,

                grounding_passed=False,

                groundedness_passed=False,

                citation_passed=False,

                sources=sources
            )

        # =====================================================
        # STEP 9 - GENERATE + VALIDATE
        # =====================================================

        (
            answer,
            grounding_passed,
            groundedness_passed,
            citation_passed,
            retry_count
        ) = self._generate_validated_answer(

            user_query=user_query,

            context=context,

            sources=sources,

            conversation_history=(
                conversation_history
            )
        )

        # =====================================================
        # FAILED VALIDATION
        # =====================================================

        if not answer:

            logger.warning(
                "Unable to generate a sufficiently "
                "validated answer."
            )

            if (
                grounding_passed
                and groundedness_passed
                and not citation_passed
            ):

                failure_message = (
                    "I couldn't verify that the generated "
                    "answer contains sufficiently valid "
                    "policy citations."
                )

            elif not groundedness_passed:

                failure_message = (
                    "I couldn't verify that the generated "
                    "answer is sufficiently supported by "
                    "the available SBI Insurance Policy "
                    "documents."
                )

            elif not grounding_passed:

                failure_message = (
                    "I couldn't verify that the generated "
                    "answer is grounded in the available "
                    "SBI Insurance Policy documents."
                )

            else:

                failure_message = (
                    "I couldn't generate a sufficiently "
                    "validated answer from the available "
                    "SBI Insurance Policy documents."
                )

            return RAGResponse(

                original_query=user_query,

                rewritten_query=rewritten_query,

                query_category=query_category,

                classifier_confidence=(
                    classifier_confidence
                ),

                classification_reason=(
                    classification_reason
                ),

                answer=failure_message,

                retrieved_document_count=(
                    retrieved_document_count
                ),

                reranked_document_count=(
                    reranked_document_count
                ),

                relevance_passed=relevance_passed,

                grounding_passed=(
                    grounding_passed
                ),

                groundedness_passed=(
                    groundedness_passed
                ),

                citation_passed=(
                    citation_passed
                ),

                sources=sources
            )

        # =====================================================
        # STEP 10 - STORE CONVERSATION
        # =====================================================

        try:

            self.conversation_manager.add_user_message(
                user_query
            )

            self.conversation_manager.add_assistant_message(
                answer
            )

            logger.info(
                "Conversation updated successfully."
            )

        except Exception as exc:

            logger.warning(
                f"Conversation update failed: {exc}"
            )

        # =====================================================
        # FINAL LOGGING
        # =====================================================

        logger.info(
            "RAG query completed successfully."
        )

        logger.info(
            f"Final validation state: "
            f"relevance={relevance_passed}, "
            f"grounding={grounding_passed}, "
            f"groundedness={groundedness_passed}, "
            f"citation={citation_passed}, "
            f"retries={retry_count}"
        )

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

        return RAGResponse(

            original_query=user_query,

            rewritten_query=rewritten_query,

            query_category=query_category,

            classifier_confidence=(
                classifier_confidence
            ),

            classification_reason=(
                classification_reason
            ),

            answer=answer,

            retrieved_document_count=(
                retrieved_document_count
            ),

            reranked_document_count=(
                reranked_document_count
            ),

            relevance_passed=relevance_passed,

            grounding_passed=grounding_passed,

            groundedness_passed=(
                groundedness_passed
            ),

            citation_passed=citation_passed,

            sources=sources
        )