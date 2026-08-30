"""
RAG Regression Test Suite
=========================

Purpose:
    End-to-end regression testing for the SBI Insurance RAG pipeline.

This suite verifies:

    1. Valid insurance questions are answered.
    2. Supported policy benefits pass all guardrails.
    3. Unsupported information is not hallucinated.
    4. Out-of-domain questions are rejected.
    5. Citations are produced for supported answers.
    6. The RAG pipeline remains stable after code changes.

Run:

    python -m scripts.test_rag_regression
"""

import logging
import re

from typing import Any, Dict, List

from rag.rag_service import RAGService


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

POLICY_TYPE = "Travel Insurance Policy"


TEST_CASES: List[Dict[str, Any]] = [

    # =========================================================================
    # TEST 1
    # =========================================================================

    {
        "id": 1,
        "name": "Valid medical coverage question",

        "query": (
            "What medical expenses are covered under "
            "the Travel Insurance Policy?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": True,

        "description": (
            "A supported insurance question should pass "
            "the complete RAG pipeline."
        ),
    },

    # =========================================================================
    # TEST 2
    # =========================================================================

    {
        "id": 2,
        "name": "Emergency medical evacuation",

        "query": (
            "Does the Travel Insurance Policy provide "
            "emergency medical evacuation?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": True,

        "description": (
            "A supported policy benefit should produce "
            "a grounded answer."
        ),
    },

    # =========================================================================
    # TEST 3
    # =========================================================================

    {
        "id": 3,
        "name": "Maternity exclusion",

        "query": (
            "Does the Travel Insurance Policy cover "
            "maternity expenses?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": True,

        "description": (
            "The policy contains maternity-related "
            "exclusion information, so the system should "
            "answer from policy evidence."
        ),
    },

    # =========================================================================
    # TEST 4
    # =========================================================================

    {
        "id": 4,
        "name": "Unsupported monetary limit",

        "query": (
            "What is the exact maximum medical coverage "
            "amount under the Travel Insurance Policy?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": False,

        "must_not_contain": [
            "52,258 USD",
            "43,275 USD",
        ],

        "description": (
            "The system must not expose unsupported "
            "monetary amounts."
        ),
    },

    # =========================================================================
    # TEST 5
    # =========================================================================

    {
        "id": 5,
        "name": "Waiting period",

        "query": (
            "What is the waiting period for medical coverage "
            "under the Travel Insurance Policy?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": True,

        "description": (
            "The answer should reflect the policy evidence "
            "regarding the waiting period."
        ),
    },

    # =========================================================================
    # TEST 6
    # =========================================================================

    {
        "id": 6,
        "name": "Out-of-domain question",

        "query": (
            "What is today's stock market price?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "OUT_OF_DOMAIN",

        "expect_retrieval": False,

        "expect_validated_answer": False,

        "description": (
            "The system should reject questions unrelated "
            "to the available insurance policies."
        ),
    },

    # =========================================================================
    # TEST 7
    # =========================================================================

    {
        "id": 7,
        "name": "Valid policy benefit",

        "query": (
            "Does the Travel Insurance Policy cover "
            "emergency medical expenses?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": True,

        "description": (
            "A normal supported insurance question "
            "should succeed."
        ),
    },

    # =========================================================================
    # TEST 8
    # =========================================================================

    {
        "id": 8,
        "name": "Medical treatment continuation",

        "query": (
            "Does the Travel Insurance Policy cover "
            "continuation of medical treatment in India?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": True,

        "description": (
            "Tests retrieval and grounded generation "
            "for a specific medical coverage condition."
        ),
    },

    # =========================================================================
    # TEST 9
    # =========================================================================

    {
        "id": 9,
        "name": "Unsupported benefit",

        "query": (
            "Does the Travel Insurance Policy provide "
            "unlimited cosmetic surgery coverage?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": False,

        "must_not_contain": [
            "unlimited",
            "fully covered",
            "100% covered",
        ],

        "description": (
            "The system must not invent unsupported "
            "insurance benefits."
        ),
    },

    # =========================================================================
    # TEST 10
    # =========================================================================

    {
        "id": 10,
        "name": "Valid control question",

        "query": (
            "What benefits are available for medical "
            "emergencies under the Travel Insurance Policy?"
        ),

        "policy_type": POLICY_TYPE,

        "expected_category": "INSURANCE_QUERY",

        "expect_retrieval": True,

        "expect_validated_answer": True,

        "description": (
            "Final end-to-end control test for the "
            "production RAG pipeline."
        ),
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_result_value(
    result: Any,
    attribute: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve a value from the RAG response.

    Supports:

        1. Pydantic/object-style responses
        2. Dictionary-style responses
    """

    if result is None:
        return default

    if isinstance(result, dict):
        return result.get(attribute, default)

    return getattr(
        result,
        attribute,
        default,
    )


# ----------------------------------------------------------------------------
# NORMALIZE TEXT
# ----------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    """
    Convert a value into normalized lowercase text.
    """

    if value is None:
        return ""

    return str(value).strip().lower()


# ----------------------------------------------------------------------------
# EXTRACT CITATIONS
# ----------------------------------------------------------------------------

def extract_citations(answer: str) -> List[str]:
    """
    Extract authoritative citation IDs.

    Supported examples:

        [S1]
        [S2]
        [S1][S2]

    Returns:

        ["S1", "S2"]
    """

    if not answer:
        return []

    citations = re.findall(
        r"\[(S\d+)\]",
        answer,
    )

    return sorted(
        set(citations)
    )


# ----------------------------------------------------------------------------
# GET SOURCE IDS
# ----------------------------------------------------------------------------

def extract_source_ids(
    result: Any,
) -> List[str]:
    """
    Extract authoritative source IDs from RAGResponse.sources.

    Example:

        sources = [
            Source(source_id="S1", ...),
            Source(source_id="S2", ...)
        ]

    Returns:

        ["S1", "S2"]
    """

    sources = get_result_value(
        result,
        "sources",
        [],
    )

    if not sources:
        return []

    source_ids: List[str] = []

    for source in sources:

        source_id = get_result_value(
            source,
            "source_id",
            None,
        )

        if source_id:
            source_ids.append(
                str(source_id)
            )

    return sorted(
        set(source_ids)
    )


# ----------------------------------------------------------------------------
# VALIDATE CITATION REFERENCES
# ----------------------------------------------------------------------------

def validate_citation_references(
    answer: str,
    result: Any,
) -> List[str]:
    """
    Verify that every citation appearing in the answer
    belongs to the authoritative sources returned by RAGService.

    Example:

        Answer:
            Emergency medical expenses are covered [S1].

        Sources:
            S1
            S2

    Result:

        No failure.

    If answer contains [S5] but authoritative sources are
    only S1 and S2, the test fails.
    """

    failures: List[str] = []

    answer_citations = extract_citations(
        answer
    )

    authoritative_ids = extract_source_ids(
        result
    )

    if not answer_citations:
        return failures

    for citation in answer_citations:

        if citation not in authoritative_ids:

            failures.append(
                f"Answer contains non-authoritative "
                f"citation '{citation}'. "
                f"Authoritative citations: "
                f"{authoritative_ids}"
            )

    return failures


# ----------------------------------------------------------------------------
# VALIDATE TEST CASE
# ----------------------------------------------------------------------------

def validate_test_case(
    test_case: Dict[str, Any],
    result: Any,
) -> List[str]:
    """
    Validate one regression test case.

    Returns:

        List of validation failures.

    Empty list means the test passed.
    """

    failures: List[str] = []

    expected_category = (
        test_case["expected_category"]
    )

    expect_retrieval = (
        test_case["expect_retrieval"]
    )

    expect_validated_answer = (
        test_case["expect_validated_answer"]
    )

    # =========================================================================
    # CATEGORY VALIDATION
    # =========================================================================

    # IMPORTANT:
    #
    # RAGResponse uses:
    #
    #     query_category
    #
    # NOT:
    #
    #     category

    actual_category = get_result_value(
        result,
        "query_category",
        None,
    )

    if actual_category != expected_category:

        failures.append(
            f"Category mismatch: "
            f"expected={expected_category}, "
            f"actual={actual_category}"
        )

    # =========================================================================
    # RETRIEVAL COUNT VALIDATION
    # =========================================================================

    # IMPORTANT:
    #
    # RAGResponse stores:
    #
    #     retrieved_document_count
    #
    # NOT:
    #
    #     retrieved_documents

    retrieved_document_count = get_result_value(
        result,
        "retrieved_document_count",
        0,
    )

    try:
        retrieved_document_count = int(
            retrieved_document_count
        )
    except (TypeError, ValueError):

        retrieved_document_count = 0

        failures.append(
            "Invalid retrieved_document_count "
            "returned by RAGResponse."
        )

    if (
        expect_retrieval
        and retrieved_document_count == 0
    ):

        failures.append(
            "Expected retrieval to occur, "
            "but retrieved_document_count is 0."
        )

    if (
        not expect_retrieval
        and retrieved_document_count != 0
    ):

        failures.append(
            "Expected zero retrieved documents, "
            f"but received "
            f"{retrieved_document_count}."
        )

    # =========================================================================
    # RERANKED DOCUMENT COUNT
    # =========================================================================

    reranked_document_count = get_result_value(
        result,
        "reranked_document_count",
        0,
    )

    try:
        reranked_document_count = int(
            reranked_document_count
        )
    except (TypeError, ValueError):

        reranked_document_count = 0

        failures.append(
            "Invalid reranked_document_count "
            "returned by RAGResponse."
        )

    # =========================================================================
    # GUARDRAIL STATUS
    # =========================================================================

    relevance_passed = bool(
        get_result_value(
            result,
            "relevance_passed",
            False,
        )
    )

    grounding_passed = bool(
        get_result_value(
            result,
            "grounding_passed",
            False,
        )
    )

    groundedness_passed = bool(
        get_result_value(
            result,
            "groundedness_passed",
            False,
        )
    )

    citation_passed = bool(
        get_result_value(
            result,
            "citation_passed",
            False,
        )
    )

    all_validations_passed = (
        relevance_passed
        and grounding_passed
        and groundedness_passed
        and citation_passed
    )

    # =========================================================================
    # EXPECTED VALIDATION BEHAVIOR
    # =========================================================================

    if (
        expect_validated_answer
        and not all_validations_passed
    ):

        failures.append(
            "Expected all guardrails to pass, "
            "but one or more validation checks failed."
        )

    # -------------------------------------------------------------------------
    # NEGATIVE TEST
    # -------------------------------------------------------------------------

    if (
        not expect_validated_answer
        and all_validations_passed
    ):

        failures.append(
            "Unsupported/adversarial query "
            "unexpectedly passed all validation "
            "guardrails."
        )

    # =========================================================================
    # FINAL ANSWER
    # =========================================================================

    answer = get_result_value(
        result,
        "answer",
        "",
    )

    answer_text = normalize_text(
        answer
    )

    # =========================================================================
    # ANSWER EMPTY CHECK
    # =========================================================================

    if (
        expect_validated_answer
        and not answer_text
    ):

        failures.append(
            "Expected a validated answer, "
            "but the final answer is empty."
        )

    # =========================================================================
    # UNSUPPORTED CLAIM CHECKS
    # =========================================================================

    forbidden_patterns = test_case.get(
        "must_not_contain",
        [],
    )

    for forbidden_pattern in forbidden_patterns:

        if (
            normalize_text(
                forbidden_pattern
            )
            in answer_text
        ):

            failures.append(
                "Unsupported content found in "
                f"final answer: "
                f"'{forbidden_pattern}'"
            )

    # =========================================================================
    # CITATION VALIDATION
    # =========================================================================

    if expect_validated_answer:

        citations = extract_citations(
            answer
        )

        if not citations:

            failures.append(
                "Expected a validated insurance "
                "answer with citations, but no "
                "citation was found in the final answer."
            )

        # ---------------------------------------------------------------------
        # Verify citations against RAGResponse.sources
        # ---------------------------------------------------------------------

        citation_reference_failures = (
            validate_citation_references(
                answer=answer,
                result=result,
            )
        )

        failures.extend(
            citation_reference_failures
        )

    return failures


# ============================================================================
# PRINT SOURCE INFORMATION
# ============================================================================

def print_sources(
    result: Any,
) -> None:
    """
    Print authoritative sources returned by RAGService.
    """

    sources = get_result_value(
        result,
        "sources",
        [],
    )

    if not sources:

        print(
            "Sources: None"
        )

        return

    print(
        f"Sources: {len(sources)}"
    )

    for source in sources:

        source_id = get_result_value(
            source,
            "source_id",
            "N/A",
        )

        document_name = get_result_value(
            source,
            "document_name",
            "N/A",
        )

        page_number = get_result_value(
            source,
            "page_number",
            "N/A",
        )

        policy_type = get_result_value(
            source,
            "policy_type",
            "N/A",
        )

        rerank_score = get_result_value(
            source,
            "rerank_score",
            "N/A",
        )

        print(
            f"  {source_id} -> "
            f"{document_name} | "
            f"Page: {page_number} | "
            f"Policy: {policy_type} | "
            f"Rerank Score: {rerank_score}"
        )


# ============================================================================
# MAIN REGRESSION TEST RUNNER
# ============================================================================

def run_regression_tests() -> bool:
    """
    Execute the complete RAG regression suite.

    Returns:

        True  -> all tests passed
        False -> one or more tests failed
    """

    print("\n")

    print("=" * 78)

    print(
        "SBI INSURANCE RAG - REGRESSION TEST SUITE"
    )

    print("=" * 78)

    print(
        f"Policy: {POLICY_TYPE}"
    )

    print(
        f"Total Tests: {len(TEST_CASES)}"
    )

    print("=" * 78)

    # =========================================================================
    # INITIALIZE RAG SERVICE
    # =========================================================================

    try:

        rag_service = RAGService()

    except Exception as exc:

        print(
            "\n❌ RAG SERVICE INITIALIZATION FAILED"
        )

        print(
            f"Error: {exc}"
        )

        logger.exception(
            "Unable to initialize RAGService."
        )

        return False

    # =========================================================================
    # TEST COUNTERS
    # =========================================================================

    passed_tests = 0

    failed_tests = 0

    test_results: List[Dict[str, Any]] = []

    # =========================================================================
    # EXECUTE TEST CASES
    # =========================================================================

    for test_case in TEST_CASES:

        test_id = test_case["id"]

        test_name = test_case["name"]

        query = test_case["query"]

        print("\n")

        print("-" * 78)

        print(
            f"TEST {test_id}: {test_name}"
        )

        print("-" * 78)

        print(
            f"Query: {query}"
        )

        print(
            f"Policy Type: "
            f"{test_case['policy_type']}"
        )

        print(
            f"Expected Category: "
            f"{test_case['expected_category']}"
        )

        print(
            f"Expected Retrieval: "
            f"{test_case['expect_retrieval']}"
        )

        print(
            f"Expected Validated Answer: "
            f"{test_case['expect_validated_answer']}"
        )

        print(
            f"Description: "
            f"{test_case['description']}"
        )

        try:

            # =================================================================
            # EXECUTE RAG QUERY
            # =================================================================

            result = rag_service.query(

                user_query=query,

                policy_type=(
                    test_case["policy_type"]
                ),
            )

            # =================================================================
            # EXTRACT RESULT INFORMATION
            # =================================================================

            category = get_result_value(
                result,
                "query_category",
                None,
            )

            classifier_confidence = (
                get_result_value(
                    result,
                    "classifier_confidence",
                    None,
                )
            )

            classification_reason = (
                get_result_value(
                    result,
                    "classification_reason",
                    "",
                )
            )

            rewritten_query = (
                get_result_value(
                    result,
                    "rewritten_query",
                    "",
                )
            )

            retrieved_document_count = (
                get_result_value(
                    result,
                    "retrieved_document_count",
                    0,
                )
            )

            reranked_document_count = (
                get_result_value(
                    result,
                    "reranked_document_count",
                    0,
                )
            )

            relevance_passed = (
                get_result_value(
                    result,
                    "relevance_passed",
                    False,
                )
            )

            grounding_passed = (
                get_result_value(
                    result,
                    "grounding_passed",
                    False,
                )
            )

            groundedness_passed = (
                get_result_value(
                    result,
                    "groundedness_passed",
                    False,
                )
            )

            citation_passed = (
                get_result_value(
                    result,
                    "citation_passed",
                    False,
                )
            )

            answer = get_result_value(
                result,
                "answer",
                "",
            )

            # =================================================================
            # VALIDATE TEST
            # =================================================================

            failures = validate_test_case(
                test_case=test_case,
                result=result,
            )

            # =================================================================
            # PRINT RESULT DETAILS
            # =================================================================

            print("\nRESULT")

            print(
                f"Category: {category}"
            )

            print(
                f"Classifier Confidence: "
                f"{classifier_confidence}"
            )

            print(
                f"Classification Reason: "
                f"{classification_reason}"
            )

            print(
                f"Rewritten Query: "
                f"{rewritten_query}"
            )

            print(
                f"Retrieved Documents: "
                f"{retrieved_document_count}"
            )

            print(
                f"Reranked Documents: "
                f"{reranked_document_count}"
            )

            print(
                f"Relevance Passed: "
                f"{relevance_passed}"
            )

            print(
                f"Grounding Passed: "
                f"{grounding_passed}"
            )

            print(
                f"Groundedness Passed: "
                f"{groundedness_passed}"
            )

            print(
                f"Citation Passed: "
                f"{citation_passed}"
            )

            # =================================================================
            # PRINT SOURCES
            # =================================================================

            print("\nAUTHORITATIVE SOURCES")

            print_sources(
                result
            )

            # =================================================================
            # PRINT ANSWER
            # =================================================================

            print("\nANSWER")

            if answer:

                print(
                    answer
                )

            else:

                print(
                    "[No answer returned]"
                )

            # =================================================================
            # PRINT CITATIONS FOUND IN ANSWER
            # =================================================================

            answer_citations = extract_citations(
                answer
            )

            print(
                "\nCitations Found in Answer: "
                f"{answer_citations}"
            )

            authoritative_citations = (
                extract_source_ids(
                    result
                )
            )

            print(
                "Authoritative Source IDs: "
                f"{authoritative_citations}"
            )

            # =================================================================
            # FINAL TEST RESULT
            # =================================================================

            if failures:

                failed_tests += 1

                print(
                    "\n❌ TEST FAILED"
                )

                for failure in failures:

                    print(
                        f"   - {failure}"
                    )

                test_results.append(
                    {
                        "id": test_id,
                        "name": test_name,
                        "status": "FAILED",
                        "failures": failures,
                    }
                )

            else:

                passed_tests += 1

                print(
                    "\n✅ TEST PASSED"
                )

                test_results.append(
                    {
                        "id": test_id,
                        "name": test_name,
                        "status": "PASSED",
                        "failures": [],
                    }
                )

        except Exception as exc:

            failed_tests += 1

            print(
                "\n❌ TEST FAILED WITH EXCEPTION"
            )

            print(
                f"Error: {exc}"
            )

            logger.exception(
                "Regression test failed: %s",
                test_name,
            )

            test_results.append(
                {
                    "id": test_id,
                    "name": test_name,
                    "status": "FAILED",
                    "failures": [
                        f"Exception: {exc}"
                    ],
                }
            )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n")

    print("=" * 78)

    print(
        "RAG REGRESSION TEST SUITE SUMMARY"
    )

    print("=" * 78)

    print(
        f"Total Tests : {len(TEST_CASES)}"
    )

    print(
        f"Passed      : {passed_tests}"
    )

    print(
        f"Failed      : {failed_tests}"
    )

    # Calculate pass percentage

    total_tests = len(TEST_CASES)

    if total_tests > 0:

        pass_percentage = (
            passed_tests
            / total_tests
            * 100
        )

    else:

        pass_percentage = 0.0

    print(
        f"Pass Rate   : {pass_percentage:.2f}%"
    )

    print("=" * 78)

    # =========================================================================
    # DETAILED FAILURE SUMMARY
    # =========================================================================

    if failed_tests > 0:

        print(
            "\nFAILED TESTS"
        )

        print(
            "-" * 78
        )

        for test_result in test_results:

            if (
                test_result["status"]
                == "FAILED"
            ):

                print(
                    f"\nTEST "
                    f"{test_result['id']}: "
                    f"{test_result['name']}"
                )

                for failure in (
                    test_result["failures"]
                ):

                    print(
                        f"  - {failure}"
                    )

        print("\n")

        print(
            "RESULT: REGRESSION TEST SUITE FAILED"
        )

        print(
            "=" * 78
        )

        return False

    # =========================================================================
    # SUCCESSFUL SUITE
    # =========================================================================

    print(
        "RESULT: REGRESSION TEST SUITE PASSED"
    )

    print(
        "All RAG regression tests passed successfully."
    )

    print("=" * 78)

    return True


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    success = run_regression_tests()

    if not success:

        raise SystemExit(1)

    raise SystemExit(0)