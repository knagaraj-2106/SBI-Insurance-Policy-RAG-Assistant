"""
SBI Insurance RAG - Evaluation Test Suite

Purpose:
    Evaluate the complete SBI Insurance RAG pipeline.

The evaluation covers:

    1. Valid Travel Insurance questions
    2. Valid Critical Illness Insurance questions
    3. Unsupported / fabricated benefit questions
    4. Out-of-domain questions
    5. Policy filtering
    6. Relevance validation
    7. Grounding validation
    8. Groundedness validation
    9. Citation validation
    10. Final answer generation

This script uses the existing RAGService.

It does NOT duplicate the RAG pipeline logic.
"""


from rag.rag_service import RAGService


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # TEST 1 - Travel Insurance Medical Expenses
    # --------------------------------------------------------

    {
        "name": "Travel - Medical Expenses",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "What medical expenses are covered "
            "under the Travel Insurance Policy?"
        ),

        "expected_category": "INSURANCE_QUERY",

        "expect_answer": True,

        "expect_sources": True,

        "expected_relevance": True,

        "expected_grounding": True,

        "expected_groundedness": True,

        "expected_citation": True,
    },


    # --------------------------------------------------------
    # TEST 2 - Travel Insurance Emergency Evacuation
    # --------------------------------------------------------

    {
        "name": "Travel - Emergency Medical Evacuation",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "What is covered under Emergency "
            "Medical Evacuation?"
        ),

        "expected_category": "INSURANCE_QUERY",

        "expect_answer": True,

        "expect_sources": True,

        "expected_relevance": True,

        "expected_grounding": True,

        "expected_groundedness": True,

        "expected_citation": True,
    },


    # --------------------------------------------------------
    # TEST 3 - Travel Insurance Exclusions
    # --------------------------------------------------------

    {
        "name": "Travel - Exclusions",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "What are the exclusions under the "
            "Travel Insurance Policy?"
        ),

        "expected_category": "INSURANCE_QUERY",

        "expect_answer": True,

        "expect_sources": True,

        "expected_relevance": True,

        "expected_grounding": True,

        "expected_groundedness": True,

        "expected_citation": True,
    },


    # --------------------------------------------------------
    # TEST 4 - Travel Insurance Continuation Treatment
    # --------------------------------------------------------

    {
        "name": "Travel - Continuation of Medical Treatment",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "Does the Travel Insurance Policy cover "
            "continuation of medical treatment in India?"
        ),

        "expected_category": "INSURANCE_QUERY",

        "expect_answer": True,

        "expect_sources": True,

        "expected_relevance": True,

        "expected_grounding": True,

        "expected_groundedness": True,

        "expected_citation": True,
    },


    # --------------------------------------------------------
    # TEST 5 - Critical Illness Coverage
    # --------------------------------------------------------

    {
        "name": "Critical Illness - Coverage",

        "policy_type": "Critical Illness Insurance Policy",

        "query": (
            "What illnesses are covered under the "
            "Critical Illness Insurance Policy?"
        ),

        "expected_category": "INSURANCE_QUERY",

        "expect_answer": True,

        "expect_sources": True,

        "expected_relevance": True,

        "expected_grounding": True,

        "expected_groundedness": True,

        "expected_citation": True,
    },


    # --------------------------------------------------------
    # TEST 6 - Critical Illness Exclusions
    # --------------------------------------------------------

    {
        "name": "Critical Illness - Exclusions",

        "policy_type": "Critical Illness Insurance Policy",

        "query": (
            "What are the exclusions under the "
            "Critical Illness Insurance Policy?"
        ),

        "expected_category": "INSURANCE_QUERY",

        "expect_answer": True,

        "expect_sources": True,

        "expected_relevance": True,

        "expected_grounding": True,

        "expected_groundedness": True,

        "expected_citation": True,
    },


    # --------------------------------------------------------
    # TEST 7 - Unsupported / Fabricated Benefit
    # --------------------------------------------------------

    {
        "name": "Unsupported - Fabricated Benefit",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "Does the Travel Insurance Policy provide "
            "a Rs. 50 lakh benefit for dental treatment?"
        ),

        "expected_category": "INSURANCE_QUERY",

        # The system should still generate a response.
        # That response may be a refusal / insufficient
        # information message rather than a policy benefit.
        "expect_answer": True,

        # Sources are not mandatory for a refusal.
        "expect_sources": False,

        # This test specifically checks that the system
        # does not fabricate the requested Rs. 50 lakh benefit.
        "unsupported_query": True,
    },


    # --------------------------------------------------------
    # TEST 8 - Out-of-domain Weather
    # --------------------------------------------------------

    {
        "name": "Out of Scope - Weather",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "What will the weather be like in Bengaluru tomorrow?"
        ),

        "expected_category": "OUT_OF_DOMAIN",

        "expect_answer": True,

        "expect_sources": False,

        "out_of_domain": True,
    },


    # --------------------------------------------------------
    # TEST 9 - General Knowledge
    # --------------------------------------------------------

    {
        "name": "Out of Scope - General Knowledge",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "Who is the Prime Minister of India?"
        ),

        "expected_category": "OUT_OF_DOMAIN",

        "expect_answer": True,

        "expect_sources": False,

        "out_of_domain": True,
    },


    # --------------------------------------------------------
    # TEST 10 - Travel Insurance Repatriation
    # --------------------------------------------------------

    {
        "name": "Travel - Repatriation",

        "policy_type": "Travel Insurance Policy",

        "query": (
            "What does the Travel Insurance Policy say "
            "about repatriation of mortal remains?"
        ),

        "expected_category": "INSURANCE_QUERY",

        "expect_answer": True,

        "expect_sources": True,

        "expected_relevance": True,

        "expected_grounding": True,

        "expected_groundedness": True,

        "expected_citation": True,
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_separator():

    print("\n" + "=" * 70)


def print_test_header(test_number, test_case):

    print_separator()

    print(
        f"TEST {test_number} - "
        f"{test_case['name']}"
    )

    print("-" * 70)

    print(
        f"Policy Type: "
        f"{test_case['policy_type']}"
    )

    print(
        f"Query: "
        f"{test_case['query']}"
    )


def get_result_value(
    result,
    key,
    default=None
):
    """
    Safely retrieve values from:

        - Pydantic RAGResponse
        - Dictionary responses

    """

    if result is None:

        return default

    if isinstance(result, dict):

        return result.get(
            key,
            default
        )

    return getattr(
        result,
        key,
        default
    )


def normalize_category(category):
    """
    Normalize category values so that minor naming
    differences do not incorrectly fail evaluation.

    Current RAGService uses:

        OUT_OF_DOMAIN

    """

    if category is None:

        return None

    category = str(category).strip().upper()

    aliases = {

        "OUT_OF_SCOPE": "OUT_OF_DOMAIN",

        "OUT-OF-SCOPE": "OUT_OF_DOMAIN",

        "OUT OF SCOPE": "OUT_OF_DOMAIN",

        "OUT-OF-DOMAIN": "OUT_OF_DOMAIN",

        "OUT OF DOMAIN": "OUT_OF_DOMAIN",
    }

    return aliases.get(
        category,
        category
    )


def answer_contains_refusal(answer):
    """
    Check whether an unsupported / fabricated query
    received a safe response.

    This is intentionally not an exact string match.

    The RAG system may use different wording such as:

        - available policy context does not provide
          sufficient information
        - could not find relevant information
        - cannot verify
        - no information available
        - not specified in the policy

    """

    if not answer:

        return False

    answer_text = str(answer).lower()

    refusal_indicators = [

        "does not provide sufficient information",

        "does not provide enough information",

        "insufficient information",

        "could not find relevant information",

        "couldn't find relevant information",

        "not provide information",

        "not specified",

        "cannot verify",

        "could not verify",

        "couldn't verify",

        "not available",

        "not mentioned",

        "not found",

        "unable to determine",

        "cannot determine",

        "could not determine",
    ]

    return any(
        phrase in answer_text
        for phrase in refusal_indicators
    )


def answer_contains_fabricated_benefit(answer):
    """
    Detect the specific fabricated benefit that
    TEST 7 is trying to prevent.

    The test asks for:

        Rs. 50 lakh dental benefit

    The system should NOT confidently claim that
    the policy provides this benefit.
    """

    if not answer:

        return False

    answer_text = str(answer).lower()

    fabricated_patterns = [

        "rs. 50 lakh",

        "rs 50 lakh",

        "50 lakh benefit",

        "50,00,000 benefit",

        "₹50 lakh",

        "₹ 50 lakh",
    ]

    return any(
        phrase in answer_text
        for phrase in fabricated_patterns
    )


def validate_test_result(
    result,
    test_case
):
    """
    Validate one complete RAG test.

    Validation behavior depends on test type:

    --------------------------------------------------------
    Normal insurance query
    --------------------------------------------------------

        category
        answer
        relevance
        grounding
        groundedness
        citation
        sources

    --------------------------------------------------------
    Unsupported query
    --------------------------------------------------------

        category
        answer
        no fabricated benefit

    --------------------------------------------------------
    Out-of-domain query
    --------------------------------------------------------

        category
        safe answer
        no sources required
    """

    passed = True

    # ========================================================
    # EXTRACT RESULT VALUES
    # ========================================================

    # IMPORTANT:
    #
    # RAGResponse uses:
    #
    #     query_category
    #
    # NOT:
    #
    #     category
    #
    category = get_result_value(
        result,
        "query_category",
        None
    )

    category = normalize_category(
        category
    )

    answer = get_result_value(
        result,
        "answer",
        ""
    )

    relevance_passed = get_result_value(
        result,
        "relevance_passed",
        False
    )

    grounding_passed = get_result_value(
        result,
        "grounding_passed",
        False
    )

    groundedness_passed = get_result_value(
        result,
        "groundedness_passed",
        False
    )

    citation_passed = get_result_value(
        result,
        "citation_passed",
        False
    )

    sources = get_result_value(
        result,
        "sources",
        []
    )

    # ========================================================
    # CATEGORY VALIDATION
    # ========================================================

    expected_category = normalize_category(
        test_case.get(
            "expected_category"
        )
    )

    if expected_category:

        if category != expected_category:

            passed = False

            print(
                "\n❌ Category validation failed."
            )

            print(
                f"Expected: {expected_category}"
            )

            print(
                f"Actual:   {category}"
            )

        else:

            print(
                "\n✅ Category validation passed."
            )

    # ========================================================
    # ANSWER VALIDATION
    # ========================================================

    if test_case.get(
        "expect_answer",
        False
    ):

        if not answer or not str(answer).strip():

            passed = False

            print(
                "\n❌ Answer validation failed."
            )

            print(
                "Expected an answer, "
                "but no answer was generated."
            )

        else:

            print(
                "\n✅ Answer validation passed."
            )

    # ========================================================
    # OUT-OF-DOMAIN VALIDATION
    # ========================================================

    if test_case.get(
        "out_of_domain",
        False
    ):

        print(
            "\n🔒 Out-of-domain test detected."
        )

        # ----------------------------------------------------
        # Sources should NOT be required.
        # ----------------------------------------------------

        if sources:

            print(
                "⚠️ Out-of-domain query returned "
                f"{len(sources)} source(s)."
            )

            print(
                "This is not treated as a failure because "
                "the authoritative response category is "
                "still the primary requirement."
            )

        else:

            print(
                "✅ No policy sources returned."
            )

        return passed

    # ========================================================
    # UNSUPPORTED / FABRICATED BENEFIT VALIDATION
    # ========================================================

    if test_case.get(
        "unsupported_query",
        False
    ):

        print(
            "\n🛡️ Unsupported/fabricated benefit "
            "test detected."
        )

        # ----------------------------------------------------
        # Check whether the answer contains the fabricated
        # Rs. 50 lakh benefit as a factual claim.
        # ----------------------------------------------------

        if answer_contains_fabricated_benefit(
            answer
        ):

            # We need to distinguish between:

            # "The policy does NOT provide Rs. 50 lakh"
            #
            # and:
            #
            # "The policy provides Rs. 50 lakh"

            # A simple presence check is therefore not
            # automatically considered failure.

            answer_text = str(answer).lower()

            negative_context = [

                "does not",

                "doesn't",

                "not",

                "cannot verify",

                "cannot confirm",

                "could not verify",

                "couldn't verify",

                "no evidence",

                "not specified",

                "not mentioned",

                "insufficient",
            ]

            contains_negative_context = any(
                phrase in answer_text
                for phrase in negative_context
            )

            if contains_negative_context:

                print(
                    "✅ Fabricated benefit was not "
                    "confidently accepted."
                )

            else:

                passed = False

                print(
                    "❌ Potential fabricated benefit detected."
                )

                print(
                    "The answer appears to claim or "
                    "accept the requested Rs. 50 lakh "
                    "benefit."
                )

        else:

            print(
                "✅ No fabricated Rs. 50 lakh benefit "
                "detected."
            )

        # ----------------------------------------------------
        # We intentionally DO NOT require all guardrails
        # to be True for this test.
        #
        # The important requirement is:
        #
        #     DO NOT FABRICATE
        #
        # ----------------------------------------------------

        return passed

    # ========================================================
    # NORMAL INSURANCE QUERY GUARD VALIDATION
    # ========================================================

    # --------------------------------------------------------
    # Relevance
    # --------------------------------------------------------

    expected_relevance = test_case.get(
        "expected_relevance"
    )

    if expected_relevance is not None:

        if relevance_passed != expected_relevance:

            passed = False

            print(
                "\n❌ Relevance validation failed."
            )

            print(
                f"Expected: {expected_relevance}"
            )

            print(
                f"Actual:   {relevance_passed}"
            )

        else:

            print(
                "\n✅ Relevance validation passed."
            )

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    expected_grounding = test_case.get(
        "expected_grounding"
    )

    if expected_grounding is not None:

        if grounding_passed != expected_grounding:

            passed = False

            print(
                "\n❌ Grounding validation failed."
            )

            print(
                f"Expected: {expected_grounding}"
            )

            print(
                f"Actual:   {grounding_passed}"
            )

        else:

            print(
                "\n✅ Grounding validation passed."
            )

    # --------------------------------------------------------
    # Groundedness
    # --------------------------------------------------------

    expected_groundedness = test_case.get(
        "expected_groundedness"
    )

    if expected_groundedness is not None:

        if (
            groundedness_passed
            != expected_groundedness
        ):

            passed = False

            print(
                "\n❌ Groundedness validation failed."
            )

            print(
                f"Expected: {expected_groundedness}"
            )

            print(
                f"Actual:   {groundedness_passed}"
            )

        else:

            print(
                "\n✅ Groundedness validation passed."
            )

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    expected_citation = test_case.get(
        "expected_citation"
    )

    if expected_citation is not None:

        if citation_passed != expected_citation:

            passed = False

            print(
                "\n❌ Citation validation failed."
            )

            print(
                f"Expected: {expected_citation}"
            )

            print(
                f"Actual:   {citation_passed}"
            )

        else:

            print(
                "\n✅ Citation validation passed."
            )

    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    expect_sources = test_case.get(
        "expect_sources",
        False
    )

    if expect_sources:

        if not sources:

            passed = False

            print(
                "\n❌ Source validation failed."
            )

            print(
                "Expected supporting policy sources, "
                "but no sources were returned."
            )

        else:

            print(
                f"\n✅ Source validation passed."
                f" {len(sources)} source(s) returned."
            )

    else:

        print(
            "\nℹ️ Policy sources were not required "
            "for this test."
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return passed


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print_separator()

    print(
        "SBI INSURANCE POLICY - "
        "RAG EVALUATION TEST SUITE"
    )

    print_separator()

    print(
        f"\nTotal test cases: "
        f"{len(TEST_CASES)}"
    )

    # ========================================================
    # INITIALIZE RAG SERVICE
    # ========================================================

    print_separator()

    print(
        "INITIALIZING RAG SERVICE"
    )

    print_separator()

    rag_service = RAGService()

    print(
        "\nRAG Service initialized successfully."
    )

    # ========================================================
    # EVALUATION COUNTERS
    # ========================================================

    passed_tests = 0

    failed_tests = 0

    results = []

    # ========================================================
    # EXECUTE TEST CASES
    # ========================================================

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        print_test_header(
            test_number,
            test_case
        )

        try:

            result = rag_service.query(
                user_query=test_case["query"],
                policy_type=test_case["policy_type"]
            )

            # ==================================================
            # RESULT SUMMARY
            # ==================================================

            print("\nRESULT SUMMARY")

            print("-" * 70)

            category = get_result_value(
                result,
                "query_category",
                None
            )

            answer = get_result_value(
                result,
                "answer",
                ""
            )

            relevance_passed = get_result_value(
                result,
                "relevance_passed",
                False
            )

            grounding_passed = get_result_value(
                result,
                "grounding_passed",
                False
            )

            groundedness_passed = get_result_value(
                result,
                "groundedness_passed",
                False
            )

            citation_passed = get_result_value(
                result,
                "citation_passed",
                False
            )

            sources = get_result_value(
                result,
                "sources",
                []
            )

            retrieved_document_count = (
                get_result_value(
                    result,
                    "retrieved_document_count",
                    0
                )
            )

            reranked_document_count = (
                get_result_value(
                    result,
                    "reranked_document_count",
                    0
                )
            )

            retry_count = get_result_value(
                result,
                "retry_count",
                None
            )

            print(
                f"Category: "
                f"{category}"
            )

            print(
                f"Relevance: "
                f"{relevance_passed}"
            )

            print(
                f"Grounding: "
                f"{grounding_passed}"
            )

            print(
                f"Groundedness: "
                f"{groundedness_passed}"
            )

            print(
                f"Citation: "
                f"{citation_passed}"
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
                f"Sources: "
                f"{len(sources)}"
            )

            if retry_count is not None:

                print(
                    f"Retries: "
                    f"{retry_count}"
                )

            # ==================================================
            # DISPLAY ANSWER
            # ==================================================

            if answer:

                print(
                    "\nGenerated Answer:"
                )

                print("-" * 70)

                print(
                    answer
                )

            # ==================================================
            # VALIDATE TEST
            # ==================================================

            test_passed = validate_test_result(
                result,
                test_case
            )

            # ==================================================
            # TEST STATUS
            # ==================================================

            if test_passed:

                passed_tests += 1

                print(
                    "\nTEST STATUS: PASSED"
                )

            else:

                failed_tests += 1

                print(
                    "\nTEST STATUS: FAILED"
                )

            # ==================================================
            # SAVE RESULT
            # ==================================================

            results.append(
                {
                    "test_number": test_number,

                    "name": test_case["name"],

                    "passed": test_passed,

                    "category": category,

                    "relevance": relevance_passed,

                    "grounding": grounding_passed,

                    "groundedness": groundedness_passed,

                    "citation": citation_passed,

                    "sources": len(sources),
                }
            )

        except Exception as exc:

            failed_tests += 1

            print(
                "\nTEST STATUS: FAILED"
            )

            print(
                f"Error: {exc}"
            )

            results.append(
                {
                    "test_number": test_number,

                    "name": test_case["name"],

                    "passed": False,

                    "error": str(exc),
                }
            )

    # ========================================================
    # FINAL EVALUATION REPORT
    # ========================================================

    print_separator()

    print(
        "FINAL RAG EVALUATION REPORT"
    )

    print_separator()

    print(
        f"\nTotal Tests   : "
        f"{len(TEST_CASES)}"
    )

    print(
        f"Passed Tests  : "
        f"{passed_tests}"
    )

    print(
        f"Failed Tests  : "
        f"{failed_tests}"
    )

    if TEST_CASES:

        success_rate = (
            passed_tests /
            len(TEST_CASES)
        ) * 100

    else:

        success_rate = 0

    print(
        f"Success Rate  : "
        f"{success_rate:.2f}%"
    )

    # ========================================================
    # TEST RESULT TABLE
    # ========================================================

    print(
        "\nTest Results"
    )

    print("-" * 70)

    for item in results:

        status = (
            "PASSED"
            if item["passed"]
            else "FAILED"
        )

        print(
            f"{item['test_number']:02d}. "
            f"{item['name']:<45} "
            f"{status}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_separator()

    if failed_tests == 0:

        print(
            "RESULT: ALL RAG EVALUATION TESTS PASSED"
        )

    else:

        print(
            "RESULT: SOME RAG EVALUATION TESTS FAILED"
        )

    print_separator()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

