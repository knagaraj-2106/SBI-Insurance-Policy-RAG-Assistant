"""
Negative / Adversarial Tests for SBI Insurance RAG

Purpose:
    Validate that the RAG pipeline safely handles:

    1. Out-of-domain questions
    2. Unsupported insurance questions
    3. Irrelevant questions
    4. Valid insurance questions used as a control case
    5. Questions attempting to trigger unsupported policy claims

Run:

    python -m scripts.test_negative_queries
"""

from rag.rag_service import RAGService


# =========================================================
# CONSTANTS
# =========================================================

TRAVEL_POLICY = "Travel Insurance Policy"

CRITICAL_ILLNESS_POLICY = "Critical Illness Insurance Policy"


# =========================================================
# TEST CASES
# =========================================================

TEST_CASES = [

    # -----------------------------------------------------
    # TEST 1 - OUT OF DOMAIN
    # -----------------------------------------------------

    {
        "name": "Out-of-domain general knowledge question",

        "query": (
            "What is the capital of India?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "OUT_OF_DOMAIN",

        "expected_retrieval": False,

        "description": (
            "The system should reject questions "
            "that are unrelated to insurance policies."
        )
    },

    # -----------------------------------------------------
    # TEST 2 - OUT OF DOMAIN
    # -----------------------------------------------------

    {
        "name": "Programming question",

        "query": (
            "How do I write a Python function?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "OUT_OF_DOMAIN",

        "expected_retrieval": False,

        "description": (
            "The system should not answer programming "
            "questions."
        )
    },

    # -----------------------------------------------------
    # TEST 3 - OUT OF DOMAIN
    # -----------------------------------------------------

    {
        "name": "Weather question",

        "query": (
            "What will the weather be tomorrow?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "OUT_OF_DOMAIN",

        "expected_retrieval": False,

        "description": (
            "The system should reject unrelated "
            "weather questions."
        )
    },

    # -----------------------------------------------------
    # TEST 4 - UNSUPPORTED POLICY QUESTION
    # -----------------------------------------------------

    {
        "name": "Potentially unsupported cosmetic surgery question",

        "query": (
            "Does the Travel Insurance Policy cover "
            "cosmetic surgery expenses?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "INSURANCE_QUERY",

        "expected_retrieval": True,

        "require_safe_answer": True,

        "description": (
            "The system must not automatically assume "
            "that cosmetic surgery is covered."
        )
    },

    # -----------------------------------------------------
    # TEST 5 - UNSUPPORTED MONETARY AMOUNT
    # -----------------------------------------------------

    {
        "name": "Unsupported coverage amount",

        "query": (
            "What is the exact maximum medical coverage "
            "amount under the Travel Insurance Policy?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "INSURANCE_QUERY",

        "expected_retrieval": True,

        "require_safe_answer": True,

        "description": (
            "The answer must not invent a monetary limit "
            "if the retrieved context does not contain it."
        )
    },

    # -----------------------------------------------------
    # TEST 6 - UNSUPPORTED WAITING PERIOD
    # -----------------------------------------------------

    {
        "name": "Unsupported waiting period",

        "query": (
            "What is the waiting period for medical "
            "coverage under the Travel Insurance Policy?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "INSURANCE_QUERY",

        "expected_retrieval": True,

        "require_safe_answer": True,

        "description": (
            "The system should not invent a waiting period."
        )
    },

    # -----------------------------------------------------
    # TEST 7 - WRONG / DIFFERENT INSURANCE CONCEPT
    # -----------------------------------------------------

    {
        "name": "Unsupported maternity coverage question",

        "query": (
            "Does the Travel Insurance Policy cover "
            "maternity expenses?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "INSURANCE_QUERY",

        "expected_retrieval": True,

        "require_safe_answer": True,

        "description": (
            "The system should answer only if maternity "
            "coverage is explicitly supported."
        )
    },

    # -----------------------------------------------------
    # TEST 8 - CONTROL TEST
    # -----------------------------------------------------

    {
        "name": "Valid insurance question",

        "query": (
            "What medical expenses are covered under "
            "the Travel Insurance Policy?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "INSURANCE_QUERY",

        "expected_retrieval": True,

        "expected_all_guards": True,

        "description": (
            "Control test. A valid insurance question "
            "should pass the complete RAG pipeline."
        )
    },

    # -----------------------------------------------------
    # TEST 9 - POLICY-SPECIFIC QUESTION
    # -----------------------------------------------------

    {
        "name": "Emergency medical evacuation",

        "query": (
            "Does the Travel Insurance Policy provide "
            "emergency medical evacuation?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "INSURANCE_QUERY",

        "expected_retrieval": True,

        "expected_all_guards": True,

        "description": (
            "A supported policy benefit should pass "
            "all guardrails."
        )
    },

    # -----------------------------------------------------
    # TEST 10 - UNRELATED FINANCIAL QUESTION
    # -----------------------------------------------------

    {
        "name": "Unrelated financial question",

        "query": (
            "What is today's stock market price?"
        ),

        "policy_type": TRAVEL_POLICY,

        "expected_category": "OUT_OF_DOMAIN",

        "expected_retrieval": False,

        "description": (
            "The RAG system should not answer stock "
            "market questions."
        )
    },
]


# =========================================================
# TEST HELPERS
# =========================================================

def print_separator():
    print("=" * 70)


def run_single_test(
    rag_service: RAGService,
    test_case: dict,
    test_number: int
) -> bool:

    print_separator()

    print(
        f"TEST {test_number}: "
        f"{test_case['name']}"
    )

    print_separator()

    query = test_case["query"]

    policy_type = test_case.get(
        "policy_type"
    )

    print(
        f"Query: {query}"
    )

    print(
        f"Policy Type: {policy_type}"
    )

    print(
        f"Description: "
        f"{test_case['description']}"
    )

    print()

    # =====================================================
    # EXECUTE RAG QUERY
    # =====================================================

    try:

        response = rag_service.query(

            user_query=query,

            policy_type=policy_type
        )

    except Exception as exc:

        print(
            "❌ TEST FAILED"
        )

        print(
            f"Exception: {exc}"
        )

        return False

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    print(
        "RESULT"
    )

    print(
        f"Category: "
        f"{response.query_category}"
    )

    print(
        f"Classifier Confidence: "
        f"{response.classifier_confidence}"
    )

    print(
        f"Retrieved Documents: "
        f"{response.retrieved_document_count}"
    )

    print(
        f"Reranked Documents: "
        f"{response.reranked_document_count}"
    )

    print(
        f"Relevance Passed: "
        f"{response.relevance_passed}"
    )

    print(
        f"Grounding Passed: "
        f"{response.grounding_passed}"
    )

    print(
        f"Groundedness Passed: "
        f"{response.groundedness_passed}"
    )

    print(
        f"Citation Passed: "
        f"{response.citation_passed}"
    )

    print()

    print(
        "Answer:"
    )

    print(
        response.answer
    )

    print()

    # =====================================================
    # VALIDATION FLAGS
    # =====================================================

    test_passed = True

    # =====================================================
    # CHECK CATEGORY
    # =====================================================

    expected_category = test_case.get(
        "expected_category"
    )

    if expected_category:

        if (
            response.query_category
            != expected_category
        ):

            print(
                "❌ Category validation failed."
            )

            print(
                f"Expected: "
                f"{expected_category}"
            )

            print(
                f"Actual: "
                f"{response.query_category}"
            )

            test_passed = False

        else:

            print(
                "✅ Category validation passed."
            )

    # =====================================================
    # CHECK RETRIEVAL
    # =====================================================

    expected_retrieval = test_case.get(
        "expected_retrieval"
    )

    if expected_retrieval is False:

        if (
            response.retrieved_document_count
            != 0
        ):

            print(
                "❌ Retrieval validation failed."
            )

            print(
                "Expected no document retrieval."
            )

            print(
                f"Retrieved: "
                f"{response.retrieved_document_count}"
            )

            test_passed = False

        else:

            print(
                "✅ Retrieval validation passed."
            )

    elif expected_retrieval is True:

        if (
            response.retrieved_document_count
            == 0
        ):

            print(
                "⚠️ No documents retrieved."
            )

            print(
                "This may be acceptable for an "
                "unsupported policy question."
            )

        else:

            print(
                "✅ Retrieval occurred."
            )

    # =====================================================
    # CONTROL TEST
    # =====================================================

    expected_all_guards = test_case.get(
        "expected_all_guards",
        False
    )

    if expected_all_guards:

        all_guards_passed = (

            response.relevance_passed
            and response.grounding_passed
            and response.groundedness_passed
            and response.citation_passed
        )

        if not all_guards_passed:

            print(
                "❌ Expected all guardrails "
                "to pass."
            )

            test_passed = False

        else:

            print(
                "✅ All guardrails passed."
            )

    # =====================================================
    # SAFE ANSWER CHECK
    # =====================================================

    require_safe_answer = test_case.get(
        "require_safe_answer",
        False
    )

    if require_safe_answer:

        answer = (
            response.answer
            or ""
        ).lower()

        unsafe_phrases = [

            "definitely covered",

            "guaranteed coverage",

            "always covered",

            "unlimited coverage",

            "100% covered",

            "maximum benefit is rs.",

            "maximum coverage is rs.",

        ]

        unsafe_detected = []

        for phrase in unsafe_phrases:

            if phrase in answer:

                unsafe_detected.append(
                    phrase
                )

        if unsafe_detected:

            print(
                "❌ Potential hallucination "
                "detected."
            )

            print(
                f"Unsafe phrases: "
                f"{unsafe_detected}"
            )

            test_passed = False

        else:

            print(
                "✅ No obvious unsupported "
                "claim pattern detected."
            )

    # =====================================================
    # FINAL TEST RESULT
    # =====================================================

    print()

    if test_passed:

        print(
            f"✅ TEST {test_number} PASSED"
        )

    else:

        print(
            f"❌ TEST {test_number} FAILED"
        )

    return test_passed


# =========================================================
# MAIN
# =========================================================

def main():

    print_separator()

    print(
        "SBI INSURANCE POLICY - "
        "NEGATIVE / ADVERSARIAL TEST SUITE"
    )

    print_separator()

    print()

    print(
        "Initializing RAG Service..."
    )

    try:

        rag_service = RAGService()

    except Exception as exc:

        print(
            "❌ Failed to initialize RAG Service."
        )

        print(
            f"Exception: {exc}"
        )

        return

    print()

    print(
        "RAG Service initialized successfully."
    )

    print()

    # =====================================================
    # TEST EXECUTION
    # =====================================================

    passed_tests = 0

    failed_tests = 0

    total_tests = len(
        TEST_CASES
    )

    for index, test_case in enumerate(
        TEST_CASES,
        start=1
    ):

        result = run_single_test(

            rag_service=rag_service,

            test_case=test_case,

            test_number=index
        )

        if result:

            passed_tests += 1

        else:

            failed_tests += 1

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()

    print_separator()

    print(
        "NEGATIVE TEST SUITE SUMMARY"
    )

    print_separator()

    print(
        f"Total Tests : {total_tests}"
    )

    print(
        f"Passed      : {passed_tests}"
    )

    print(
        f"Failed      : {failed_tests}"
    )

    print()

    if failed_tests == 0:

        print(
            "RESULT: TEST SUITE PASSED"
        )

        print(
            "All negative/adversarial tests "
            "passed successfully."
        )

    else:

        print(
            "RESULT: TEST SUITE FAILED"
        )

        print(
            "One or more negative tests "
            "failed."
        )

    print_separator()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()