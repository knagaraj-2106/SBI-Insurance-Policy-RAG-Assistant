from guardrails.relevance_guard import RelevanceGuard
from langchain_core.documents import Document


def create_document(score):

    return Document(
        page_content="Sample insurance policy content.",
        metadata={
            "rerank_score": score
        }
    )


def run_test(
    test_name,
    scores,
    expected
):

    print("\n" + "=" * 70)
    print(test_name)
    print("=" * 70)

    documents = [
        create_document(score)
        for score in scores
    ]

    guard = RelevanceGuard(
        minimum_score=5,
        minimum_relevant_documents=1,
        minimum_average_score=5.0
    )

    result = guard.validate(
        documents
    )

    print(
        f"Scores: {scores}"
    )

    print(
        f"Expected: {expected}"
    )

    print(
        f"Actual: {result}"
    )

    if result == expected:

        print("TEST PASSED")

    else:

        print("TEST FAILED")


def main():

    # -------------------------------------------------
    # Test 1: Strongly relevant documents
    # -------------------------------------------------

    run_test(
        test_name="TEST 1 - Strongly Relevant",
        scores=[10, 9, 8, 7, 6],
        expected=True
    )

    # -------------------------------------------------
    # Test 2: Completely irrelevant documents
    # -------------------------------------------------

    run_test(
        test_name="TEST 2 - Completely Irrelevant",
        scores=[2, 1, 1, 0, 0],
        expected=False
    )

    # -------------------------------------------------
    # Test 3: One relevant document
    # -------------------------------------------------

    run_test(
    test_name="TEST 3 - Only One Relevant Document",
    scores=[10, 3, 2, 1, 0],
    expected=False
)

    # -------------------------------------------------
    # Test 4: Borderline documents
    # -------------------------------------------------

    run_test(
        test_name="TEST 4 - Borderline Relevance",
        scores=[5, 5, 4, 4, 3],
        expected=False
    )

    # -------------------------------------------------
    # Test 5: Strong average relevance
    # -------------------------------------------------

    run_test(
        test_name="TEST 5 - Strong Average Relevance",
        scores=[8, 8, 7, 6, 5],
        expected=True
    )


if __name__ == "__main__":
    main()