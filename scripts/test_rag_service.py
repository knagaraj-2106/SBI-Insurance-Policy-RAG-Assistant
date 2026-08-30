from rag.rag_service import RAGService


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "END-TO-END RAG TEST"
    )

    print("=" * 70)

    # =========================================================
    # TEST INPUT
    # =========================================================

    query = (
        "What medical expenses are covered under "
        "the Travel Insurance Policy?"
    )

    policy_type = "Travel Insurance Policy"

    # =========================================================
    # INITIALIZE RAG SERVICE
    # =========================================================

    print("\n" + "=" * 70)

    print("INITIALIZING RAG SERVICE")

    print("=" * 70)

    rag_service = RAGService()

    print(
        "RAG Service initialized successfully."
    )

    # =========================================================
    # EXECUTE RAG QUERY
    # =========================================================

    print("\n" + "=" * 70)

    print("EXECUTING RAG QUERY")

    print("=" * 70)

    print(
        "Policy Type:",
        policy_type
    )

    print(
        "User Query:",
        query
    )

    response = rag_service.query(
        user_query=query,
        policy_type=policy_type
    )

    # =========================================================
    # ORIGINAL QUERY
    # =========================================================

    print("\n" + "=" * 70)

    print("ORIGINAL USER QUERY")

    print("=" * 70)

    print(
        response.original_query
    )

    # =========================================================
    # REWRITTEN QUERY
    # =========================================================

    print("\n" + "=" * 70)

    print("REWRITTEN QUERY")

    print("=" * 70)

    print(
        response.rewritten_query
    )

    # =========================================================
    # QUERY CLASSIFICATION
    # =========================================================

    print("\n" + "=" * 70)

    print("QUERY CLASSIFICATION")

    print("=" * 70)

    print(
        "Category:",
        response.query_category
    )

    print(
        "Confidence:",
        response.classifier_confidence
    )

    print(
        "Reason:",
        response.classification_reason
    )

    # =========================================================
    # RETRIEVAL INFORMATION
    # =========================================================

    print("\n" + "=" * 70)

    print("RETRIEVAL INFORMATION")

    print("=" * 70)

    print(
        "Retrieved Documents:",
        response.retrieved_document_count
    )

    print(
        "Reranked Documents:",
        response.reranked_document_count
    )

    # =========================================================
    # GUARDRAIL VALIDATION STATUS
    # =========================================================

    print("\n" + "=" * 70)

    print("GUARDRAIL VALIDATION")

    print("=" * 70)

    print(
        "Relevance Passed:",
        response.relevance_passed
    )

    print(
        "Grounding Passed:",
        response.grounding_passed
    )

    print(
        "Groundedness Passed:",
        response.groundedness_passed
    )

    print(
        "Citation Passed:",
        response.citation_passed
    )

    # =========================================================
    # FINAL ANSWER
    # =========================================================

    print("\n" + "=" * 70)

    print("FINAL ANSWER")

    print("=" * 70)

    print(
        response.answer
    )

    # =========================================================
    # SOURCES
    # =========================================================

    print("\n" + "=" * 70)

    print("SOURCES")

    print("=" * 70)

    if not response.sources:

        print(
            "No sources available."
        )

    else:

        for index, source in enumerate(
            response.sources,
            start=1
        ):

            print(
                f"\nSource #{index}"
            )

            print(
                "-" * 70
            )

            print(
                "Source ID:",
                source.source_id
            )

            print(
                "Policy Type:",
                source.policy_type
            )

            print(
                "Document:",
                source.document_name
            )

            print(
                "Page:",
                source.page_number
            )

            print(
                "Rerank Score:",
                source.rerank_score
            )

            print(
                "Source:",
                source.source
            )

    # =========================================================
    # FINAL TEST STATUS
    # =========================================================

    print("\n" + "=" * 70)

    print("FINAL TEST STATUS")

    print("=" * 70)

    all_guards_passed = (
        response.relevance_passed
        and response.grounding_passed
        and response.groundedness_passed
        and response.citation_passed
    )

    if all_guards_passed:

        print(
            "RESULT: TEST PASSED"
        )

        print(
            "All RAG guardrails passed successfully."
        )

    else:

        print(
            "RESULT: TEST FAILED"
        )

        print(
            "One or more RAG guardrails failed."
        )

        print(
            "\nGuardrail Status:"
        )

        print(
            "  Relevance:",
            response.relevance_passed
        )

        print(
            "  Grounding:",
            response.grounding_passed
        )

        print(
            "  Groundedness:",
            response.groundedness_passed
        )

        print(
            "  Citation:",
            response.citation_passed
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":

    main()