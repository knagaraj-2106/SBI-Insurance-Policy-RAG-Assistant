from rag.rag_service import RAGService


def main():

    print("=" * 70)
    print("SBI INSURANCE RAG - QUERY ROUTING TEST")
    print("=" * 70)

    service = RAGService()

    # =========================================================
    # TEST 1 — INSURANCE QUERY
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 1 - INSURANCE QUERY")
    print("=" * 70)

    response = service.query(
        user_query=(
            "What medical expenses are covered "
            "under the Travel Insurance Policy?"
        ),
        policy_type="Travel Insurance Policy"
    )

    print("\nANSWER:")
    print(response.answer)

    print("\nSOURCES:")
    for source in response.sources:
        print(source)

    # =========================================================
    # TEST 2 — OUT OF DOMAIN
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 2 - OUT OF DOMAIN")
    print("=" * 70)

    response = service.query(
        user_query=(
            "What is the weather in Bangalore today?"
        ),
        policy_type="Travel Insurance Policy"
    )

    print("\nANSWER:")
    print(response.answer)

    print("\nSOURCES:")
    print(response.sources)

    # =========================================================
    # TEST 3 — FOLLOW UP
    # =========================================================

    print("\n" + "=" * 70)
    print("TEST 3 - FOLLOW UP")
    print("=" * 70)

    conversation_history = """
User: What medical expenses are covered?
Assistant: The policy covers medically necessary medical
expenses incurred overseas.
"""

    response = service.query(
        user_query="What about emergency evacuation?",
        policy_type="Travel Insurance Policy",
        conversation_history=conversation_history
    )

    print("\nANSWER:")
    print(response.answer)

    print("\nSOURCES:")
    for source in response.sources:
        print(source)

    print("\n" + "=" * 70)
    print("QUERY ROUTING TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()