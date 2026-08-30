from retrieval.semantic_retriever import SemanticRetriever
from reranking.llm_reranker import LLMReranker


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "RERANKING TEST"
    )

    print("=" * 70)

    query = (
        "What medical expenses are covered "
        "under the Travel Insurance Policy?"
    )

    policy_type = "Travel Insurance Policy"

    # --------------------------------------------------
    # Step 1: Retrieve candidate documents
    # --------------------------------------------------

    retriever = SemanticRetriever()

    documents = retriever.retrieve(
        query=query,
        top_k=10,
        policy_type=policy_type
    )

    print(
        f"\nInitial candidates retrieved: "
        f"{len(documents)}"
    )

    # --------------------------------------------------
    # Step 2: Rerank candidates
    # --------------------------------------------------

    reranker = LLMReranker()

    ranked_documents = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5
    )

    # --------------------------------------------------
    # Step 3: Display results
    # --------------------------------------------------

    print(
        f"\nFinal documents after reranking: "
        f"{len(ranked_documents)}"
    )

    print("\n" + "=" * 70)

    for index, document in enumerate(
        ranked_documents,
        start=1
    ):

        print(
            f"\nRESULT #{index}"
        )

        print("-" * 70)

        print(
            "Policy Type:",
            document.metadata.get(
                "policy_type"
            )
        )

        print(
            "Page:",
            document.metadata.get(
                "page_number"
            )
        )

        print(
            "Rerank Score:",
            document.metadata.get(
                "rerank_score"
            )
        )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:700]
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()