from retrieval.semantic_retriever import SemanticRetriever


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "FILTERED RETRIEVAL TEST"
    )

    print("=" * 70)

    retriever = SemanticRetriever()

    query = (
        "What medical expenses are covered "
        "under the insurance policy?"
    )

    policy_type = "Travel Insurance Policy"

    print(
        f"\nUser Query:\n{query}"
    )

    print(
        f"\nPolicy Filter:\n{policy_type}"
    )

    documents = retriever.retrieve(
        query=query,
        top_k=5,
        policy_type=policy_type
    )

    print(
        f"\nRetrieved Documents: "
        f"{len(documents)}"
    )

    print("\n" + "=" * 70)

    for index, document in enumerate(
        documents,
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
            "Document:",
            document.metadata.get(
                "document_name"
            )
        )

        print("\nContent:")

        print(
            document.page_content[:700]
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":

    main()