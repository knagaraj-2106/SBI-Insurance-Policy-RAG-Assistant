from embeddings.openai_embeddings import EmbeddingService


def main():

    print("=" * 70)
    print("SBI INSURANCE POLICY - OPENAI EMBEDDING TEST")
    print("=" * 70)

    embedding_service = EmbeddingService()

    test_text = (
        "SBI insurance policy provides coverage "
        "subject to the terms and conditions "
        "specified in the policy document."
    )

    print("\nGenerating embedding...")

    vector = embedding_service.embed_query(
        test_text
    )

    print("\n✅ Embedding generation successful.")

    print(
        f"Embedding dimensions: {len(vector)}"
    )

    print(
        f"First 5 values: {vector[:5]}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()