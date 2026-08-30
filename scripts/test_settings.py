from config.settings import settings


def main():

    print("=" * 60)
    print("SBI INSURANCE RAG - CONFIGURATION TEST")
    print("=" * 60)

    print(f"Project Root       : {settings.BASE_DIR}")

    print(f"Raw Data Directory : {settings.RAW_DATA_DIR}")

    print(
        f"Processed Directory: {settings.PROCESSED_DATA_DIR}"
    )

    print(
        f"Vector Store       : {settings.VECTORSTORE_DIR}"
    )

    print(f"LLM Model          : {settings.LLM_MODEL}")

    print(
        f"Embedding Model    : {settings.EMBEDDING_MODEL}"
    )

    print(f"Chunk Size         : {settings.CHUNK_SIZE}")

    print(f"Chunk Overlap      : {settings.CHUNK_OVERLAP}")

    print(f"Top K              : {settings.TOP_K}")

    print(
        f"OpenAI API Key     : "
        f"{'Available' if settings.OPENAI_API_KEY else 'Missing'}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()