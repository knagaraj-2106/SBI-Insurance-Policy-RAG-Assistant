from ingestion.pdf_loader import PDFLoader
from chunking.text_chunker import TextChunker


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "CHUNKING TEST"
    )

    print("=" * 70)

    # ---------------------------------------------
    # Load PDF pages
    # ---------------------------------------------

    loader = PDFLoader()

    documents = loader.load_all_documents()

    print(
        f"\nPDF pages loaded: {len(documents)}"
    )

    # ---------------------------------------------
    # Create chunks
    # ---------------------------------------------

    chunker = TextChunker()

    chunks = chunker.create_chunks(
        documents
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    # ---------------------------------------------
    # Display first chunk
    # ---------------------------------------------

    if not chunks:

        print(
            "\n❌ No chunks were created."
        )

        return

    print(
        "\n✅ Chunking successful."
    )

    print("\nFirst chunk metadata:")

    print(
        chunks[0].metadata
    )

    print("\nFirst chunk content:")

    print(
        chunks[0].page_content[:1000]
    )

    # ---------------------------------------------
    # Display chunk size
    # ---------------------------------------------

    print(
        f"\nFirst chunk character count: "
        f"{len(chunks[0].page_content)}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()