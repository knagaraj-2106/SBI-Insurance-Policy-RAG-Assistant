from ingestion.pdf_loader import PDFLoader


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "PDF INGESTION TEST"
    )

    print("=" * 70)

    loader = PDFLoader()

    documents = loader.load_all_documents()

    print()

    print(
        f"Total pages loaded: {len(documents)}"
    )

    print()

    if not documents:

        print(
            "❌ No PDF documents were loaded."
        )

        return

    print(
        "✅ PDF ingestion successful."
    )

    print()

    print("First document metadata:")

    print(
        documents[0].metadata
    )

    print()

    print("First document content preview:")

    print(
        documents[0].page_content[:500]
    )

    print()

    print("=" * 70)


if __name__ == "__main__":

    main()