from rag.indexing_pipeline import IndexingPipeline


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "INDEXING PIPELINE TEST"
    )

    print("=" * 70)

    pipeline = IndexingPipeline()

    result = pipeline.run()

    print()

    print("✅ INDEXING COMPLETED")

    print(
        f"Pages processed : {result['pages']}"
    )

    print(
        f"Chunks created  : {result['chunks']}"
    )

    print(
        f"Vectors stored  : {result['vectors']}"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()