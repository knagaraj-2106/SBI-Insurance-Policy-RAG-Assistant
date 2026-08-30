from evaluation.rag_evaluator import (
    RAGEvaluator
)


def main():

    print("=" * 70)
    print("SBI INSURANCE POLICY - RAG EVALUATION")
    print("=" * 70)

    evaluator = RAGEvaluator()

    report = evaluator.evaluate()

    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Average Keyword Score: "
        f"{report['average_keyword_score']:.2%}"
    )

    for result in report["results"]:

        print("\n")
        print(
            f"Test Case: {result['id']}"
        )

        print(
            f"Question: {result['question']}"
        )

        print(
            f"Keyword Score: "
            f"{result['keyword_score']:.2%}"
        )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()