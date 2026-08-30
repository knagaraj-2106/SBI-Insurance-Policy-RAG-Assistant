from evaluation.evaluation_dataset import (
    EVALUATION_DATASET
)

from evaluation.answer_evaluator import (
    AnswerEvaluator
)

from evaluation.retrieval_evaluator import (
    RetrievalEvaluator
)

from rag.rag_service import RAGService


def main():

    print("=" * 70)
    print("SBI INSURANCE RAG - EVALUATION")
    print("=" * 70)

    rag_service = RAGService()

    answer_evaluator = AnswerEvaluator()

    retrieval_evaluator = RetrievalEvaluator()

    total_answer_score = 0.0

    total_retrieval_score = 0.0

    total_questions = len(
        EVALUATION_DATASET
    )

    for item in EVALUATION_DATASET:

        print("\n")
        print("-" * 70)

        print(
            f"QUESTION {item['id']}"
        )

        print("-" * 70)

        question = item["question"]

        print(
            f"Query: {question}"
        )

        response = rag_service.query(
            user_query=question,
            policy_type=item["policy_type"]
        )

        answer = response.answer

        print(
            f"\nAnswer:\n{answer}"
        )

        # ----------------------------------------------
        # Answer evaluation
        # ----------------------------------------------

        answer_score = (
            answer_evaluator.keyword_match_score(
                answer=answer,
                expected_keywords=item[
                    "expected_keywords"
                ]
            )
        )

        # ----------------------------------------------
        # Retrieval evaluation
        # ----------------------------------------------

        # Current RAGResponse contains final sources.
        # This first evaluator therefore evaluates
        # answer-level signals rather than raw retrieval.

        retrieval_score = (
            1.0
            if response.sources
            else 0.0
        )

        total_answer_score += answer_score

        total_retrieval_score += retrieval_score

        print(
            f"\nAnswer Score: "
            f"{answer_score:.2f}"
        )

        print(
            f"Retrieval Score: "
            f"{retrieval_score:.2f}"
        )

    # ==================================================
    # Final metrics
    # ==================================================

    average_answer_score = (
        total_answer_score /
        total_questions
    )

    average_retrieval_score = (
        total_retrieval_score /
        total_questions
    )

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Total Questions: "
        f"{total_questions}"
    )

    print(
        f"Average Answer Score: "
        f"{average_answer_score:.2%}"
    )

    print(
        f"Average Retrieval Score: "
        f"{average_retrieval_score:.2%}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()