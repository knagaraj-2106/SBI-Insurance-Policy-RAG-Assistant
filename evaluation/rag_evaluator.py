from rag.rag_service import RAGService

from evaluation.evaluation_dataset import (
    TEST_CASES
)

from evaluation.metrics import (
    keyword_match,
    calculate_average
)


class RAGEvaluator:

    def __init__(self):

        self.rag_service = RAGService()

    def evaluate(self):

        results = []

        keyword_scores = []

        for test_case in TEST_CASES:

            print(
                f"\nRunning {test_case['id']}..."
            )

            response = self.rag_service.query(
                user_query=test_case["question"],
                policy_type=test_case[
                    "policy_type"
                ]
            )

            score = keyword_match(
                response.answer,
                test_case[
                    "expected_keywords"
                ]
            )

            keyword_scores.append(score)

            results.append(
                {
                    "id": test_case["id"],
                    "question": test_case[
                        "question"
                    ],
                    "answer": response.answer,
                    "keyword_score": score,
                    "sources": response.sources
                }
            )

        return {
            "results": results,
            "average_keyword_score":
                calculate_average(
                    keyword_scores
                )
        }