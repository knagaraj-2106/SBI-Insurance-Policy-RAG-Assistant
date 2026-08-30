from query.query_classifier import QueryClassifier


def main():

    print("=" * 70)
    print("SBI INSURANCE RAG - QUERY CLASSIFIER TEST")
    print("=" * 70)

    classifier = QueryClassifier()

    test_cases = [

        {
            "query": (
                "What medical expenses are covered "
                "under the Travel Insurance Policy?"
            ),
            "history": ""
        },

        {
            "query": (
                "What about emergency evacuation?"
            ),
            "history": (
                "User: What medical expenses are covered?\n"
                "Assistant: The policy covers medically "
                "necessary medical expenses."
            )
        },

        {
            "query": (
                "What is the weather in Bangalore today?"
            ),
            "history": ""
        },

        {
            "query": (
                "Is dental treatment covered?"
            ),
            "history": ""
        },

        {
            "query": (
                "Who won the cricket match yesterday?"
            ),
            "history": ""
        }
    ]

    for index, test_case in enumerate(
        test_cases,
        start=1
    ):

        print("\n" + "=" * 70)
        print(f"TEST {index}")
        print("=" * 70)

        print(
            f"Query: {test_case['query']}"
        )

        result = classifier.classify(
            query=test_case["query"],
            conversation_history=test_case["history"]
        )

        print(
            f"Category: {result['category']}"
        )

        print(
            f"Confidence: {result['confidence']}"
        )

        print(
            f"Reason: {result['reason']}"
        )


if __name__ == "__main__":
    main()