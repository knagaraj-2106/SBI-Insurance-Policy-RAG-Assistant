from query.query_rewriter import QueryRewriter


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "QUERY REWRITER TEST"
    )

    print("=" * 70)

    rewriter = QueryRewriter()

    conversation_history = """
User: What medical expenses are covered
under the travel insurance policy?

Assistant: The policy provides coverage
for specified medical expenses subject
to the policy terms and conditions.
"""

    query = (
        "What about dental treatment?"
    )

    policy_type = "Travel Insurance Policy"

    print(
        f"\nOriginal Query:\n{query}"
    )

    print(
        f"\nPolicy:\n{policy_type}"
    )

    rewritten_query = rewriter.rewrite(
        query=query,
        conversation_history=conversation_history,
        policy_type=policy_type
    )

    print(
        f"\nRewritten Query:\n"
        f"{rewritten_query}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":

    main()