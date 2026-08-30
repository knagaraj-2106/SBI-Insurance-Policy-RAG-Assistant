from guardrails.citation_guard import CitationGuard


def main():

    guard = CitationGuard()

    sources = [
        {
            "source_id": "S1",
            "policy_type": "Travel Insurance Policy",
            "document_name": "Travel Insurance Policy",
            "page_number": 23
        },
        {
            "source_id": "S2",
            "policy_type": "Travel Insurance Policy",
            "document_name": "Travel Insurance Policy",
            "page_number": 24
        }
    ]

    # =================================================
    # TEST 1
    # =================================================

    answer_1 = (
        "Medical expenses are covered under the policy. "
        "[S1]"
    )

    print("=" * 70)
    print("TEST 1 - Valid Citation")
    print("=" * 70)

    result_1 = guard.validate(
        answer=answer_1,
        sources=sources
    )

    print("Expected: True")
    print(f"Actual: {result_1}")

    if result_1:
        print("TEST PASSED")
    else:
        print("TEST FAILED")

    # =================================================
    # TEST 2
    # =================================================

    answer_2 = (
        "Medical expenses are covered. "
        "[S1][S2]"
    )

    print()
    print("=" * 70)
    print("TEST 2 - Multiple Valid Citations")
    print("=" * 70)

    result_2 = guard.validate(
        answer=answer_2,
        sources=sources
    )

    print("Expected: True")
    print(f"Actual: {result_2}")

    if result_2:
        print("TEST PASSED")
    else:
        print("TEST FAILED")

    # =================================================
    # TEST 3
    # =================================================

    answer_3 = (
        "Medical expenses are covered. "
        "[S999]"
    )

    print()
    print("=" * 70)
    print("TEST 3 - Invalid Citation")
    print("=" * 70)

    result_3 = guard.validate(
        answer=answer_3,
        sources=sources
    )

    print("Expected: False")
    print(f"Actual: {result_3}")

    if not result_3:
        print("TEST PASSED")
    else:
        print("TEST FAILED")

    # =================================================
    # TEST 4
    # =================================================

    answer_4 = (
        "Medical expenses are covered under the policy."
    )

    print()
    print("=" * 70)
    print("TEST 4 - Missing Citation")
    print("=" * 70)

    result_4 = guard.validate(
        answer=answer_4,
        sources=sources
    )

    print("Expected: False")
    print(f"Actual: {result_4}")

    if not result_4:
        print("TEST PASSED")
    else:
        print("TEST FAILED")

    # =================================================
    # TEST 5
    # =================================================

    answer_5 = (
        "Medical expenses are covered. "
        "[s1]"
    )

    print()
    print("=" * 70)
    print("TEST 5 - Lowercase Citation")
    print("=" * 70)

    result_5 = guard.validate(
        answer=answer_5,
        sources=sources
    )

    print("Expected: True")
    print(f"Actual: {result_5}")

    if result_5:
        print("TEST PASSED")
    else:
        print("TEST FAILED")


if __name__ == "__main__":
    main()