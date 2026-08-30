from guardrails.grounding_guard import GroundingGuard


def main():

    guard = GroundingGuard()

    # =================================================
    # TEST 1 - Fully Supported Answer
    # =================================================

    print("\n" + "=" * 70)
    print("TEST 1 - Fully Supported Answer")
    print("=" * 70)

    context = """
    The policy covers physician services,
    hospital services, medically necessary
    services and local emergency medical
    transportation.
    """

    answer = """
    The policy covers physician services,
    hospital services, medically necessary
    services and local emergency medical
    transportation.
    """

    result = guard.validate(
        answer=answer,
        context=context
    )

    print(f"Expected: True")
    print(f"Actual: {result}")

    assert result is True

    print("TEST PASSED")


    # =================================================
    # TEST 2 - Unsupported Answer
    # =================================================

    print("\n" + "=" * 70)
    print("TEST 2 - Unsupported Answer")
    print("=" * 70)

    context = """
    The policy covers physician services,
    hospital services and medically necessary
    medical transportation.
    """

    answer = """
    The policy covers cosmetic surgery
    and dental implants.
    """

    result = guard.validate(
        answer=answer,
        context=context
    )

    print(f"Expected: False")
    print(f"Actual: {result}")

    assert result is False

    print("TEST PASSED")


    # =================================================
    # TEST 3 - Partially Supported Answer
    # =================================================

    print("\n" + "=" * 70)
    print("TEST 3 - Partially Supported Answer")
    print("=" * 70)

    context = """
    The policy covers physician services
    and hospital services.
    """

    answer = """
    The policy covers physician services,
    hospital services and cosmetic surgery.
    """

    result = guard.validate(
        answer=answer,
        context=context
    )

    print(f"Expected: False")
    print(f"Actual: {result}")

    assert result is False

    print("TEST PASSED")


if __name__ == "__main__":
    main()