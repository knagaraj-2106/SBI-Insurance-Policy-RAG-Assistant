from guardrails.groundedness_guard import GroundednessGuard


def main():

    guard = GroundednessGuard()

    context = """
    Accident and Sickness Medical Expenses:

    The Insurer will indemnify the Insured person, up to
    the limit of cover shown in the Policy Schedule, in
    respect of medically necessary medical expenses
    incurred overseas for medical treatment.

    The expenses covered include physician's services,
    hospital services, medically necessary services and
    local emergency medical transportation.
    """

    # =================================================
    # TEST 1
    # =================================================

    answer_1 = """
    The policy covers medically necessary medical
    expenses incurred overseas. These include
    physician services, hospital services and
    local emergency medical transportation.
    """

    print("=" * 70)
    print("TEST 1 - Fully Supported Answer")
    print("=" * 70)

    result_1 = guard.validate(
        answer=answer_1,
        context=context
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

    answer_2 = """
    The policy covers medically necessary medical
    expenses incurred overseas and includes hospital
    and physician services.
    """

    print()
    print("=" * 70)
    print("TEST 2 - Supported Paraphrased Answer")
    print("=" * 70)

    result_2 = guard.validate(
        answer=answer_2,
        context=context
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

    answer_3 = """
    The policy covers medically necessary medical
    expenses overseas and provides a maximum benefit
    of Rs. 50 lakh for such expenses.
    """

    print()
    print("=" * 70)
    print("TEST 3 - Unsupported Information")
    print("=" * 70)

    result_3 = guard.validate(
        answer=answer_3,
        context=context
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

    answer_4 = """
    The policy covers emergency medical evacuation,
    repatriation of mortal remains and dental treatment.
    """

    print()
    print("=" * 70)
    print("TEST 4 - Mostly Unsupported Answer")
    print("=" * 70)

    result_4 = guard.validate(
        answer=answer_4,
        context=context
    )

    print("Expected: False")
    print(f"Actual: {result_4}")

    if not result_4:
        print("TEST PASSED")
    else:
        print("TEST FAILED")


if __name__ == "__main__":
    main()