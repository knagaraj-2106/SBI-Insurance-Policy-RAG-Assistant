from conversation.conversation_manager import ConversationManager


def main():

    print("=" * 70)
    print("SBI INSURANCE RAG - CONVERSATION MEMORY TEST")
    print("=" * 70)

    manager = ConversationManager(
        max_messages=10
    )

    # --------------------------------------------------
    # Exchange 1
    # --------------------------------------------------

    manager.add_exchange(
        user_query="What medical expenses are covered?",
        assistant_answer=(
            "The policy covers medically necessary "
            "medical expenses incurred overseas."
        )
    )

    # --------------------------------------------------
    # Exchange 2
    # --------------------------------------------------

    manager.add_exchange(
        user_query="What about emergency evacuation?",
        assistant_answer=(
            "Emergency medical evacuation is covered "
            "subject to the policy conditions and limits."
        )
    )

    # --------------------------------------------------
    # Display history
    # --------------------------------------------------

    print("\nCONVERSATION HISTORY")
    print("=" * 70)

    print(
        manager.get_history()
    )

    # --------------------------------------------------
    # Display messages
    # --------------------------------------------------

    print("\nRAW MESSAGES")
    print("=" * 70)

    for message in manager.get_messages():

        print(message)

    # --------------------------------------------------
    # Clear
    # --------------------------------------------------

    manager.clear_history()

    print("\nAFTER CLEAR")
    print("=" * 70)

    print(
        manager.get_history()
    )

    print("\nTEST COMPLETED")


if __name__ == "__main__":
    main()