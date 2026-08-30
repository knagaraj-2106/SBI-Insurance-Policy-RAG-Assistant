from conversation.conversation_manager import (
    ConversationManager
)


def main():

    print("=" * 70)
    print("SBI INSURANCE RAG - CONVERSATION MANAGER TEST")
    print("=" * 70)

    conversation = ConversationManager(
        max_messages=6
    )

    # --------------------------------------------------
    # Add messages
    # --------------------------------------------------

    conversation.add_user_message(
        "What medical expenses are covered?"
    )

    conversation.add_assistant_message(
        "The policy covers medically necessary "
        "medical expenses incurred overseas."
    )

    conversation.add_user_message(
        "What about emergency evacuation?"
    )

    conversation.add_assistant_message(
        "Emergency medical evacuation is covered "
        "subject to the policy terms."
    )

    # --------------------------------------------------
    # Display messages
    # --------------------------------------------------

    print("\nMESSAGES")
    print("-" * 70)

    for message in conversation.get_messages():

        print(
            f"{message['role']}: "
            f"{message['content']}"
        )

    # --------------------------------------------------
    # Display formatted history
    # --------------------------------------------------

    print("\nCONVERSATION HISTORY")
    print("-" * 70)

    print(
        conversation.get_history()
    )

    # --------------------------------------------------
    # Clear
    # --------------------------------------------------

    conversation.clear()

    print("\nAFTER CLEAR")
    print("-" * 70)

    print(
        conversation.get_history()
        or "Conversation is empty."
    )


if __name__ == "__main__":
    main()