from conversation.conversation_manager import (
    ConversationManager
)


def main():

    print("=" * 70)
    print("SBI INSURANCE POLICY - CONVERSATION MANAGER TEST")
    print("=" * 70)

    conversation = ConversationManager(
        max_messages=6
    )

    # --------------------------------------------------
    # Message 1
    # --------------------------------------------------

    conversation.add_user_message(
        "What medical expenses are covered?"
    )

    conversation.add_assistant_message(
        "The policy covers accident and sickness "
        "medical expenses, emergency medical evacuation "
        "and other covered medical expenses."
    )

    # --------------------------------------------------
    # Message 2
    # --------------------------------------------------

    conversation.add_user_message(
        "What about emergency evacuation?"
    )

    conversation.add_assistant_message(
        "Emergency medical evacuation covers the "
        "additional expenses required to transport "
        "the insured person to the nearest hospital "
        "or back to India, subject to policy conditions."
    )

    # --------------------------------------------------
    # Display history
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CONVERSATION HISTORY")
    print("=" * 70)

    print(
        conversation.get_history()
    )

    # --------------------------------------------------
    # Display messages
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MESSAGE OBJECTS")
    print("=" * 70)

    for message in conversation.get_messages():

        print(message)

    # --------------------------------------------------
    # Count
    # --------------------------------------------------

    print("\n")
    print(
        f"Message Count: "
        f"{conversation.message_count()}"
    )

    # --------------------------------------------------
    # Clear
    # --------------------------------------------------

    conversation.clear()

    print("\n")
    print(
        f"After Clear: "
        f"{conversation.message_count()} messages"
    )

    print("\nTEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()