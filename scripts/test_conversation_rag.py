from rag.rag_service import RAGService
from conversation.conversation_manager import ConversationManager


def print_turn(turn_number, user_query, response):

    print()
    print("=" * 70)
    print(f"TURN {turn_number}")
    print("=" * 70)

    print()
    print("USER:")
    print(user_query)

    print()
    print("ASSISTANT:")

    if isinstance(response, dict):

        print(
            response.get(
                "answer",
                "No answer generated."
            )
        )

    else:

        print(
            getattr(
                response,
                "answer",
                "No answer generated."
            )
        )


def main():

    print("=" * 70)
    print("SBI INSURANCE RAG - CONVERSATION AWARE RAG TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Initialize RAG service
    # --------------------------------------------------

    rag_service = RAGService()

    # --------------------------------------------------
    # Initialize conversation manager
    # --------------------------------------------------

    conversation_manager = ConversationManager(
        max_messages=10
    )

    # ==================================================
    # TURN 1
    # ==================================================

    query_1 = (
        "What medical expenses are covered "
        "under the Travel Insurance Policy?"
    )

    # Get current conversation history
    history_1 = conversation_manager.get_history()

    response_1 = rag_service.query(
        user_query=query_1,
        policy_type="Travel Insurance Policy",
        conversation_history=history_1
    )

    print_turn(
        1,
        query_1,
        response_1
    )

    # --------------------------------------------------
    # Store Turn 1
    # --------------------------------------------------

    answer_1 = (
        response_1.get("answer")
        if isinstance(response_1, dict)
        else response_1.answer
    )

    conversation_manager.add_user_message(
        query_1
    )

    conversation_manager.add_assistant_message(
        answer_1
    )

    # ==================================================
    # TURN 2
    # ==================================================

    query_2 = (
        "What about emergency evacuation?"
    )

    # IMPORTANT:
    # Retrieve previous conversation
    history_2 = conversation_manager.get_history()

    print()
    print("-" * 70)
    print("CONVERSATION HISTORY BEFORE TURN 2")
    print("-" * 70)
    print(history_2)

    response_2 = rag_service.query(
        user_query=query_2,
        policy_type="Travel Insurance Policy",
        conversation_history=history_2
    )

    print_turn(
        2,
        query_2,
        response_2
    )

    # --------------------------------------------------
    # Store Turn 2
    # --------------------------------------------------

    answer_2 = (
        response_2.get("answer")
        if isinstance(response_2, dict)
        else response_2.answer
    )

    conversation_manager.add_user_message(
        query_2
    )

    conversation_manager.add_assistant_message(
        answer_2
    )

    # ==================================================
    # TURN 3
    # ==================================================

    query_3 = (
        "What conditions apply to it?"
    )

    history_3 = conversation_manager.get_history()

    print()
    print("-" * 70)
    print("CONVERSATION HISTORY BEFORE TURN 3")
    print("-" * 70)
    print(history_3)

    response_3 = rag_service.query(
        user_query=query_3,
        policy_type="Travel Insurance Policy",
        conversation_history=history_3
    )

    print_turn(
        3,
        query_3,
        response_3
    )

    # --------------------------------------------------
    # Store Turn 3
    # --------------------------------------------------

    answer_3 = (
        response_3.get("answer")
        if isinstance(response_3, dict)
        else response_3.answer
    )

    conversation_manager.add_user_message(
        query_3
    )

    conversation_manager.add_assistant_message(
        answer_3
    )

    # ==================================================
    # FINAL HISTORY
    # ==================================================

    print()
    print("=" * 70)
    print("FINAL CONVERSATION HISTORY")
    print("=" * 70)

    print(
        conversation_manager.get_history()
    )


if __name__ == "__main__":
    main()