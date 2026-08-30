from typing import List, Dict

from utils.logger import get_logger


logger = get_logger(__name__)


class ConversationManager:

    def __init__(
        self,
        max_messages: int = 10
    ):

        self.max_messages = max_messages

        self.messages: List[Dict[str, str]] = []

        logger.info(
            f"Conversation manager initialized. "
            f"Maximum messages: {max_messages}"
        )

    # =================================================
    # ADD USER MESSAGE
    # =================================================

    def add_user_message(
        self,
        message: str
    ) -> None:

        if not message or not message.strip():
            return

        self.messages.append(
            {
                "role": "user",
                "content": message.strip()
            }
        )

        self._trim_messages()

        logger.info(
            "User message added to conversation."
        )

    # =================================================
    # ADD ASSISTANT MESSAGE
    # =================================================

    def add_assistant_message(
        self,
        message: str
    ) -> None:

        if not message or not message.strip():
            return

        self.messages.append(
            {
                "role": "assistant",
                "content": message.strip()
            }
        )

        self._trim_messages()

        logger.info(
            "Assistant message added to conversation."
        )

    # =================================================
    # ADD GENERIC MESSAGE
    # =================================================

    def add_message(
        self,
        role: str,
        content: str
    ) -> None:

        if role not in {"user", "assistant"}:

            raise ValueError(
                "Role must be either 'user' or 'assistant'."
            )

        if not content or not content.strip():
            return

        self.messages.append(
            {
                "role": role,
                "content": content.strip()
            }
        )

        self._trim_messages()

    # =================================================
    # TRIM CONVERSATION
    # =================================================

    def _trim_messages(self) -> None:

        if len(self.messages) > self.max_messages:

            self.messages = self.messages[
                -self.max_messages:
            ]

            logger.info(
                "Conversation history trimmed."
            )

    # =================================================
    # GET MESSAGES
    # =================================================

    def get_messages(
        self
    ) -> List[Dict[str, str]]:

        return list(self.messages)

    # =================================================
    # GET HISTORY AS TEXT
    # =================================================

    def get_history(
        self
    ) -> str:

        if not self.messages:

            return ""

        history = []

        for message in self.messages:

            role = message["role"].capitalize()

            content = message["content"]

            history.append(
                f"{role}: {content}"
            )

        return "\n".join(history)

    # =================================================
    # GET LAST USER MESSAGE
    # =================================================

    def get_last_user_message(
        self
    ) -> str:

        for message in reversed(self.messages):

            if message["role"] == "user":

                return message["content"]

        return ""

    # =================================================
    # GET LAST ASSISTANT MESSAGE
    # =================================================

    def get_last_assistant_message(
        self
    ) -> str:

        for message in reversed(self.messages):

            if message["role"] == "assistant":

                return message["content"]

        return ""

    # =================================================
    # CHECK EMPTY
    # =================================================

    def is_empty(
        self
    ) -> bool:

        return len(self.messages) == 0

    # =================================================
    # CLEAR
    # =================================================

    def clear(self) -> None:

        self.messages.clear()

        logger.info(
            "Conversation history cleared."
        )