from typing import List, Dict

from utils.logger import get_logger


logger = get_logger(__name__)


class ConversationMemory:

    def __init__(self, max_messages: int = 10):

        self.max_messages = max_messages

        self.messages: List[Dict[str, str]] = []

        logger.info(
            f"Conversation memory initialized. "
            f"Maximum messages: {max_messages}"
        )

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

        self._trim_history()

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

        self._trim_history()

    def _trim_history(self) -> None:

        if len(self.messages) > self.max_messages:

            self.messages = self.messages[
                -self.max_messages:
            ]

    def get_history(self) -> str:

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

    def get_messages(
        self
    ) -> List[Dict[str, str]]:

        return list(self.messages)

    def clear(self) -> None:

        self.messages.clear()

        logger.info(
            "Conversation memory cleared."
        )

    def is_empty(self) -> bool:

        return len(self.messages) == 0