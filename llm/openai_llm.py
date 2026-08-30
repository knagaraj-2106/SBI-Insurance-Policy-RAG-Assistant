from langchain_openai import ChatOpenAI

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class OpenAILLM:

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0
        )

        logger.info(
            f"OpenAI LLM initialized with model: "
            f"{settings.LLM_MODEL}"
        )

    def generate(
        self,
        prompt: str
    ) -> str:

        if not prompt or not prompt.strip():

            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info(
            "Sending prompt to OpenAI LLM."
        )

        response = self.llm.invoke(prompt)

        answer = response.content.strip()

        logger.info(
            "LLM response generated successfully."
        )

        return answer