import re


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:

        if not text:
            return ""

        # Normalize line breaks
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces
        text = re.sub(r"[ \t]+", " ", text)

        # Reduce excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text