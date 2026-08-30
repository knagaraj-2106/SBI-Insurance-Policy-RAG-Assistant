from typing import List

from langchain_core.documents import Document


def retrieval_page_match(
    documents: List[Document],
    expected_pages: List[int]
) -> bool:

    if not expected_pages:

        return len(documents) == 0

    retrieved_pages = set()

    for document in documents:

        page = document.metadata.get(
            "page_number"
        )

        if page is not None:

            try:
                retrieved_pages.add(
                    int(page)
                )
            except (ValueError, TypeError):
                continue

    return any(
        page in retrieved_pages
        for page in expected_pages
    )


def keyword_match(
    answer: str,
    expected_keywords: List[str]
) -> float:

    if not expected_keywords:

        return 1.0

    if not answer:

        return 0.0

    answer_lower = answer.lower()

    matched = sum(
        1
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    )

    return matched / len(
        expected_keywords
    )


def calculate_average(
    values: List[float]
) -> float:

    if not values:

        return 0.0

    return sum(values) / len(values)