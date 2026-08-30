from typing import List, Dict

from langchain_core.documents import Document

from utils.logger import get_logger


logger = get_logger(__name__)


class ContextBuilder:

    def __init__(self):

        logger.info(
            "Context builder initialized."
        )

    def build_context(
        self,
        documents: List[Document]
    ) -> str:

        if not documents:

            logger.warning(
                "No documents available "
                "for context construction."
            )

            return ""

        context_parts = []

        # -------------------------------------------------
        # Maintain a stable mapping between the actual
        # source document and Source ID.
        #
        # Same document = same Source ID
        # -------------------------------------------------

        source_id_map: Dict[str, str] = {}

        next_source_number = 1

        for index, document in enumerate(
            documents,
            start=1
        ):

            metadata = document.metadata

            policy_type = metadata.get(
                "policy_type",
                "Unknown Policy"
            )

            document_name = metadata.get(
                "document_name",
                "Unknown Document"
            )

            page_number = metadata.get(
                "page_number",
                "Unknown Page"
            )

            source = metadata.get(
                "source",
                "Unknown Source"
            )

            rerank_score = metadata.get(
                "rerank_score",
                "N/A"
            )

            content = (
                document.page_content.strip()
            )

            if not content:

                logger.warning(
                    f"Skipping empty document "
                    f"at context position {index}."
                )

                continue

            # -------------------------------------------------
            # Determine unique source key
            #
            # Prefer the document name because multiple chunks
            # from the same PDF should share one citation ID.
            # Fall back to source path if document name is absent.
            # -------------------------------------------------

            source_key = (
                str(document_name).strip()
                if document_name
                and document_name != "Unknown Document"
                else str(source).strip()
            )

            # -------------------------------------------------
            # Create stable Source ID for this source
            # -------------------------------------------------

            if source_key not in source_id_map:

                source_id_map[source_key] = (
                    f"S{next_source_number}"
                )

                next_source_number += 1

            source_id = source_id_map[source_key]

            # -------------------------------------------------
            # Store Source ID in document metadata.
            #
            # RAGService can later use exactly the same
            # identifier while building final sources.
            # -------------------------------------------------

            document.metadata[
                "source_id"
            ] = source_id

            context_part = f"""
--- CONTEXT {index} ---

Source ID:
{source_id}

Policy Type:
{policy_type}

Document:
{document_name}

Page:
{page_number}

Relevance Score:
{rerank_score}

Source:
{source}

Content:
{content}
"""

            context_parts.append(
                context_part.strip()
            )

        if not context_parts:

            logger.warning(
                "No valid context sections were created."
            )

            return ""

        final_context = "\n\n".join(
            context_parts
        )

        logger.info(
            f"Context built successfully from "
            f"{len(context_parts)} documents."
        )

        logger.info(
            f"Unique citation sources in context: "
            f"{len(source_id_map)}"
        )

        logger.info(
            f"Citation mapping: "
            f"{source_id_map}"
        )

        return final_context