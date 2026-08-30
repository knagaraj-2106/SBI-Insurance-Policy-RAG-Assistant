import re

from typing import List, Dict, Any

from utils.logger import get_logger


logger = get_logger(__name__)


class CitationGuard:

    def __init__(self):

        logger.info(
            "Citation guard initialized."
        )

    # =========================================================
    # EXTRACT CITATION IDS
    # =========================================================

    @staticmethod
    def _extract_citation_ids(
        answer: str
    ) -> List[str]:

        """
        Extract citation IDs from the generated answer.

        Expected citation formats:

            [S1]
            [S2]
            [S1][S2]

        Returns unique normalized citation IDs.
        """

        if not answer or not answer.strip():

            return []

        # -----------------------------------------------------
        # Match citations such as [S1], [S2], [S10]
        # -----------------------------------------------------

        matches = re.findall(
            r"\[(S\d+)\]",
            answer,
            flags=re.IGNORECASE
        )

        normalized_ids = []

        for citation_id in matches:

            normalized_id = (
                citation_id.upper()
            )

            if normalized_id not in normalized_ids:

                normalized_ids.append(
                    normalized_id
                )

        return normalized_ids

    # =========================================================
    # EXTRACT SOURCE ID
    # =========================================================

    @staticmethod
    def _get_source_id(
        source: Any
    ) -> str:

        """
        Safely extract source_id from either:

        1. Pydantic Source model
        2. Dictionary
        3. Compatible object

        This prevents:

            AttributeError:
            'Source' object has no attribute 'get'
        """

        # -----------------------------------------------------
        # Pydantic / object based source
        # -----------------------------------------------------

        if hasattr(
            source,
            "source_id"
        ):

            source_id = getattr(
                source,
                "source_id",
                None
            )

            if source_id:

                return str(
                    source_id
                ).upper()

        # -----------------------------------------------------
        # Dictionary based source
        # -----------------------------------------------------

        if isinstance(
            source,
            dict
        ):

            source_id = source.get(
                "source_id"
            )

            if source_id:

                return str(
                    source_id
                ).upper()

        # -----------------------------------------------------
        # Unsupported source object
        # -----------------------------------------------------

        return ""

    # =========================================================
    # VALIDATE CITATIONS
    # =========================================================

    def validate(
        self,
        answer: str,
        sources: List[Any]
    ) -> bool:

        """
        Validate citations in the generated answer.

        Validation rules:

        1. Answer must not be empty.
        2. Sources must exist.
        3. Answer must contain at least one citation.
        4. Every cited source ID must exist in authoritative
           sources.
        5. Citation IDs are case-insensitive.
        """

        # -----------------------------------------------------
        # Validate answer
        # -----------------------------------------------------

        if not answer or not answer.strip():

            logger.warning(
                "Citation validation failed: "
                "empty answer."
            )

            return False

        # -----------------------------------------------------
        # Validate sources
        # -----------------------------------------------------

        if not sources:

            logger.warning(
                "Citation validation failed: "
                "no sources available."
            )

            return False

        # -----------------------------------------------------
        # Extract citation IDs from answer
        # -----------------------------------------------------

        cited_ids = (
            self._extract_citation_ids(
                answer
            )
        )

        logger.info(
            f"Citations found in answer: "
            f"{cited_ids}"
        )

        # -----------------------------------------------------
        # Require at least one citation
        # -----------------------------------------------------

        if not cited_ids:

            logger.warning(
                "Citation validation failed: "
                "answer contains no citations."
            )

            return False

        # -----------------------------------------------------
        # Build authoritative source ID set
        # -----------------------------------------------------

        valid_source_ids = set()

        for source in sources:

            source_id = (
                self._get_source_id(
                    source
                )
            )

            if source_id:

                valid_source_ids.add(
                    source_id
                )

            else:

                logger.warning(
                    "Source does not contain "
                    "a valid source_id."
                )

        logger.info(
            f"Valid source IDs: "
            f"{sorted(valid_source_ids)}"
        )

        # -----------------------------------------------------
        # Validate every citation
        # -----------------------------------------------------

        invalid_citations = []

        for citation_id in cited_ids:

            if citation_id not in valid_source_ids:

                invalid_citations.append(
                    citation_id
                )

        # -----------------------------------------------------
        # Invalid citation detected
        # -----------------------------------------------------

        if invalid_citations:

            logger.warning(
                "Invalid citation(s) detected: "
                f"{invalid_citations}"
            )

            return False

        # -----------------------------------------------------
        # Validation passed
        # -----------------------------------------------------

        logger.info(
            "Citation validation passed."
        )

        return True