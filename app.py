"""
SBI Insurance Policy RAG Assistant
==================================

Streamlit UI for the SBI Insurance Policy
Retrieval-Augmented Generation system.

The UI communicates with the existing RAGService
and does not duplicate RAG pipeline logic.

RAG Pipeline:

    User Query
        ↓
    Query Classifier
        ↓
    Query Rewriter
        ↓
    Semantic Retrieval
        ↓
    LLM Reranking
        ↓
    Relevance Guard
        ↓
    Context Builder
        ↓
    LLM Generation
        ↓
    Grounding Guard
        ↓
    Groundedness Guard
        ↓
    Citation Guard
        ↓
    Final Answer
"""

import json
import time
from typing import Any, List


import streamlit as st


from rag.rag_service import RAGService
from conversation.conversation_manager import ConversationManager


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SBI Insurance Policy RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

POLICIES = [
    "Travel Insurance Policy",
    "Critical Illness Insurance Policy"
]


DEFAULT_POLICY = "Travel Insurance Policy"


WELCOME_MESSAGE = """
👋 **Welcome to the SBI Insurance Policy RAG Assistant.**

I can answer questions using the available SBI Insurance
Policy documents.

### You can ask about:

- Medical expenses
- Emergency medical evacuation
- Dental treatment
- Repatriation of mortal remains
- Policy coverage
- Benefits
- Exclusions
- Policy conditions
- Other information available in the policy documents

Select a policy from the sidebar and ask your question below.
"""


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state() -> None:
    """
    Initialize Streamlit session-state variables.
    """

    if "rag_service" not in st.session_state:

        st.session_state.rag_service = RAGService()

    if "conversation_manager" not in st.session_state:

        st.session_state.conversation_manager = (
            ConversationManager(
                max_messages=10
            )
        )

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "selected_policy" not in st.session_state:

        st.session_state.selected_policy = (
            DEFAULT_POLICY
        )

    if "last_response" not in st.session_state:

        st.session_state.last_response = None

    if "show_pipeline" not in st.session_state:

        st.session_state.show_pipeline = False

    if "last_latency" not in st.session_state:

        st.session_state.last_latency = None


# Initialize application state.

initialize_session_state()


# ============================================================
# GENERIC RESULT ACCESS
# ============================================================

def get_result_value(
    result: Any,
    key: str,
    default: Any = None
) -> Any:
    """
    Safely retrieve values from:

    - dictionaries
    - Pydantic/model objects
    - generic Python objects
    """

    if result is None:

        return default

    if isinstance(result, dict):

        return result.get(
            key,
            default
        )

    return getattr(
        result,
        key,
        default
    )


# ============================================================
# SOURCE VALUE HELPER
# ============================================================

def get_source_value(
    source: Any,
    field_name: str,
    default: Any = "N/A"
) -> Any:
    """
    Safely retrieve fields from Source objects
    or dictionaries.
    """

    if source is None:

        return default

    if isinstance(source, dict):

        value = source.get(
            field_name,
            default
        )

    else:

        value = getattr(
            source,
            field_name,
            default
        )

    if value is None:

        return default

    return value


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    sources: List[Any]
) -> None:
    """
    Display authoritative supporting sources.
    """

    if not sources:

        st.info(
            "No supporting sources were returned."
        )

        return

    with st.expander(
        f"📚 View Sources ({len(sources)})",
        expanded=False
    ):

        for index, source in enumerate(
            sources,
            start=1
        ):

            source_id = get_source_value(
                source,
                "source_id",
                f"S{index}"
            )

            policy_type = get_source_value(
                source,
                "policy_type"
            )

            document_name = get_source_value(
                source,
                "document_name"
            )

            page_number = get_source_value(
                source,
                "page_number"
            )

            rerank_score = get_source_value(
                source,
                "rerank_score"
            )

            source_path = get_source_value(
                source,
                "source"
            )

            st.markdown(
                f"### 📄 [{source_id}]"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Policy:** {policy_type}"
                )

                st.write(
                    f"**Document:** {document_name}"
                )

                st.write(
                    f"**Page:** {page_number}"
                )

            with col2:

                st.write(
                    f"**Rerank Score:** "
                    f"{rerank_score}"
                )

                if source_path != "N/A":

                    st.write(
                        f"**Source:** {source_path}"
                    )

            if index < len(sources):

                st.divider()


# ============================================================
# DISPLAY RAG PIPELINE DETAILS
# ============================================================

def display_pipeline_details(
    response: Any,
    latency: float | None = None
) -> None:
    """
    Display detailed RAG pipeline diagnostics.

    Useful during local development and project
    demonstration.
    """

    if response is None:

        return

    with st.expander(
        "🔎 RAG Pipeline Details",
        expanded=False
    ):

        # ----------------------------------------------------
        # Query Processing
        # ----------------------------------------------------

        st.markdown(
            "### 🧠 Query Processing"
        )

        query_category = get_result_value(
            response,
            "query_category",
            get_result_value(
                response,
                "category",
                "N/A"
            )
        )

        classifier_confidence = get_result_value(
            response,
            "classifier_confidence",
            "N/A"
        )

        rewritten_query = get_result_value(
            response,
            "rewritten_query",
            "N/A"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Category:** {query_category}"
            )

            st.write(
                f"**Classifier Confidence:** "
                f"{classifier_confidence}"
            )

        with col2:

            st.write(
                "**Rewritten Query:**"
            )

            st.write(
                rewritten_query
            )

        st.divider()

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        st.markdown(
            "### 🔍 Retrieval"
        )

        retrieved_count = get_result_value(
            response,
            "retrieved_document_count",
            0
        )

        reranked_count = get_result_value(
            response,
            "reranked_document_count",
            0
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Retrieved Documents",
                retrieved_count
            )

        with col2:

            st.metric(
                "Reranked Documents",
                reranked_count
            )

        st.divider()

        # ----------------------------------------------------
        # Guardrails
        # ----------------------------------------------------

        st.markdown(
            "### 🛡️ Guardrail Validation"
        )

        relevance_passed = get_result_value(
            response,
            "relevance_passed",
            False
        )

        grounding_passed = get_result_value(
            response,
            "grounding_passed",
            False
        )

        groundedness_passed = get_result_value(
            response,
            "groundedness_passed",
            False
        )

        citation_passed = get_result_value(
            response,
            "citation_passed",
            False
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Relevance",
                "PASS"
                if relevance_passed
                else "FAIL"
            )

        with col2:

            st.metric(
                "Grounding",
                "PASS"
                if grounding_passed
                else "FAIL"
            )

        with col3:

            st.metric(
                "Groundedness",
                "PASS"
                if groundedness_passed
                else "FAIL"
            )

        with col4:

            st.metric(
                "Citation",
                "PASS"
                if citation_passed
                else "FAIL"
            )

        st.divider()

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        st.markdown(
            "### ⚡ Performance"
        )

        retry_count = get_result_value(
            response,
            "retry_count",
            "N/A"
        )

        col1, col2 = st.columns(2)

        with col1:

            if latency is not None:

                st.metric(
                    "Response Time",
                    f"{latency:.2f} sec"
                )

            else:

                st.metric(
                    "Response Time",
                    "N/A"
                )

        with col2:

            st.metric(
                "Retries",
                retry_count
            )


# ============================================================
# CLEAR CONVERSATION
# ============================================================

def clear_conversation() -> None:
    """
    Clear the Streamlit UI conversation and
    conversation manager history.
    """

    st.session_state.messages = []

    st.session_state.last_response = None

    st.session_state.last_latency = None

    try:

        st.session_state.conversation_manager.clear()

    except Exception as exc:

        st.warning(
            f"Conversation history could not be cleared: {exc}"
        )


# ============================================================
# BUILD CONVERSATION DOWNLOAD
# ============================================================

def build_download_content() -> str:
    """
    Convert the current conversation into JSON.
    """

    conversation = []

    for message in st.session_state.messages:

        role = message.get(
            "role"
        )

        content = message.get(
            "content",
            ""
        )

        sources = []

        if role == "assistant":

            for source in message.get(
                "sources",
                []
            ):

                sources.append(
                    {
                        "source_id": get_source_value(
                            source,
                            "source_id"
                        ),
                        "policy_type": get_source_value(
                            source,
                            "policy_type"
                        ),
                        "document_name": get_source_value(
                            source,
                            "document_name"
                        ),
                        "page_number": get_source_value(
                            source,
                            "page_number"
                        ),
                        "rerank_score": get_source_value(
                            source,
                            "rerank_score"
                        ),
                    }
                )

        conversation.append(
            {
                "role": role,
                "content": content,
                "sources": sources
            }
        )

    return json.dumps(
        conversation,
        indent=4,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    """
    Render the application sidebar.
    """

    with st.sidebar:

        st.header(
            "⚙️ Settings"
        )

        # ----------------------------------------------------
        # Policy Selection
        # ----------------------------------------------------

        st.subheader(
            "📑 Insurance Policy"
        )

        selected_policy = st.selectbox(
            "Select Insurance Policy",
            POLICIES,
            index=POLICIES.index(
                st.session_state.selected_policy
            )
        )

        st.session_state.selected_policy = (
            selected_policy
        )

        st.divider()

        # ----------------------------------------------------
        # Developer Mode
        # ----------------------------------------------------

        st.subheader(
            "🔧 Developer Mode"
        )

        show_pipeline = st.checkbox(
            "Show RAG Pipeline Details",
            value=(
                st.session_state.show_pipeline
            )
        )

        st.session_state.show_pipeline = (
            show_pipeline
        )

        st.caption(
            "Displays classification, retrieval, "
            "guardrail and performance information."
        )

        st.divider()

        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------

        st.subheader(
            "💬 Conversation"
        )

        message_count = len(
            st.session_state.messages
        )

        st.write(
            f"Messages: **{message_count}**"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🗑️ Clear",
                use_container_width=True
            ):

                clear_conversation()

                st.rerun()

        with col2:

            if st.button(
                "🔄 New Chat",
                use_container_width=True
            ):

                clear_conversation()

                st.rerun()

        st.divider()

        # ----------------------------------------------------
        # Conversation Export
        # ----------------------------------------------------

        st.subheader(
            "💾 Conversation Export"
        )

        if st.session_state.messages:

            download_content = (
                build_download_content()
            )

            st.download_button(
                label="⬇️ Download Chat",
                data=download_content,
                file_name=(
                    "sbi_insurance_conversation.json"
                ),
                mime="application/json",
                use_container_width=True
            )

        else:

            st.caption(
                "No conversation available to download."
            )

        st.divider()

        # ----------------------------------------------------
        # Sample Questions
        # ----------------------------------------------------

        st.subheader(
            "💡 Sample Questions"
        )

        st.markdown(
            """
**Travel Insurance**

- What medical expenses are covered?
- Is emergency evacuation covered?
- Is dental treatment covered?
- What is repatriation of mortal remains?

**Critical Illness**

- What illnesses are covered?
- What are the policy benefits?
- What are the exclusions?
"""
        )

        st.divider()

        # ----------------------------------------------------
        # RAG Pipeline
        # ----------------------------------------------------

        st.subheader(
            "🔎 RAG Pipeline"
        )

        st.markdown(
            """
**User Query**
↓
**Query Classifier**
↓
**Query Rewriter**
↓
**Semantic Retrieval**
↓
**LLM Reranking**
↓
**Relevance Guard**
↓
**Context Builder**
↓
**LLM Generation**
↓
**Grounding Guard**
↓
**Groundedness Guard**
↓
**Citation Guard**
↓
**Final Answer**
"""
        )

        st.divider()

        # ----------------------------------------------------
        # Project Information
        # ----------------------------------------------------

        st.subheader(
            "ℹ️ Project"
        )

        st.caption(
            "SBI Insurance Policy RAG Assistant"
        )

        st.caption(
            "Python • LangChain • OpenAI • "
            "ChromaDB • Streamlit"
        )

        st.caption(
            "Production-style RAG pipeline with "
            "retrieval, reranking and guardrails."
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 SBI Insurance Policy RAG Assistant"
)

st.caption(
    "AI-powered question answering over SBI Insurance "
    "Policy documents using Retrieval-Augmented Generation."
)


# ============================================================
# SIDEBAR
# ============================================================

render_sidebar()


# ============================================================
# SELECTED POLICY INDICATOR
# ============================================================

st.info(
    f"📑 **Selected Policy:** "
    f"{st.session_state.selected_policy}"
)


# ============================================================
# WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            WELCOME_MESSAGE
        )


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role"
    )

    content = message.get(
        "content",
        ""
    )

    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )

        if role == "assistant":

            sources = message.get(
                "sources",
                []
            )

            display_sources(
                sources
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask a question about the insurance policy..."
)


# ============================================================
# PROCESS USER QUERY
# ============================================================

if user_query:

    user_query = user_query.strip()

    # --------------------------------------------------------
    # Validate query
    # --------------------------------------------------------

    if not user_query:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # Save user message to UI history
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_query
        )

    # --------------------------------------------------------
    # Retrieve conversation history
    # --------------------------------------------------------

    try:

        conversation_history = (
            st.session_state
            .conversation_manager
            .get_history()
        )

    except Exception:

        conversation_history = ""

        st.warning(
            "Conversation history could not be loaded. "
            "Continuing with the current query."
        )

    # --------------------------------------------------------
    # Generate RAG response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        response = None

        latency = None

        try:

            start_time = time.perf_counter()

            with st.spinner(
                "🔍 Searching insurance policy documents..."
            ):

                response = (
                    st.session_state
                    .rag_service
                    .query(
                        user_query=user_query,
                        policy_type=(
                            st.session_state
                            .selected_policy
                        ),
                        conversation_history=(
                            conversation_history
                        )
                    )
                )

            latency = (
                time.perf_counter()
                - start_time
            )

            st.session_state.last_latency = (
                latency
            )

            # =================================================
            # EXTRACT RESPONSE
            # =================================================

            answer = get_result_value(
                response,
                "answer",
                ""
            )

            sources = get_result_value(
                response,
                "sources",
                []
            )

            query_category = get_result_value(
                response,
                "query_category",
                get_result_value(
                    response,
                    "category",
                    None
                )
            )

            # =================================================
            # DISPLAY ANSWER
            # =================================================

            if answer:

                st.markdown(
                    answer
                )

            else:

                st.warning(
                    "The RAG pipeline did not return "
                    "a final answer."
                )

            # =================================================
            # SAVE CONVERSATION
            # =================================================

            try:

                st.session_state.conversation_manager.add_user_message(
                    user_query
                )

                if answer:

                    st.session_state.conversation_manager.add_assistant_message(
                        answer
                    )

            except Exception as exc:

                st.warning(
                    "The answer was generated, but the "
                    f"conversation history could not be saved: {exc}"
                )

            # =================================================
            # SAVE ASSISTANT MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

            # =================================================
            # SAVE LAST RESPONSE
            # =================================================

            st.session_state.last_response = (
                response
            )

            # =================================================
            # DISPLAY SOURCES
            # =================================================

            if sources:

                display_sources(
                    sources
                )

            # =================================================
            # DISPLAY PIPELINE DETAILS
            # =================================================

            if st.session_state.show_pipeline:

                display_pipeline_details(
                    response=response,
                    latency=latency
                )

            # =================================================
            # OUT-OF-DOMAIN INFORMATION
            # =================================================

            if (
                query_category == "OUT_OF_DOMAIN"
                and answer
            ):

                st.info(
                    "ℹ️ This query was identified as "
                    "outside the supported insurance "
                    "policy domain."
                )

        except Exception as exc:

            # ------------------------------------------------
            # User-friendly error
            # ------------------------------------------------

            st.error(
                "❌ An error occurred while processing "
                "your question."
            )

            st.info(
                "Please try again. If the problem continues, "
                "verify that the RAG service and vector store "
                "are available."
            )

            # ------------------------------------------------
            # Developer error details
            # ------------------------------------------------

            if st.session_state.show_pipeline:

                with st.expander(
                    "🐞 Technical Error Details",
                    expanded=False
                ):

                    st.exception(
                        exc
                    )

            # ------------------------------------------------
            # Remove failed user message
            # ------------------------------------------------

            if st.session_state.messages:

                last_message = (
                    st.session_state.messages[-1]
                )

                if (
                    last_message.get("role")
                    == "user"
                ):

                    st.session_state.messages.pop()

