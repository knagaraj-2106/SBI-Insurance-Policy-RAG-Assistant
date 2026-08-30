from retrieval.semantic_retriever import SemanticRetriever
from reranking.llm_reranker import LLMReranker
from context.context_builder import ContextBuilder


def main():

    print("=" * 70)

    print(
        "SBI INSURANCE POLICY - "
        "CONTEXT BUILDER TEST"
    )

    print("=" * 70)

    query = (
        "What medical expenses are covered "
        "under the Travel Insurance Policy?"
    )

    policy_type = "Travel Insurance Policy"

    # --------------------------------------------------
    # STEP 1: Retrieve candidates
    # --------------------------------------------------

    retriever = SemanticRetriever()

    documents = retriever.retrieve(
        query=query,
        top_k=10,
        policy_type=policy_type
    )

    print(
        f"\nCandidates retrieved: "
        f"{len(documents)}"
    )

    # --------------------------------------------------
    # STEP 2: Rerank candidates
    # --------------------------------------------------

    reranker = LLMReranker()

    ranked_documents = reranker.rerank(
        query=query,
        documents=documents,
        top_k=5
    )

    print(
        f"Documents after reranking: "
        f"{len(ranked_documents)}"
    )

    # --------------------------------------------------
    # STEP 3: Build context
    # --------------------------------------------------

    context_builder = ContextBuilder()

    context = context_builder.build_context(
        ranked_documents
    )

    # --------------------------------------------------
    # STEP 4: Display context
    # --------------------------------------------------

    print("\n" + "=" * 70)

    print("FINAL RAG CONTEXT")

    print("=" * 70)

    print(context)

    print("\n" + "=" * 70)


if __name__ == "__main__":

    main()