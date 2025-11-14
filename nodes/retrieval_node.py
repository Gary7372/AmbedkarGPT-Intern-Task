from Graphs.graph import GraphState
from langchain_core.runnables import RunnableLambda


def retrieval_node(state: GraphState) -> GraphState:
    """
    Node responsible for retrieving the most relevant chunks
    using the LangChain retriever interface.
    """

    if state.vector_db is None:
        raise ValueError("Vector DB is not initialized before retrieval.")

    if not state.question:
        raise ValueError("Question is empty. Retrieval cannot proceed.")

    # Convert the vector store into a retriever
    retriever = state.vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    # Use retriever via LCEL
    retrieved_docs = retriever.invoke(state.question)

    state.retrieved_docs = retrieved_docs
    return state