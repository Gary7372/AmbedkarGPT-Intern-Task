

from Graphs.graph import build_graph
from Graphs.graph import GraphState




def run_rag_pipeline(file_path: str, question: str):
    """
    Runs the RAG pipeline using your LangGraph state machine.
    """

    graph = build_graph()

    # Initial state for graph
    initial_state = GraphState(
        file_path=file_path,
        question=question
    )

    # Run the graph
    final_state = graph.invoke(initial_state)

    return final_state


if __name__ == "__main__":

    # -------- INPUTS ----------
    file_path = "speech.txt"
    question = "According to the passage, what is the 'real remedy' for ending caste?"

    # -------- RUN RAG ---------
    result = run_rag_pipeline(file_path, question)

    # -------- PRINT RAW CHUNKS --------
    print("\n========== STORED CHUNKS ==========\n")
    for i, chunk in enumerate(result["chunks"], start=1):
        print(f"\n--- Chunk {i} ---\n{chunk}")

    # -------- PRINT RETRIEVED CHUNKS --------
    print("\n========== RETRIEVED CHUNKS ==========\n")
    for i, doc in enumerate(result["retrieved_docs"], start=1):
        print(f"\n--- Retrieved {i} ---\n{doc.page_content}")

    # -------- PRINT FINAL ANSWER --------
    print("\n========== FINAL ANSWER ==========\n")
    print(result["answer"])
