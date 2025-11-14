from pydantic import BaseModel
from typing import Any, List, Optional
from langgraph.graph import StateGraph

# --- STATE --------------------------------------------------------------

class GraphState(BaseModel):
    file_path: Optional[str] = None      # file to ingest
    raw_text: Optional[str] = None       # extracted text
    chunks: Optional[List[str]] = None   # split chunks
    vector_db: Any = None                # FAISS / Chroma DB
    question: Optional[str] = None       # query text
    retrieved_docs: Optional[Any] = None # retrieved chunks
    answer: Optional[str] = None         # LLM response


# --- IMPORT NODES ------------------------------------------------------

from nodes.extraction_and_storage_node import extraction_and_storage_node
from nodes.retrieval_node import retrieval_node
from nodes.generation_node import generation_node


# --- GRAPH -------------------------------------------------------------

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("extract_store", extraction_and_storage_node)
    builder.add_node("retrieve", retrieval_node)
    builder.add_node("generate", generation_node)

    builder.set_entry_point("extract_store")
    builder.add_edge("extract_store", "retrieve")
    builder.add_edge("retrieve", "generate")

    builder.set_finish_point("generate")

    return builder.compile()