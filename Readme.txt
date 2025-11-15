# RAG Pipeline Documentation

This README explains how your Retrieval-Augmented Generation (RAG) pipeline works using LangGraph, LangChain, ChromaDB, and Ollama (mistral:7b).

---

##  Overview

This project implements a **3‑step RAG workflow** using a LangGraph state machine:

1. **Extraction + Storage Node**

   * Loads `.txt` file
   * Splits text into chunks
   * Creates vector embeddings using MiniLM-L6-v2
   * Stores them in ChromaDB

2. **Retrieval Node**

   * Performs vector similarity search based on the user's query
   * Returns the most relevant chunks

3. **Generation Node**

   * Uses retrieved chunks as context
   * Passes them to an Ollama‑powered LCEL chain (PromptTemplate → Mistral‑7B)
   * Produces the final answer

---

##  Project Structure

```
project/
│
├── Graphs/
│   ├── graph.py                     # GraphState + build_graph()
│
├── nodes/
│   ├── extraction_and_storage_node.py
│   ├── retrieval_node.py
│   ├── generation_node.py
│
├── chains/
│   ├── generation_node_chain.py
│
├── main.py                           # main RAG runner
└── speech.txt                        # sample input file
```

---

##  Graph State

```python
class GraphState(BaseModel):
    file_path: Optional[str]
    raw_text: Optional[str]
    chunks: Optional[List[str]]
    vector_db: Any
    question: Optional[str]
    retrieved_docs: Optional[Any]
    answer: Optional[str]
```

This state moves through all nodes and gets progressively filled.

---

##  How Build Graph Works

```python
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
```

---

## Extraction & Storage Node

* Loads text using **TextLoader**
* Splits into chunks (400 chars, 100 overlap)
* Creates embeddings using **MiniLM-L6-v2**
* Stores in **ChromaDB**

---

##  Retrieval Node

Performs similarity search:

```python
docs = vectordb.similarity_search(query, k=2)
```

---

##  Generation Node

Uses an LCEL chain:

```
prompt → llm (mistral:7b) → output
```

Enforces strict rules:

* No outside knowledge
* No guessing
* Ambedkar tone

---

## ▶ How to Run the Pipeline

Execute --> main.py
You can change the query or question in main.py

Then prints:

* Stored chunks
* Retrieved chunks
* Final answer

---

##  Requirements

* Python 3.10+
* Ollama installed
* Model pulled:

```
ollama pull mistral:7b
```

Install dependencies:

```
pip install langgraph langchain langchain-community chromadb sentence-transformers
```

---

##  Notes

* Only `.txt` files are supported
* Chroma memory is in-memory unless persisted
* Retrieval `k=2` can be tuned

---

##  Output Example

Your script prints:

* All text chunks
* Best-matching retrieved chunks
* Final Mistral-generated answer

---

