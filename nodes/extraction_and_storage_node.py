from Graphs.graph import GraphState   # for type hinting only
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------- TEXT EXTRACTION (TXT ONLY using TextLoader) ----------

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext != ".txt":
        raise ValueError(f"Unsupported format: {ext}. Only .txt files are allowed.")

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # TextLoader returns [Document], so extract the content
    full_text = "\n".join([doc.page_content for doc in docs])
    return full_text


# --------- TEXT CHUNKING ----------

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# --------- VECTOR STORAGE USING CHROMA ----------

def store_vectors(chunks):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedder,
        collection_name="rag_collection"
    )
    return vectordb


# --------- MAIN NODE ----------

def extraction_and_storage_node(state: GraphState) -> GraphState:
    # Extract text (.txt only)
    text = extract_text(state.file_path)
    state.raw_text = text

    # Chunk the text
    chunks = chunk_text(text)
    state.chunks = chunks

    # Store vectors
    vectordb = store_vectors(chunks)
    state.vector_db = vectordb

    return state
