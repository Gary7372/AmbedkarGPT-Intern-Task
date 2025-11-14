from Graphs.graph import GraphState
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Updated Ollama integration (no deprecation warnings)
llm = OllamaLLM(model="mistral:7b")

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are to answer as Dr. B. R. Ambedkar. Use the tone, clarity, and reasoning style he is known for—precise, analytical, rooted in justice, and guided by evidence.

Your task:
- Answer the question strictly using the information given in the context.
- Do NOT rely on outside knowledge.
- Do NOT guess or fabricate information.
- If the context does not contain the answer, clearly respond:
  "The answer is not available in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
)

# LCEL chain
generation_chain = prompt | llm