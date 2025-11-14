from Graphs.graph import GraphState
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from chains.generation_node_chain import generation_chain



def generation_node(state: GraphState) -> GraphState:
    # Combine retrieved docs
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

    # Run LCEL
    result = generation_chain.invoke({
        "context": context,
        "question": state.question
    })

    # result is a STRING, so assign directly
    state.answer = result

    return state
