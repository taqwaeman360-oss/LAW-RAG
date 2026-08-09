import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, OPENAI_API_KEY

def get_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def build_or_load_vector_store(chunks=None):
    """Creates a new vector store if chunks provided, else loads existing."""
    embeddings = get_embeddings()
    if chunks:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        return vector_store
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    return None
