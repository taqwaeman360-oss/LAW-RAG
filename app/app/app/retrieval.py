import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENAI_API_KEY
from vector_store import build_or_load_vector_store

LEGAL_PROMPT = """You are an AI Legal Information Assistant for Pakistan.
Answer the user's question strictly using the provided context from Pakistani laws.

Guidelines:
1. Grounding: Answer ONLY using the context. If information is missing, state: "The provided legal documents do not contain enough information to answer this question."
2. Sources: Cite document names and sections/pages where possible.
3. Disclaimer: Always append this legal disclaimer at the very end:
   "\n\n--- \n*Disclaimer: This information is provided for educational purposes only and does not constitute formal legal advice.*"

Context:
{context}

Question:
{question}

Answer:"""

class LegalRAGPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
        self.vector_store = build_or_load_vector_store()

    def get_retriever(self):
        if not self.vector_store:
            self.vector_store = build_or_load_vector_store()
        if not self.vector_store:
            raise RuntimeError("Vector database not found. Please run document ingestion first.")
        return self.vector_store.as_retriever(search_kwargs={"k": 4})

    def query(self, question: str):
        retriever = self.get_retriever()
        docs = retriever.invoke(question)

        # Format context string
        context_parts = []
        sources = []
        for doc in docs:
            src_name = os.path.basename(doc.metadata.get("source", "Legal Document"))
            page_num = doc.metadata.get("page", None)
            page_str = f" (Page {page_num + 1})" if page_num is not None else ""
            context_parts.append(f"[{src_name}{page_str}]:\n{doc.page_content}")
            
            sources.append({
                "document": src_name,
                "page": page_num + 1 if page_num is not None else "N/A",
                "snippet": doc.page_content[:200] + "..."
            })

        formatted_context = "\n\n".join(context_parts)
        prompt = ChatPromptTemplate.from_template(LEGAL_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({"context": formatted_context, "question": question})

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
