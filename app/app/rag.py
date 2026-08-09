import os
from typing import Dict, List, Any

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import CHROMA_PATH, DATA_DIR, OPENAI_API_KEY

LEGAL_SYSTEM_PROMPT = """You are a specialized Pakistan Legal Information Assistant. 
Your primary goal is to answer legal questions strictly using the provided legal context (such as the Constitution of Pakistan, Pakistan Penal Code, Contract Act, PECA, etc.).

Strict Operational Guidelines:
1. Grounding: Answer ONLY using the context provided below. If the information is not explicitly present in the retrieved context, state: "The provided legal documents do not contain enough information to answer this question."
2. Citations: Always mention the source document name and section/page numbers if present in the context.
3. Educational Disclaimer: Always append the following exact legal disclaimer at the very end of your response:
   
   "--- \n*Disclaimer: This information is provided strictly for educational and informational purposes and does not constitute professional legal advice. Consult a qualified legal practitioner in Pakistan for formal legal counsel.*"

Context:
{context}

Question:
{question}

Answer:"""


class PakistanLawRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
        self.vector_store = None
        self.retriever = None

        # Load existing collection if it exists, otherwise initialize empty
        if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
            self.vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embeddings
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

    def ingest_documents(self):
        """Loads PDFs and TXT files from data directory, chunks them, and builds Chroma DB."""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            raise ValueError(f"Data directory '{DATA_DIR}' was missing and has been created. Please add legal documents (.pdf/.txt) to proceed.")

        # Load PDF and TXT documents
        pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
        
        documents = pdf_loader.load() + txt_loader.load()

        if not documents:
            raise ValueError(f"No legal documents found in '{DATA_DIR}'. Please place PDF or TXT files inside.")

        # Legal-optimized text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\nSection", "\n\nArticle", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # Build Chroma Store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_PATH
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        return len(chunks)

    def format_docs(self, docs):
        """Formats docs for prompt injection."""
        formatted = []
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
            page = doc.metadata.get("page", None)
            loc = f" (Page {page + 1})" if page is not None else ""
            formatted.append(f"[Source {i+1}: {source}{loc}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def query(self, question: str) -> Dict[str, Any]:
        """Queries the vector store and generates a grounded response with source metadata."""
        if not self.retriever:
            raise RuntimeError("Vector database is not initialized. Run document ingestion first.")

        # Retrieve relevant context documents
        retrieved_docs = self.retriever.invoke(question)

        # Format sources bonus payload
        sources = []
        for doc in retrieved_docs:
            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
            page_num = doc.metadata.get("page", None)
            sources.append({
                "document": source_file,
                "page": page_num + 1 if page_num is not None else "N/A",
                "content_snippet": doc.page_content[:200] + "..."
            })

        # Run LCEL Chain
        prompt = ChatPromptTemplate.from_template(LEGAL_SYSTEM_PROMPT)
        context_str = self.format_docs(retrieved_docs)
        
        chain = prompt | self.llm | StrOutputParser()
        response_text = chain.invoke({"context": context_str, "question": question})

        return {
            "question": question,
            "answer": response_text,
            "sources": sources
        }
