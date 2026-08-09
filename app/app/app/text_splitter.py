from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """Splits documents into legal-optimized chunks with section boundary awareness."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\nSection", "\n\nArticle", "\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)
