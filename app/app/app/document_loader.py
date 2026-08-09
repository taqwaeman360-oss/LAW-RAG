import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from config import DATA_DIR

def load_documents():
    """Loads all PDF and TXT files from the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        raise ValueError(f"Directory '{DATA_DIR}' created. Please place legal documents inside.")

    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)

    docs = pdf_loader.load() + txt_loader.load()
    return docs
