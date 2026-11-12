import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "./data/"
PERSIST_DIRECTORY = "./db/chroma_db"

def build_vector_store():
    print("Loading documents...")
    loader = DirectoryLoader(path=DATA_PATH, glob='*.txt', loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        raise FileNotFoundError(f"No text files found in {DATA_PATH}. Add your .txt files there.")

    print(f"Loaded {len(documents)} document(s). Splitting chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    print("Generating embeddings and saving to Chroma DB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Vector database successfully built and stored locally!")

if __name__ == "__main__":
    build_vector_store()
