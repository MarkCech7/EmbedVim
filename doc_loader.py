from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def LoadDocuments(doc_directory, chunk_size=1024, chunk_overlap=512):
    documents = []
    for filename in os.listdir(doc_directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(doc_directory, filename)
            loader = TextLoader(filepath)
            document = loader.load()  # Load the document
            documents.append(document)
    
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", ". ", "? ", "! "])
    splitted_docs = []
    
    for document in documents:
        splitted_docs.extend(text_splitter.split_documents(document))
    
    texts = [doc.page_content for doc in splitted_docs]
    return texts
