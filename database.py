from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def CreateAndAddToVectorDB(texts, model_name="llama3.1", persist_directory="./chroma"):
  
    embeddings = OllamaEmbeddings(model=model_name)
    vectordb = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory)
    
    vectordb.add_texts(texts=texts, embeddings=embeddings)
    
    return vectordb