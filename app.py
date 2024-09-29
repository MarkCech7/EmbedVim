#from langchain_community.document_loaders import TextLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.llms import Ollama
import os
from doc_loader import LoadDocuments, RemoveDuplicates
from llm import GenerateResponse

promptTemplate = """Context information is below.
---------------------
[context_str]
---------------------
Given the context information and not prior knowledge, answer the query. Context may contain duplicates.
Query: [query_str]
Answer:"""

model_name = "llama3.1"
db_directory = "./chroma"
doc_directory = "./documents"

embeddings = OllamaEmbeddings(model=model_name)
vectordb = Chroma(persist_directory=db_directory, embedding_function=embeddings)
embeddings_count = vectordb._collection.count()

if embeddings_count == 0:
    print("Vector database does not exist. Creating a new one.")
    texts = LoadDocuments(doc_directory)
    vectordb.add_texts(texts=texts, embeddings=embeddings)

else:
    print("Vector database exists. Loading the existing database.")

query = "Where is James Webb Telescope now?"
retrieved_docs = vectordb.similarity_search(query, k=3)
unique_docs = retrieved_docs

response = GenerateResponse(unique_docs, query, model_name, promptTemplate)

print(response)
