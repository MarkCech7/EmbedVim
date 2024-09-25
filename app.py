from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os
from doc_loader import LoadDocuments
from database import CreateAndAddToVectorDB

promptTemplare = """Context information is below.
---------------------
[context_str]
---------------------
Given the context information and not prior knowledge, answer the query.
Query: [query_str]
Answer:"""

def remove_duplicates(docs):
    seen_texts = set()
    unique_docs = []
    
    for doc in docs:
        # Normalize the text (e.g., lowercasing, stripping spaces) to catch similar duplicates
        normalized_text = doc.page_content.strip().lower()
        
        if normalized_text not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(normalized_text)
    
    return unique_docs

llmModel = "llama3.1"

"""
# Text load, Text splitting, Embedding model initialization, Embedding Generation, Vector database creation
#loader = TextLoader("./text.txt")
doc_directory = "./documents"  # Assuming the directory contains text files
documents = []
for filename in os.listdir(doc_directory):
    if filename.endswith(".txt"):
        filepath = os.path.join(doc_directory, filename)
        loader = TextLoader(filepath)
        document = loader.load()
        #print(document)
        documents.append(document)

#document = loader.load()

llmModel = "llama3.1"

splitted_docs = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
for document in documents:
    splitted_docs.extend(text_splitter.split_documents(document))

texts = [doc.page_content for doc in splitted_docs]

embeddings = OllamaEmbeddings(model=llmModel)

vectordb = Chroma.from_texts(texts, embeddings, persist_directory="./chroma")
vectordb.add_texts(texts=texts, embeddings=embeddings)

query = "What did James Webb telescope study?"
# retrieve context window
retrieved_docs = vectordb.similarity_search(query, 1)

if len(retrieved_docs) >= 1:
    #doc = retrieved_docs[0]
    joined_context = "\n".join([doc.page_content for doc in retrieved_docs])
    print(joined_context)
    #prompt = promptTemplare.replace("[context_str]", doc.page_content).replace("[query_str]", query)
    prompt = promptTemplare.replace("[context_str]", joined_context).replace("[query_str]", query)
    #llm = Ollama(model = llmModel)
    #response = llm.invoke(prompt)
    #print(response)

else:
    print("No context found.")

"""
db_directory = "./chroma"
doc_directory = "./documents"

if not os.path.exists(db_directory):
    print("Vector database does not exist. Creating a new one.")

    texts = LoadDocuments(doc_directory)

    vectordb = CreateAndAddToVectorDB(texts)

else:
    print("Vector database exists. Loading the existing database.")

query = "Where is James Webb Telescope now?"
retrieved_docs = vectordb.similarity_search(query, k=3)
unique_docs = remove_duplicates(retrieved_docs)
combined_context = " ".join([doc.page_content for doc in unique_docs])
#for doc in retrieved_docs:
#print(combined_context)

if len(retrieved_docs) >= 1:
    #doc = retrieved_docs[0]
    #joined_context = "".join([doc.page_content for doc in retrieved_docs])
    #print(joined_context)
    #prompt = promptTemplare.replace("[context_str]", doc.page_content).replace("[query_str]", query)
    prompt = promptTemplare.replace("[context_str]", combined_context).replace("[query_str]", query)
    llm = Ollama(model = llmModel)
    response = llm.invoke(prompt)
    print(response)