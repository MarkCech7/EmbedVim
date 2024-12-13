import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

template = (
        "You are a leading expert in advanced AI and space sciences, with extensive knowledge in astronomy, space agencies, telescopes, planetary science, and cosmic exploration."
        "Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
        "Here is context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry with detailed references:\n\n"
        "Question: {query_str}\n\n"
        "Answer succinctly and DO NOT mention context, or phrases such as 'According to my knowledge...'."
        #"Use Tree-of-thought prompting technique."
        "DO NOT mention REFERENCES in any response!"
        "DO NOT use prior knowledge!"
        "DO NOT answer questions from non-related topics!"
    )

class KnowledgeBase:
    def __init__(self, db_directory, model_name):
        embeddings = OllamaEmbedding(model_name=model_name)
        llm = Ollama(model=model_name, request_timeout=120.0)
        db = chromadb.PersistentClient(path=db_directory)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index  = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embeddings,  
            show_progress=True
        )

        qa_template = PromptTemplate(template)
        query_engine = self.index.as_query_engine(
            llm=llm,
            text_qa_template=qa_template,
            similarity_top_k=3
        )

        hyde = HyDEQueryTransform(llm=llm, include_original=True)
        self.hyde_query_engine = TransformQueryEngine(query_engine, hyde)

    def load_from_folder(self, doc_directory):
        reader = SimpleDirectoryReader(input_dir=doc_directory, recursive=True)
        content = reader.load_data()

        for item in content:
            self.index.insert(document=item)
    
    def load_file(self, filepath):
        reader = SimpleDirectoryReader(input_files=[filepath])
        content = reader.load_data()
        self.index.insert(document=content[0])

    def query(self, query):
        response = self.hyde_query_engine.query(query)
        return response