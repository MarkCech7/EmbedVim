import os
from pathlib import Path
from flask import Flask, request, jsonify
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
import chromadb
from werkzeug.utils import secure_filename

model_name = os.getenv("RAG_MODEL_NAME", "llama3.1") 
db_directory = os.getenv("RAG_DB_DIRECTORY", "./chroma")
doc_directory = os.getenv("RAG_DOC_DIRECTORY", "./documents")
upload_folder = os.getenv("UPLOAD_FOLDER", "./uploads")
allowed_extensions = {'txt', 'pdf', 'md', 'doc', 'docx'}

os.makedirs(upload_folder, exist_ok=True)
os.makedirs(doc_directory, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

llm = Ollama(model=model_name, request_timeout=120.0)
embeddings = OllamaEmbedding(model_name=model_name)

db = chromadb.PersistentClient(path=db_directory)
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def load_initial_documents():
    reader = SimpleDirectoryReader(input_dir=doc_directory, recursive=True)
    return reader.load_data()

def initialize_index():
    embeddings_count = vector_store._collection.count()
    if embeddings_count == 0:
        print("Vector database does not exist. Creating a new one.")
        documents = load_initial_documents()
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embeddings,  
            show_progress=True
        )
    else:
        print("Vector database exists. Loading the existing database.")
        return VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embeddings,  
            show_progress=True
        )

index = initialize_index()

@app.route('/upload', methods=['POST'])
def upload_documents():
    if 'documents' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    files = request.files.getlist('documents')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    processed_files = []
    new_documents = []

    for file in files:
        if file.filename == '':
            continue

        try:
            filepath = save_uploaded_file(file)
            if filepath is None:
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            doc = Document(text=content, metadata={"filename": file.filename})
            new_documents.append(doc)
            processed_files.append(file.filename)

            os.remove(filepath)

        except Exception as e:
            return jsonify({'error': f'Error processing file {file.filename}: {str(e)}'}), 500

    if not new_documents:
        return jsonify({'error': 'No valid documents to process'}), 400

    try:
        for doc in new_documents:
            index.insert(document=doc)

        return jsonify({
            'message': f'Successfully processed {len(processed_files)} documents',
            'processed_files': processed_files
        })

    except Exception as e:
        return jsonify({'error': f'Error updating index: {str(e)}'}), 500

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
    )

qa_template = PromptTemplate(template)
query_engine = index.as_query_engine(
    llm=llm,
    text_qa_template=qa_template,
    similarity_top_k=3
)
hyde = HyDEQueryTransform(llm=llm, include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

@app.route('/', methods=['POST'])
def rag():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({'error': "Empty query!"}), 400

    response = hyde_query_engine.query(query)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)

    #test