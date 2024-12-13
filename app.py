import os
from flask import Flask, request, jsonify
from knowledgebase import KnowledgeBase
from file import save_uploaded_file

model_name = os.getenv("RAG_MODEL_NAME", "llama3.2") 
db_directory = os.getenv("RAG_DB_DIRECTORY", "./chroma")
doc_directory = os.getenv("RAG_DOC_DIRECTORY", "./documents")
upload_folder = os.getenv("UPLOAD_FOLDER", "uploads")

os.makedirs(upload_folder, exist_ok=True)
os.makedirs(doc_directory, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "hello")
app.config['UPLOAD_FOLDER'] = upload_folder

knowledge_base = KnowledgeBase(db_directory=db_directory,model_name=model_name)
knowledge_base.load_from_folder(doc_directory)

@app.route('/upload', methods=['POST'])
def upload_file():     
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = save_uploaded_file(file)
            
    try:
        knowledge_base.load_file(filepath)
            
        return jsonify({"message": "File successfully processed"}), 200
            
    except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/', methods=['POST'])
def rag():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({'error': "Empty query!"}), 400

    response = knowledge_base.query(query)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)
