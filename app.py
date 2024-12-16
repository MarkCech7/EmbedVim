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

@app.route('/upload', methods=['POST'])
def upload_file():     
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = save_uploaded_file(file, upload_folder)
            
    try:
        knowledge_base.load_file(filepath)
            
        return jsonify({"message": "File successfully processed"}), 200
            
    except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/', methods=['POST'])
def rag():
    data = request.get_json()
    query = data.get("query")
    # 'true' -> True, 'false' -> False, 'None' -> False
    hyde_param = request.args.get('hyde') #enabled
    hyde_enabled = hyde_param == 'enabled' 

    if not query:
        return jsonify({'error': "Empty query!"}), 400

    response = knowledge_base.query(query, hyde_enabled)
    print(response)
    return jsonify({'response': str(response)})

@app.route('/reset', methods=['POST'])
def reset():
    knowledge_base.reset()
    return jsonify({"message": "KnowledgeBase cleared sucessfully."}), 200

@app.route('/init', methods=['POST'])
def init():
    knowledge_base.load_from_folder(doc_directory)
    return jsonify({"message": "KnowledgeBase loaded sucessfully."}), 200

if __name__ == '__main__':
    app.run(debug=True)
