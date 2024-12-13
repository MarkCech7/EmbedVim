import os
from werkzeug.utils import secure_filename
from flask import request, jsonify

def allowed_file(filename):
    allowed_extensions = {'txt', 'pdf', 'md', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, app):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None