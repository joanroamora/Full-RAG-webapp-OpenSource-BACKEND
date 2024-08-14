import os
from flask import Flask, request

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'

@app.route('/upload', methods=['POST'])
def upload_files():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Crear un directorio único para cada carga
    upload_dir = os.path.join(UPLOAD_FOLDER, str(len(os.listdir(UPLOAD_FOLDER)) + 1))
    os.makedirs(upload_dir)

    for filename, file in request.files.items():
        file.save(os.path.join(upload_dir, filename))

    return {'status': 'success'}, 200

if __name__ == "__main__":
    app.run(port=5001, debug=True)
