import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from rag import generar_respuesta, load_documents, vectorstore_n_retriever, load_retriever
from rag import clean_subsystem

app = Flask(__name__)
CORS(app)
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

@app.route('/doit', methods=['POST'])
def do_it():
    # Recibir los datos enviados en la solicitud POST
    data = request.json
    
    # Imprimir los datos recibidos por consola
    print("Received data:")
    print(f"Temperature: {data.get('temperature')}")
    print(f"Role: {data.get('role')}")
    print(f"Question: {data.get('question')}")

    # Checker para confirmar la recepción de los parámetros
    print("Checker: Parámetros recibidos correctamente")
    
    # Reemplazar los parámetros y llamar a la función generar_respuesta
    role = data.get('role')
    question = data.get('question')
    temperature = data.get('temperature')
    
    # Verificar si existe un archivo .sqlite3 en /vectordb y un archivo .json en /retriever
    sqlite3_exists = any(f.endswith('.sqlite3') for f in os.listdir('vectordb'))
    json_exists = any(f.endswith('.json') for f in os.listdir('retriever'))

    if sqlite3_exists and json_exists:
        response = generar_respuesta(role, question, temperature)
        print(response)
        # Por ahora, devolver la respuesta de la función generar_respuesta
        return jsonify({"status": "success", "response": response}), 200
    else:
        vectorstore_n_retriever(model_name='all-MiniLM-L6-v2', docs=load_documents(), retriever_save_path='./retriever/retriever_config.json')
        response = generar_respuesta(role, question, temperature)
        print(response)
        # Por ahora, devolver la respuesta de la función generar_respuesta
        return jsonify({"status": "success", "response": response}), 200

@app.route('/clean', methods=['POST'])
def clean_system():
    try:
        clean_subsystem()  # Llamar a la función para limpiar el subsistema
        print("System cleaned successfully")
        return jsonify({"status": "success", "message": "System cleaned successfully"}), 200
    except Exception as e:
        print(f"Failed to clean the system: {str(e)}")
        return jsonify({"status": "error", "message": f"Failed to clean the system: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)