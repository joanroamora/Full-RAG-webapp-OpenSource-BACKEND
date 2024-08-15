import os
import json
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


#FUNCION DE CARGA DE DOCUMENTOS
def load_documents():
    # Obtener todos los subdirectorios en ./uploads
    directorios = [d for d in os.listdir('./uploads') if os.path.isdir(os.path.join('./uploads', d))]
    
    # Convertir nombres de directorios a enteros y encontrar el máximo
    directorios_numericos = [int(d) for d in directorios if d.isdigit()]
    if not directorios_numericos:
        print("No numeric directories were found in ./uploads")
        return []
    
    max_dir = str(max(directorios_numericos))
    ruta_directorio = os.path.join('./uploads', max_dir)
    
    # Inicializar una lista para acumular todos los documentos
    docs = []
    
    # Cargar todos los archivos PDF en el directorio de número mayor
    for archivo in os.listdir(ruta_directorio):
        if archivo.endswith('.pdf'):
            ruta_pdf = os.path.join(ruta_directorio, archivo)
            loader = PyPDFLoader(ruta_pdf)
            docs1 = loader.load()
            docs.extend(docs1)  # Agregar documentos a la lista
    
    print("Total docs inside " + str(len(docs)))
    return docs


#FUNCION DE CREACION DE VECTOR STORE EN LOCAL Y CREACION DEL MOLDE DEL RETRIEVER
def vectorstore_n_retriever(model_name, docs, retriever_save_path='./retriever/retriever_config.json'):
    # Crear los embeddings usando el modelo especificado
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Dividir los documentos en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Crear el vectorstore y almacenar los fragmentos con sus embeddings
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory='./vectordb')

    # Configurar el retriever para recuperar documentos relevantes
    retriever = vectorstore.as_retriever()

    # Guardar la configuración del retriever para su reconstrucción
    retriever_config = {
        "model_name": model_name,
        "vectorstore_path": './vectordb',
        "retriever_save_path": retriever_save_path
    }

    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(retriever_save_path), exist_ok=True)

    # Guardar la configuración en un archivo JSON
    with open(retriever_save_path, 'w') as f:
        json.dump(retriever_config, f)

    return retriever


#FUNCION DE CARGA Y GENERACION DEL RETRIEVER 
def load_retriever(retriever_save_path='./retriever/retriever_config.json'):
    # Cargar la configuración del archivo
    with open(retriever_save_path, 'r') as f:
        retriever_config = json.load(f)

    # Reconstruir el retriever usando la configuración guardada
    embeddings = HuggingFaceEmbeddings(model_name=retriever_config["model_name"])
    vectorstore = Chroma(persist_directory=retriever_config["vectorstore_path"], embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


#SOLO LA RPIMERA VEZvectorstore_n_retriever(model_name='all-MiniLM-L6-v2', docs=load_documents(), retriever_save_path='./retriever/retriever_config.json')
#COMO SE INVOCA EL RETRIEVERretriever = load_retriever(retriever_save_path='./retriever/retriever_config.json')
#PRUEBA DE RETRIEVERoutput=retriever.get_relevant_documents('mano de obra y recursos')
#PRUEBA DE OUTPUT print(output)


#---------------------------------------------------------------------------------------------------------------------------

#VAS A TENER A OLLAMA CORRIENDO PARA PODER CORRER ESTA FUNCION.
#FUNCION QUE GENERA UNA RESPUESTA DE FONDO 
def generar_respuesta(rol, question, temperature):
    # Plantillas de system_prompt basadas en el rol
    system_prompts = {
        "assistant": (
            "Tú eres un asistente para tareas de respuesta a preguntas."
            "Usa los siguientes fragmentos de contexto recuperado para responder "
            "la pregunta. Si no sabes la respuesta, di que no "
            "sabes. Usa un máximo de tres oraciones y mantén la "
            "respuesta concisa."
            "\n\n"
            "{context}"
        ),
        "expert": (
            "Eres un experto en el tema que se va a discutir."
            "Utiliza los fragmentos de contexto proporcionados para brindar una respuesta "
            "precisa y detallada. Si no tienes suficiente información, indica que no puedes "
            "responder con certeza. Limita la respuesta a cuatro oraciones."
            "\n\n"
            "{context}"
        ),
        "advisor": (
            "Eres un consejero ofreciendo recomendaciones basadas en el contexto provisto."
            "Usa los fragmentos de contexto recuperado para dar un consejo breve y útil. "
            "Si no tienes suficiente información, menciona que necesitas más detalles para "
            "dar un consejo adecuado."
            "\n\n"
            "{context}"
        )
    }
    
    # Selección del system_prompt basado en el rol
    if rol not in system_prompts:
        raise ValueError("Invalid role. Valid roles are: 'assistant', 'expert', 'advisor'.")
    system_prompt = system_prompts[rol]
    
    # Configuración del modelo de lenguaje con la temperatura proporcionada
    llm = ChatOllama(model='llama3', temperature=temperature)
    
    # Configuración del prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Creación de la cadena de recuperación y combinación
    chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(load_retriever(retriever_save_path='./retriever/retriever_config.json'), chain)
    
    # Invocar la cadena con la pregunta proporcionada
    results = rag.invoke({"input": question})
    
    # Devolver la respuesta
    return results['answer']

# Ejemplo de uso:
#respuesta = generar_respuesta(rol='assistant', question='¿Quien es pauselino?', temperature=0.7)
#print(respuesta)

