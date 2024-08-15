# Full-RAG-webapp-OpenSource-BACKEND 🇺🇸

## Project Description

**Full-RAG-webapp-OpenSource-BACKEND** is a backend web application designed to process and manage responses generated from user-uploaded documents. It utilizes advanced information retrieval techniques and natural language generation, enabling the system to provide contextualized answers based on pre-stored and analyzed documents.

The project facilitates the uploading, processing, and searching within PDF documents, extracting relevant fragments that are then used to generate precise responses through a language model. This backend is ideal for applications requiring the integration of automatic responses based on documented content, such as virtual assistants, recommendation systems, or technical support tools.

## Project Structure

### 1. `rag.py`
This file contains the core logic for loading, processing, and retrieving documents, as well as generating responses based on those documents.

#### Main Functions:

- **`load_documents()`**: Loads all PDF documents in the most recent subdirectory within the `./uploads` folder. The documents are extracted into text fragments for further processing.

- **`vectorstore_n_retriever(model_name, docs, retriever_save_path)`**: Creates a local vector store using the loaded documents and generates a retriever to fetch relevant fragments. The retriever is saved for future use.

- **`load_retriever(retriever_save_path)`**: Loads a previously saved retriever, enabling the retrieval of relevant text fragments without reprocessing the documents.

- **`generar_respuesta(rol, question, temperature)`**: Generates a response based on the context provided by the retrieved document fragments. The response is tailored according to the specified role (assistant, expert, advisor) and utilizes a pre-trained language model to generate the text.

### 2. `backend.py`
This file sets up and manages the web application using Flask. It defines the API routes for uploading documents and generating responses based on specific questions.

#### Main Routes:

- **`/upload`**: Allows uploading of PDF documents to the server. The files are stored in a new directory within `./uploads`.

- **`/doit`**: Processes a user-submitted question using the previously loaded documents to generate a contextualized response. If this is the first invocation, the system sets up the retriever; otherwise, it loads it to speed up the response.

### 3. Dependencies and Tools Used

- **`Flask`**: A web framework for Python, used to create the backend's REST API.
- **`Flask-CORS`**: A Flask extension to allow cross-origin requests.
- **`langchain`**: A library for document processing and natural language generation.
- **`PyPDFLoader`**: Used to load and process PDF files.
- **`Chroma`**: A tool for creating and managing the vector store based on loaded documents.
- **`HuggingFaceEmbeddings`**: Used to generate embeddings for text fragments.
- **`ChatOllama`**: A language model used for generating responses based on the retrieved context.

## Problem Solved

The **Full-RAG-webapp-OpenSource-BACKEND** solves the problem of extracting and utilizing relevant information from large volumes of documents. In scenarios where quick and accurate responses based on user-uploaded documents are required, this system provides an effective solution by creating a vector store and utilizing natural language models. This is particularly useful in applications such as customer support, information retrieval from large document databases, or any system requiring responses based on pre-existing textual content.

## Usage and Deployment

To use this backend, simply run the `backend.py` file. The service will be available at `http://localhost:5001`, from where POST requests can be made to upload documents and generate responses. The system ensures that responses are generated efficiently, either by creating a new retriever or reusing an existing one as needed.

---

# Full-RAG-webapp-OpenSource-BACKEND 🇨🇴

## Descripción del proyecto

**Full-RAG-webapp-OpenSource-BACKEND** es una aplicación web backend diseñada para procesar y gestionar respuestas generadas a partir de documentos cargados por el usuario. Utiliza técnicas avanzadas de recuperación de información y generación de lenguaje natural, permitiendo que el sistema responda preguntas de manera contextualizada basándose en documentos previamente almacenados y analizados.

El proyecto facilita la carga, procesamiento y búsqueda en documentos PDF, extrayendo fragmentos relevantes que luego son utilizados para generar respuestas precisas mediante un modelo de lenguaje. Este backend es ideal para aplicaciones que requieren la integración de respuestas automáticas basadas en contenido documentado, como asistentes virtuales, sistemas de recomendación o herramientas de soporte técnico.

## Estructura del proyecto

### 1. `rag.py`
Este archivo contiene la lógica central para la carga, procesamiento y recuperación de documentos, así como para la generación de respuestas basadas en esos documentos.

#### Funciones principales:

- **`load_documents()`**: Carga todos los documentos PDF en el subdirectorio más reciente dentro de la carpeta `./uploads`. Los documentos se extraen en forma de fragmentos de texto para su posterior procesamiento.

- **`vectorstore_n_retriever(model_name, docs, retriever_save_path)`**: Crea un almacén de vectores local utilizando los documentos cargados y genera un retriever para recuperar fragmentos relevantes. El retriever se guarda para su uso futuro.

- **`load_retriever(retriever_save_path)`**: Carga un retriever previamente guardado, permitiendo la recuperación de fragmentos de texto relevantes sin necesidad de procesar nuevamente los documentos.

- **`generar_respuesta(rol, question, temperature)`**: Genera una respuesta basada en el contexto provisto por los fragmentos de documentos recuperados. La respuesta se ajusta según el rol especificado (asistente, experto, consejero) y utiliza un modelo de lenguaje preentrenado para generar el texto.

### 2. `backend.py`
Este archivo configura y gestiona la aplicación web utilizando Flask. Define las rutas API para cargar documentos y para generar respuestas basadas en preguntas específicas.

#### Rutas principales:

- **`/upload`**: Permite la carga de documentos PDF al servidor. Los archivos se almacenan en un directorio nuevo dentro de `./uploads`.

- **`/doit`**: Procesa una pregunta enviada por el usuario, utilizando los documentos previamente cargados para generar una respuesta contextualizada. Si es la primera vez que se llama, el sistema configura el retriever; en caso contrario, lo carga para agilizar la respuesta.

### 3. Dependencias y herramientas utilizadas

- **`Flask`**: Marco de trabajo web para Python, utilizado para crear la API REST del backend.
- **`Flask-CORS`**: Extensión de Flask para permitir solicitudes desde diferentes dominios (Cross-Origin Resource Sharing).
- **`langchain`**: Librería para el procesamiento de documentos y la generación de lenguaje natural.
- **`PyPDFLoader`**: Utilizada para cargar y procesar archivos PDF.
- **`Chroma`**: Herramienta para crear y gestionar el almacén de vectores basado en los documentos cargados.
- **`HuggingFaceEmbeddings`**: Utilizada para generar embeddings de los fragmentos de texto.
- **`ChatOllama`**: Modelo de lenguaje utilizado para la generación de respuestas en función del contexto recuperado.

## Problema que resuelve

El backend de **Full-RAG-webapp-OpenSource** resuelve el problema de extraer y utilizar información relevante de grandes volúmenes de documentos. En escenarios donde se necesita generar respuestas rápidas y precisas basadas en documentos cargados por usuarios, este sistema ofrece una solución efectiva mediante la creación de un almacén de vectores y la utilización de modelos de lenguaje natural. Esto es particularmente útil en aplicaciones como asistencia al cliente, búsqueda de información en grandes bases de datos documentales o cualquier sistema que requiera respuestas basadas en contenido textual preexistente.

## Uso y despliegue

Para utilizar este backend, basta con ejecutar el archivo `backend.py`. El servicio estará disponible en `http://localhost:5001`, desde donde se pueden realizar solicitudes POST para cargar documentos y generar respuestas. El sistema garantiza que las respuestas se generen de manera eficiente, ya sea creando un nuevo retriever o reutilizando uno existente, según sea necesario.
