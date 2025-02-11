from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
import os
from getpass import getpass
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage  import StorageContext
from llama_index.core.service_context import ServiceContext
from llama_index.core.prompts import PromptTemplate
from llama_index.core import Settings
from llama_index.core import DocumentSummaryIndex
from llama_index.core.tools import FunctionTool
from sys import exit
import http.server
import socketserver
from flask import Flask, request, send_from_directory, render_template_string

app = Flask(__name__)

@app.route('/')
def form():
    return render_template_string('''
        <form method="POST" action="/submit" style="text-align: center;" id="myForm" onsubmit="showLoading()">
            <img src="{{ url_for('static', filename='images/XULIA1.jpg') }}" alt="Descripción de la imagen" style="display: block; margin-left: auto; margin-right: auto;">
            <br>
            <label for="textbox">Escribe tu consulta:</label>
            <br>
            <br>
            <textarea id="textbox" name="textbox" rows="5" cols="60"></textarea>
            <br>
            <br>
            <label for="options">Selecciona la colección de documentos:</label>
            <select id="options" name="options">
                <option value="Funcionarios">Normativa Funcionarios</option>
                <option value="laboral">Normativa Laborales</option>
                <option value="listas_contratacion">Listas de contratación</option>
                <option value="concursos">Concurso de traslados</option>
                <option value="funcionarizacion">Funcionarización</option>
                <option value="fas">Fondo de Acción Social</option>
                <option value="incompatibilidades">Incompatibilidades</option>
                <option value="lofasga">LOFASGA</option>
            </select>
            <br>
            <br>
            <button type="submit" style="background-color: black; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border: none; border-radius: 4px;" id="submitButton">
                Enviar
            </button>
            <div id="loading" style="display:none; margin-top: 10px;">
                <img src="{{ url_for('static', filename='images/loading.gif') }}" alt="Estoy analizando tu pregunta..." style="width: 100px; height: 100px;">
                <p>Estoy analizando tu pregunta...</p>
            </div>
        </form>

        <script>
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('submitButton').style.display = 'none'; // Hide the submit button
            }
        </script>
    ''')

@app.route('/submit', methods=['POST'])
def submit():
    textbox_value = request.form['textbox']
    option_value = request.form['options']

    chroma_collection = db.get_or_create_collection(option_value)
    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    Settings.embed_model = embed_model
    Settings.vector_stores=ChromaVectorStore(chroma_collection=chroma_collection)
    Settings.llm = llm

    index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)
    query_engine = index.as_query_engine()
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template, "max_context_size": 8000 })
    pregunta = textbox_value

    response = query_engine.query(pregunta)

    # Obtener el contexto de la respuesta
    context_nodes = response.source_nodes
    context_str = "\n\n".join([node.get_content() for node in context_nodes])
    
    return f'''
        <b>Pregunta:</b><br> {textbox_value},<br><br>
        <b>Respuesta:</b><br> {response}<br><br>
        <b>Contexto Completo:</b><br> {context_str}<br><br>
        <form action="/" method="get" style="text-align: center;">
            <button type="submit" style="background-color: black; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">
                Nueva pregunta
            </button>
        </form>
    '''


#llm = Ollama(model="phi4_temp0:latest", request_timeout=120.0,temperature=0)
llm = Gemini(model="models/gemini-2.0-flash", request_timeout=120.0,temperature=0)

#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

db = chromadb.PersistentClient(path="C:/GOOGLE_DRIVE/GDRIVE/TONI/PROY/LLM/RAGNormativa/BD")

template = (
    "Dado el contexto que te proporcionare responde en idioma español las preguntas \n"
    "Contexto:\n"
    "################################\n"
    "{context_str}"
    "Pregunta:\n"
    "################################\n"
    "{query_str}\n"
)

qa_template = PromptTemplate(template)


if __name__ == '__main__':
    app.run(debug=False)
    use_reloader=True,
    reloader_type="stat", # <-- Set the reloader_type!
"""
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(('', port), Handler) as httpd:
    print(f'Servidor en el puerto {port}')
    httpd.serve_forever()
    """


