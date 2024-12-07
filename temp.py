# %% [markdown]
# ### **PASO 1: Verificar versión de Python**
# OPORTUNIDAD DE MEJORA: Permitir versiones posteriores compatibles, no solo la versión exacta.
# Pero por ahora se mantiene la advertencia tal cual.

# %%
import sys  # Acceder a la información de la versión de Python.
import os  # Manejo de rutas, archivos y operaciones del sistema.
import logging  # Configuración y uso de logs para monitorear la ejecución.

# Configurar la variable de entorno para desactivar la paralelización de tokenizadores y evitar advertencias
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Definir la versión requerida de Python
REQUIRED_VERSION = (3, 10, 12)
current_version = sys.version_info

# Validar la versión de Python
if (current_version.major, current_version.minor, current_version.micro) != REQUIRED_VERSION:
    logging.warning("""
    **********************************************
    ** Advertencia: Versión de Python no compatible **
    **********************************************
    Este chatbot está optimizado para Python 3.10.12.
    La versión actual es Python {}.{}.{}.
    Algunas funcionalidades pueden no funcionar correctamente.
    **********************************************
    """.format(current_version.major, current_version.minor, current_version.micro))
else:
    logging.info("""
    **********************************************
    ** Versión de Python compatible **
    **********************************************
    Python 3.10.12 detectado correctamente.
    Todas las funcionalidades deberían operar sin problemas.
    **********************************************
    """)


# %% [markdown]
# ### **PASO 2: Instalación de Paquetes Necesarios**
# Se instalan las bibliotecas necesarias para que el chatbot funcione correctamente.
# 
# - **Transformers (`transformers`)**: Para el procesamiento de lenguaje natural.
# - **Sentence Transformers (`sentence_transformers`)**: Para crear embeddings eficientes de texto.
# - **HNSWlib (`hnswlib`)**: Realiza búsquedas rápidas de vecinos más cercanos.
# - **Numpy (`numpy<2.0`)**: Utiliza una versión compatible para operaciones matemáticas.
# - **PyPDF2 (`PyPDF2`)**: Manejo y extracción de texto desde archivos PDF.
# - **Dotenv (`python-dotenv`)**: Gestiona variables de entorno desde un archivo `.env`.
# - **Tenacity (`tenacity`)**: Manejo de reintentos con lógica exponencial.
# - **Llama Index (`llama-index` y extensiones para Gemini)**: Proporciona integración con el modelo Gemini.
# - **Tqdm (`tqdm`)**: Barra de progreso visual.

# %%
# Instalación de bibliotecas necesarias
# %pip install -r requirements.txt


# %% [markdown]
# ### **PASO 3: Importar Librerías y Configurar Logging**
# Se importan las librerías necesarias y se configura un sistema de logs para monitorear todo el flujo.
# OPORTUNIDAD DE MEJORA: Agregar validación adicional de librerías o manejo de excepciones aquí.

# %%
import os
import json
import logging
import hnswlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tenacity import retry, wait_exponential, stop_after_attempt
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage
import time

# Configuración de logging para imprimir todo en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Solo consola
)

logging.info("Librerías importadas correctamente.")  # Mensaje de log para confirmar importación

# Cargar variables de entorno desde un archivo .env
load_dotenv()
logging.info("Variables de entorno cargadas desde el archivo .env.")


# %% [markdown]
# ### **PASO 4: Cargar Documentos**
# Este paso carga documentos desde un archivo o un directorio. Soporta formatos .txt, .json y .pdf.
# OPORTUNIDAD DE MEJORA: Validar contenido y estructura de los JSON para evitar errores en pasos posteriores.

# %%
def load_documents(source, is_directory=False):
    """
    Carga documentos desde un archivo o un directorio. Soporta .txt, .json y .pdf.
    Además, divide los archivos .txt en unidades usando un delimitador específico.
    """
    loaded_files = []

    # Verificar si la ruta existe
    if not os.path.exists(source):
        logging.error(f"La fuente '{source}' no existe.")
        raise FileNotFoundError(f"La fuente '{source}' no se encontró.")

    if is_directory:
        logging.info(f"Iniciando carga desde el directorio: {source}.")
        for filename in os.listdir(source):
            filepath = os.path.join(source, filename)
            if os.path.isfile(filepath) and filepath.endswith(('.txt', '.json', '.pdf')):
                content = extract_content(filepath)
                if content:
                    loaded_files.append({"filename": filename, "content": content})
                    logging.info(f"Archivo '{filename}' cargado correctamente.")
    else:
        logging.info(f"Iniciando carga del archivo: {source}.")
        content = extract_content(source)
        if content:
            loaded_files.append({"filename": os.path.basename(source), "content": content})
            logging.info(f"Archivo '{os.path.basename(source)}' cargado correctamente.")

    logging.info(f"{len(loaded_files)} documentos cargados.")
    return loaded_files

def extract_content(filepath):
    """
    Extrae el contenido del archivo según su tipo (.txt, .json, .pdf).
    Si el archivo es .txt, lo divide en unidades por el delimitador '\n-----\n'.
    """
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            units = content.split("\n-----\n")
            return units  # Devolver una lista de unidades
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            # OPORTUNIDAD DE MEJORA: Validar la estructura del JSON antes de retornarlo
            return data
        elif filepath.endswith('.pdf'):
            reader = PdfReader(filepath)
            return ''.join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        logging.error(f"Error al extraer contenido de '{filepath}': {e}")
        return None

ruta_fuente = 'data' # Ejemplo usando un directorio
documentos = load_documents(ruta_fuente, is_directory=True)
logging.info(f"Se cargaron {len(documentos)} documentos exitosamente.")


# %% [markdown]
# ### **PASO 5: Configurar la Clave API de Gemini**
# Configura la conexión con el modelo de lenguaje Gemini usando la clave API.
# OPORTUNIDAD DE MEJORA: Manejo más robusto de errores en caso de que la clave no funcione.

# %%
gemini_llm = None

def configure_gemini():
    """
    Configura la instancia de Gemini usando la clave API.
    """
    global gemini_llm
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("La clave API de Gemini no está configurada.")
        raise EnvironmentError("Configura GEMINI_API_KEY en tu archivo .env.")
    gemini_llm = Gemini(api_key=api_key)
    logging.info("Gemini configurado correctamente.")

configure_gemini()


# %% [markdown]
# ### PASO 6
# Configurar el modelo de embeddings.
# OPORTUNIDAD DE MEJORA: Permitir que el nombre del modelo se establezca desde .env para mayor flexibilidad.

# %%
model_name = "all-MiniLM-L6-v2"  # Modelo para embeddings
model = SentenceTransformer(model_name)

def doc_enfermedad(pregunta):
    """
    Identifica el índice del documento más relevante para la enfermedad en la pregunta.
    Se basa en la máxima similitud entre el embedding de la pregunta y los nombres de los archivos.
    OPORTUNIDAD DE MEJORA: Si el nombre del archivo no es representativo, esto podría fallar.
    Mejorar usando embeddings del contenido en el futuro.
    """
    if not documentos:
        logging.warning("No se encontraron documentos. Devolviendo índice 0 por defecto.")
        return 0
    preg_embedding = model.encode(pregunta)
    archivos = [documentos[i]['filename'] for i in range(len(documentos))]
    doc_filenames_embeddings = [model.encode(name) for name in archivos]
    similarities = [util.cos_sim(preg_embedding, doc_emb).item() for doc_emb in doc_filenames_embeddings]
    max_index = similarities.index(max(similarities))
    return max_index


# %% [markdown]
# ### PASO 7
# Crear clases para documentos e índices.
# OPORTUNIDAD DE MEJORA: Agregar más metadatos o estructuras más complejas a futuro.

# %%
class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}
    
    def __str__(self):
        return (
            f"Título: {self.metadata.get('Title', 'N/A')}\n"
            f"Resumen: {self.metadata.get('Summary', 'N/A')}\n"
            f"Tipo de Estudio: {self.metadata.get('StudyType', 'N/A')}\n"
            f"Paises donde se desarrolla el estudio: {self.metadata.get('Countries', 'N/A')}\n"
            f"Fase en que se encuentra el estudio: {self.metadata.get('Phases', 'N/A')}\n"
            f"Identificación en ClinicaTrial: {self.metadata.get('IDestudio', 'N/A')}.\n\n"
        )

class HNSWIndex:
    def __init__(self, embeddings, metadata=None, space='cosine', ef_construction=200, M=16):
        self.dimension = embeddings.shape[1]
        self.index = hnswlib.Index(space=space, dim=self.dimension)
        self.index.init_index(max_elements=embeddings.shape[0], ef_construction=ef_construction, M=M)
        self.index.add_items(embeddings, np.arange(embeddings.shape[0]))
        self.index.set_ef(50)  # Parámetro ef para consultas
        self.metadata = metadata or []
    
    def similarity_search(self, query_vector, k=5):
        labels, distances = self.index.knn_query(query_vector, k=k)
        return [(self.metadata[i], distances[0][j]) for j, i in enumerate(labels[0])]


# %% [markdown]
# ### PASO 8
# Se procesan los documentos y se crea un índice HNSWlib para cada conjunto de documentos.
# OPORTUNIDAD DE MEJORA: Manejar otro tipo de estructuras de datos más complejas si el JSON difiere en su formato.

# %%
def desdobla_doc(data2):
    """
    Desdobla el contenido del documento en varios Documents con metadatos.
    Maneja JSON (asumiendo estructura de ensayos clínicos) o texto/PDF genérico.
    """
    documents = []
    summaries = []
    contenido = data2['content']
    
    # OPORTUNIDAD DE MEJORA: Validar la estructura en caso de JSON
    if isinstance(contenido, list):
        for entry in contenido:
            if isinstance(entry, dict):
                nctId = entry.get("IDestudio", "")
                briefTitle = entry.get("Title", "")
                summary = entry.get("Summary", "")
                studyType = entry.get("StudyType", "")
                country = entry.get("Countries", "")
                overallStatus = entry.get("OverallStatus", "")
                conditions = entry.get("Conditions", "")
                phases = entry.get("Phases", "")

                # Texto del documento en inglés (Mantener la consistencia de idioma del resumen interno)
                Summary = (
                    f"The study titled '{briefTitle}', of type '{studyType}', "
                    f"investigates the condition(s): {conditions}. "
                    f"Brief summary: {summary}. "
                    f"Current status: {overallStatus}, taking place in {country}. "
                    f"The study is classified under: {phases} phase. "
                    f"For more info, search {nctId} on ClinicalTrials."
                )
                metadata = {
                    "Title": briefTitle,
                    "Summary": Summary,
                    "StudyType": studyType,
                    "Countries": country,
                    "Phases": phases,
                    "IDestudio": nctId
                }
                doc = Document(Summary, metadata)
                documents.append(doc)
                summaries.append(Summary)
            else:
                # Si no es dict, tratamos la entrada como texto genérico
                texto = str(entry)
                metadata = {"Summary": texto}
                doc = Document(texto, metadata)
                documents.append(doc)
                summaries.append(texto)
    else:
        # Texto genérico (PDF o TXT)
        texto = str(contenido)
        metadata = {"Summary": texto}
        doc = Document(texto, metadata)
        documents.append(doc)
        summaries.append(texto)

    if documents:
        embeddings = model.encode([doc.page_content for doc in documents], show_progress_bar=True)
        embeddings = np.array(embeddings).astype(np.float32)
        vector_store = HNSWIndex(embeddings, metadata=[doc.metadata for doc in documents])
    else:
        vector_store = None

    return documents, vector_store

trozos_archivos = []
index_archivos = []
for i in range(len(documentos)):
    trozos, index = desdobla_doc(documentos[i])
    trozos_archivos.append(trozos)
    index_archivos.append(index)

logging.info("Índices HNSWlib creados para todos los documentos.")


# %% [markdown]
# ### **PASO 9: Traducir Preguntas y Respuestas**
# Traduce preguntas y respuestas entre idiomas según sea necesario utilizando Gemini.
# OPORTUNIDAD DE MEJORA: Implementar un mecanismo de caché de traducciones.

# %%
def traducir(texto, idioma_destino):
    """
    Traduce texto al idioma especificado usando el modelo Gemini.
    En caso de error, se devuelve el texto original.
    """
    start_time = time.time()
    mensajes = [
        ChatMessage(role="system", content="Actúa como un traductor."),
        ChatMessage(role="user", content=f"Por favor, traduce este texto al {idioma_destino}: {texto}")
    ]
    try:
        respuesta = gemini_llm.chat(mensajes)
        elapsed_time = time.time() - start_time
        logging.info(f"Traducción completada en {elapsed_time:.2f} segundos.")
        return respuesta.message.content.strip()
    except Exception as e:
        logging.error(f"Error al traducir: {e}")
        return texto  # fallback

def generate_embedding(texto):
    """
    Genera un embedding para la pregunta utilizando el modelo de embeddings.
    """
    try:
        embedding = model.encode([texto])
        logging.info(f"Embedding generado para el texto: {texto}")
        return embedding
    except Exception as e:
        logging.error(f"Error al generar el embedding: {e}")
        return np.zeros((1, 384)) # Dimensión aproximada fallback para all-MiniLM-L6-v2


def obtener_contexto(pregunta, index, trozos, top_k=50):
    """
    Recupera los trozos de texto más relevantes para responder la pregunta.
    Traduce la pregunta al inglés antes de buscar en el índice.
    """
    try:
        # Traducir la pregunta al inglés
        pregunta_en_ingles = traducir(pregunta, "inglés")
        logging.info(f"Pregunta traducida al inglés: {pregunta_en_ingles}")

        # Generar embedding de la pregunta traducida
        pregunta_emb = generate_embedding(pregunta_en_ingles)
        logging.info("Embedding generado para la pregunta.")

        # Buscar en el índice
        results = index.similarity_search(pregunta_emb, k=top_k)
        texto = ""
        for entry in results:
            resum = entry[0]["Summary"]
            texto += resum + "\n"

        logging.info("Contexto relevante recuperado para la pregunta.")
        return texto
    except Exception as e:
        logging.error(f"Error al obtener el contexto: {e}")
        return ""


# %% [markdown]
# ### **PASO 10: Generar Respuestas**
# Genera respuestas utilizando el modelo Gemini y el contexto proporcionado.
# OPORTUNIDAD DE MEJORA: Ajustar prompts dinámicamente según la complejidad de la pregunta.

# %%
def categorizar_pregunta(pregunta):
    """
    Clasifica la pregunta en categorías.
    OPORTUNIDAD DE MEJORA: Usar un modelo de clasificación semántica en lugar de palabras clave.
    """
    categorias = {
        "tratamiento": ["tratamiento", "medicación", "cura", "terapia", "fármaco"],
        "ensayo": ["ensayo", "estudio", "prueba", "investigación", "trial"],
        "resultado": ["resultado", "efectividad", "resultados", "éxito", "fracaso"],
        "prevención": ["prevención", "previene", "evitar", "reducción de riesgo"]
    }
    for categoria, palabras in categorias.items():
        if any(palabra in pregunta.lower() for palabra in palabras):
            return categoria
    return "general"

def generar_prompt(categoria, pregunta):
    """
    Genera un prompt específico basado en la categoría de la pregunta.
    """
    prompts = {
        "tratamiento": f"Proporciona información sobre tratamientos en ensayos clínicos relacionados con: {pregunta}.",
        "ensayo": f"Describe los ensayos clínicos actuales relacionados con: {pregunta}.",
        "resultado": f"Explica los resultados más recientes de ensayos clínicos sobre: {pregunta}.",
        "prevención": f"Ofrece información sobre prevención y ensayos clínicos para: {pregunta}."
    }
    return prompts.get(categoria, "Por favor, responde la pregunta sobre ensayos clínicos.")


def es_saludo(pregunta):
    saludos = ["hola", "buen día", "buenas", "cómo estás", "cómo te llamas", "qué tal", "estás bien", "buenas tardes", "buenas noches"]
    return any(saludo in pregunta.lower() for saludo in saludos)

def responder_saludo():
    saludos_respuestas = [
        "¡Hola! Estoy para ayudarte con información sobre ensayos clínicos. ¿En qué puedo asistirte hoy?",
        "¡Buenas! Tenés alguna pregunta sobre ensayos clínicos en enfermedades neuromusculares?",
        "¡Hola! ¿Cómo puedo ayudarte con tus consultas sobre ensayos clínicos?"
    ]
    import random
    return random.choice(saludos_respuestas)

def generar_respuesta(pregunta, contexto, prompt_especifico):
    """
    Genera una respuesta usando el contexto proporcionado y un prompt específico.
    Primero genera la respuesta en inglés, luego la traduce al español.
    OPORTUNIDAD DE MEJORA: Si el usuario pregunta en inglés, devolver directamente en inglés.
    """
    start_time = time.time()
    mensajes = [
        ChatMessage(role="system", content="Eres un experto médico."),
        ChatMessage(role="user", content=f"{prompt_especifico}\nContexto: {contexto}\nPregunta: {pregunta}")
    ]
    try:
        respuesta = gemini_llm.chat(mensajes)
        elapsed_time = time.time() - start_time
        logging.info(f"Respuesta generada en inglés en {elapsed_time:.2f} segundos.")

        # Traducir la respuesta al español
        respuesta_en_espanol = traducir(respuesta.message.content, "español")
        logging.info("Respuesta traducida al español.")
        return respuesta_en_espanol
    except Exception as e:
        logging.error(f"Error al generar la respuesta: {e}")
        return "Lo siento, ocurrió un error al generar la respuesta."


# %% [markdown]
# ### **PASO 11: Función Principal para Responder Preguntas**
# Integra todos los pasos previos para traducir, recuperar contexto y generar respuestas.
# Incluye manejo de caché.
# OPORTUNIDAD DE MEJORA: Expirar caché después de cierto tiempo.

# %%
import json
import os
import hashlib

def generar_hash(pregunta):
    return hashlib.sha256(pregunta.encode('utf-8')).hexdigest()

def obtener_respuesta_cacheada(pregunta):
    hash_pregunta = generar_hash(pregunta)
    archivo_cache = f"cache/{hash_pregunta}.json"
    if os.path.exists(archivo_cache):
        try:
            with open(archivo_cache, "r", encoding='utf-8') as f:
                datos = json.load(f)
                return datos.get("respuesta", None)
        except Exception as e:
            logging.error(f"Error al leer el caché: {e}")
            return None
    return None

def guardar_respuesta_cacheada(pregunta, respuesta):
    hash_pregunta = generar_hash(pregunta)
    archivo_cache = f"cache/{hash_pregunta}.json"
    try:
        os.makedirs(os.path.dirname(archivo_cache), exist_ok=True)
        with open(archivo_cache, "w", encoding='utf-8') as f:
            json.dump({"pregunta": pregunta, "respuesta": respuesta}, f, ensure_ascii=False, indent=4)
        logging.info(f"Respuesta cacheada para la pregunta: '{pregunta}'")
    except Exception as e:
        logging.error(f"Error al guardar la respuesta en caché: {e}")

def responder_pregunta(pregunta, index, trozos):
    """
    Integra todos los pasos: categorización, traducción, recuperación de contexto y generación de respuestas.
    Incluye manejo de caché.
    """
    try:
        if index is None or not trozos:
            logging.warning("No se pudieron generar índices o no hay trozos. Devolviendo respuesta genérica.")
            return "No se encontró información para responder tu pregunta."

        # Verificar caché
        respuesta_cacheada = obtener_respuesta_cacheada(pregunta)
        if respuesta_cacheada:
            logging.info(f"Respuesta obtenida del caché: '{pregunta}'")
            return respuesta_cacheada

        categoria = categorizar_pregunta(pregunta)
        logging.info(f"Categoría de la pregunta: {categoria}")

        prompt_especifico = generar_prompt(categoria, pregunta)
        logging.info(f"Prompt específico: {prompt_especifico}")

        contexto = obtener_contexto(pregunta, index, trozos)
        if not contexto.strip():
            logging.warning("No se encontró contexto relevante.")
            respuesta = "No pude encontrar información relevante para responder tu pregunta."
            guardar_respuesta_cacheada(pregunta, respuesta)
            return respuesta

        respuesta = generar_respuesta(pregunta, contexto, prompt_especifico)

        guardar_respuesta_cacheada(pregunta, respuesta)
        return respuesta
    except Exception as e:
        logging.error(f"Error en el proceso de responder pregunta: {e}")
        return "Ocurrió un error al procesar tu pregunta."


# %% [markdown]
# ### **PASO 12: Interfaz CLI**
# Proporciona una interfaz interactiva para que los usuarios puedan hacer preguntas.
# OPORTUNIDAD DE MEJORA: Crear una interfaz web o un frontend más amigable.

# %%
if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    
    if len(documentos) == 0:
        print("No se cargaron documentos. Por favor, verifica el directorio 'data'.")
        logging.error("No se encontraron documentos. Finalizando.")
    else:
        print("Bienvenido al Chatbot de Ensayos Clínicos")
        print("Conversemos sobre Ensayos Clínicos\n de enfermedades neuromusculares: Distrofia Muscular de Duchenne o de Becker, Enfermedad de Pompe, Distrofia Miotónica, etc.")
        print("Escribí tu pregunta, indicando claramente la enfermedad sobre la que quieres información de ensayos clínicos. Escribe 'salir' para terminar.")
        while True:
            pregunta = input("Tu pregunta: ").strip()
            if pregunta.lower() in ['salir', 'chau', 'exit', 'quit']:
                print("¡Chau!")
                logging.info("El usuario ha finalizado la sesión.")
                break
            if es_saludo(pregunta):
                respuesta_saludo = responder_saludo()
                print(respuesta_saludo)
                logging.info("Se detectó un saludo.")
                continue
            
            # Identificar la enfermedad (documento más relevante)
            idn = doc_enfermedad(pregunta)
            index = index_archivos[idn] if idn < len(index_archivos) else None
            trozos = trozos_archivos[idn] if idn < len(trozos_archivos) else []

            # Responder pregunta
            respuesta = responder_pregunta(pregunta, index, trozos)
            print(f"Respuesta: {respuesta}")
