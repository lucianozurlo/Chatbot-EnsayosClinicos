import os
import json
import logging
import hnswlib
from sentence_transformers import SentenceTransformer, util
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage
import time
import hashlib
import streamlit as st
import random

# Configuración de logs para imprimir todo en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logging.info("Librerías importadas correctamente.")

# Cargar variables de entorno desde un archivo .env
load_dotenv()
logging.info("Variables de entorno cargadas desde el archivo .env.")

# Definición de clases necesarias

class Document:
    def __init__(self, text, metadata=None):
        """
        Inicializa un documento con su contenido y metadatos.
        
        Args:
            text (str): Texto del documento.
            metadata (dict, optional): Metadatos asociados al documento.
        """
        self.page_content = text
        self.metadata = metadata or {}
    
    def __str__(self):
        """
        Representación en string del documento.
        
        Returns:
            str: Información formateada del documento.
        """
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
        """
        Inicializa el índice HNSWlib con los embeddings proporcionados.
        
        Args:
            embeddings (np.ndarray): Matriz de embeddings.
            metadata (list, optional): Lista de metadatos asociados a cada embedding.
            space (str, optional): Espacio métrico para HNSWlib.
            ef_construction (int, optional): Parámetro ef para la construcción del índice.
            M (int, optional): Parámetro M para HNSWlib.
        """
        self.dimension = embeddings.shape[1]
        self.index = hnswlib.Index(space=space, dim=self.dimension)
        self.index.init_index(max_elements=embeddings.shape[0], ef_construction=ef_construction, M=M)
        self.index.add_items(embeddings, np.arange(embeddings.shape[0]))
        self.index.set_ef(50)  # Parámetro ef para consultas
        self.metadata = metadata or []
    
    def similarity_search(self, query_vector, k=5):
        """
        Realiza una búsqueda de los k vecinos más similares.
        
        Args:
            query_vector (np.ndarray): Vector de consulta.
            k (int, optional): Número de vecinos a buscar.
        
        Returns:
            list: Lista de tuplas con metadatos y distancias.
        """
        labels, distances = self.index.knn_query(query_vector, k=k)
        return [(self.metadata[i], distances[0][j]) for j, i in enumerate(labels[0])]

@st.cache_resource
def cargar_modelo_embeddings(model_name):
    model = SentenceTransformer(model_name)
    return model

@st.cache_data
def cargar_y_procesar_documentos(ruta_fuente, _model):
    documentos = load_documents(ruta_fuente, is_directory=True)
    logging.info(f"Se cargaron {len(documentos)} documentos exitosamente.")
    
    # Precomputar los embeddings de los nombres de archivo para eficiencia
    archivos = [doc['filename'] for doc in documentos]
    archivos_embeddings = _model.encode(archivos)
    
    # Procesar todos los documentos y crear sus respectivos índices
    trozos_archivos = []
    index_archivos = []
    for i in range(len(documentos)):
        trozos, index = desdobla_doc(documentos[i], _model)
        trozos_archivos.append(trozos)
        index_archivos.append(index)
    
    logging.info("Índices HNSWlib creados para todos los documentos.")
    
    return documentos, archivos, archivos_embeddings, trozos_archivos, index_archivos

@st.cache_resource
def configurar_gemini_cached():
    return configure_gemini()

def load_documents(source, is_directory=False):
    """
    Carga documentos desde un archivo o directorio.
    
    Args:
        source (str): Ruta al archivo o directorio.
        is_directory (bool): Indica si la fuente es un directorio.
    
    Returns:
        list: Lista de diccionarios con 'filename' y 'content'.
    """
    if not os.path.exists(source):
        logging.error(f"La fuente '{source}' no existe.")
        raise FileNotFoundError(f"La fuente '{source}' no se encontró.")

    loaded_files = []
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
    Extrae el contenido del archivo según su tipo.
    
    Args:
        filepath (str): Ruta al archivo.
    
    Returns:
        list o dict o str: Contenido procesado del archivo.
    """
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            units = content.split("\n-----\n")
            return units
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        elif filepath.endswith('.pdf'):
            reader = PdfReader(filepath)
            return ''.join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        logging.error(f"Error al extraer contenido de '{filepath}': {e}")
        return None

def desdobla_doc(data2, model):
    """
    Desdobla el contenido del documento en varios `Document` con metadatos.
    Maneja JSON (asumiendo estructura de ensayos clínicos) o texto/PDF genérico.
    
    Args:
        data2 (dict): Diccionario con 'filename' y 'content'.
        model (SentenceTransformer): Modelo para generar embeddings.
    
    Returns:
        tuple: Lista de `Document` y instancia de `HNSWIndex`.
    """
    documents = []
    summaries = []
    contenido = data2['content']
    
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

                # Crear resumen en inglés para consistencia interna
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
                # Si no es dict, tratar la entrada como texto genérico
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
        embeddings = model.encode([doc.page_content for doc in documents], show_progress_bar=False)
        embeddings = np.array(embeddings).astype(np.float32)
        vector_store = HNSWIndex(embeddings, metadata=[doc.metadata for doc in documents])
    else:
        vector_store = None

    return documents, vector_store

def configure_gemini():
    """
    Configura la instancia de Gemini usando la clave API.
    
    Returns:
        Gemini: Instancia configurada del modelo Gemini.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("La clave API de Gemini no está configurada.")
        raise EnvironmentError("Configura GEMINI_API_KEY en tu archivo .env.")
    gemini = Gemini(api_key=api_key)
    logging.info("Gemini configurado correctamente.")
    return gemini

def traducir(texto, idioma_destino, gemini_llm):
    """
    Traduce texto al idioma especificado usando el modelo Gemini.
    Sin detección de idioma.
    
    Args:
        texto (str): Texto a traducir.
        idioma_destino (str): Idioma de destino.
        gemini_llm (Gemini): Instancia del modelo Gemini.
    
    Returns:
        str: Texto traducido o original en caso de fallo.
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

def generate_embedding(texto, model, embedding_cache):
    """
    Genera un embedding para el texto utilizando el modelo de embeddings.
    Usa caché para evitar recalcular embeddings de textos repetidos.
    
    Args:
        texto (str): Texto para generar el embedding.
        model (SentenceTransformer): Modelo para generar embeddings.
        embedding_cache (dict): Caché de embeddings.
    
    Returns:
        np.ndarray: Embedding generado o vector de ceros en caso de fallo.
    """
    if texto in embedding_cache:
        logging.info(f"Embedding obtenido del caché para el texto: {texto}")
        return embedding_cache[texto]
    try:
        embedding = model.encode([texto])
        embedding_cache[texto] = embedding
        logging.info(f"Embedding generado para el texto: {texto}")
        return embedding
    except Exception as e:
        logging.error(f"Error al generar el embedding: {e}")
        # Devuelve embedding vacío como fallback
        return np.zeros((1, 384))

def obtener_contexto(pregunta, index, trozos, model, gemini_llm, embedding_cache, top_k=50):
    """
    Recupera los trozos de texto más relevantes para responder la pregunta.
    Traduce la pregunta al inglés antes de buscar en el índice.
    
    Args:
        pregunta (str): Pregunta del usuario.
        index (HNSWIndex): Índice de HNSWlib para buscar similitudes.
        trozos (list): Lista de `Document` relacionados.
        model (SentenceTransformer): Modelo para generar embeddings.
        gemini_llm (Gemini): Instancia del modelo Gemini.
        embedding_cache (dict): Caché de embeddings.
        top_k (int, optional): Número de resultados a recuperar.
    
    Returns:
        str: Contexto relevante concatenado.
    """
    try:
        # Traducir la pregunta al inglés
        pregunta_en_ingles = traducir(pregunta, "inglés", gemini_llm)
        logging.info(f"Pregunta traducida al inglés: {pregunta_en_ingles}")

        # Generar embedding de la pregunta traducida
        pregunta_emb = generate_embedding(pregunta_en_ingles, model, embedding_cache)
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

def categorizar_pregunta(pregunta):
    """
    Clasifica la pregunta en categorías basadas en palabras clave.
    
    Args:
        pregunta (str): Pregunta del usuario.
    
    Returns:
        str: Categoría identificada.
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
    
    Args:
        categoria (str): Categoría de la pregunta.
        pregunta (str): Pregunta del usuario.
    
    Returns:
        str: Prompt generado.
    """
    prompts = {
        "tratamiento": f"Proporciona información sobre tratamientos en ensayos clínicos relacionados con: {pregunta}.",
        "ensayo": f"Describe los ensayos clínicos actuales relacionados con: {pregunta}.",
        "resultado": f"Explica los resultados más recientes de ensayos clínicos sobre: {pregunta}.",
        "prevención": f"Ofrece información sobre prevención y ensayos clínicos para: {pregunta}."
    }
    return prompts.get(categoria, "Por favor, responde la pregunta sobre ensayos clínicos.")

def es_saludo(pregunta):
    """
    Verifica si la pregunta del usuario es un saludo.
    
    Args:
        pregunta (str): Pregunta del usuario.
    
    Returns:
        bool: True si es un saludo, False de lo contrario.
    """
    saludos = ["hola", "buen día", "buenas", "cómo estás", "cómo te llamas", "qué tal", "estás bien", "buenas tardes", "buenas noches"]
    return any(saludo in pregunta.lower() for saludo in saludos)

def responder_saludo():
    """
    Genera una respuesta aleatoria a un saludo.
    
    Returns:
        str: Respuesta de saludo.
    """
    saludos_respuestas = [
        "¡Hola! Estoy para ayudarte con información sobre ensayos clínicos. ¿En qué puedo asistirte hoy?",
        "¡Buenas! ¿Tienes alguna pregunta sobre ensayos clínicos en enfermedades neuromusculares?",
        "¡Hola! ¿Cómo puedo ayudarte con tus consultas sobre ensayos clínicos?"
    ]
    return random.choice(saludos_respuestas)

def generar_respuesta(pregunta, contexto, prompt_especifico, gemini_llm, model, embedding_cache):
    """
    Genera una respuesta usando el contexto proporcionado y un prompt específico.
    Primero genera la respuesta en inglés, luego la traduce al español.
    
    Args:
        pregunta (str): Pregunta del usuario.
        contexto (str): Contexto relevante recuperado.
        prompt_especifico (str): Prompt adaptado a la categoría de la pregunta.
        gemini_llm (Gemini): Instancia del modelo Gemini.
        model (SentenceTransformer): Modelo para generar embeddings.
        embedding_cache (dict): Caché de embeddings.
    
    Returns:
        str: Respuesta generada en español.
    """
    mensajes = [
        ChatMessage(role="system", content="Eres un experto médico."),
        ChatMessage(role="user", content=f"{prompt_especifico}\nContexto: {contexto}\nPregunta: {pregunta}")
    ]
    start_time = time.time()
    try:
        respuesta = gemini_llm.chat(mensajes)
        elapsed_time = time.time() - start_time
        logging.info(f"Respuesta generada en inglés en {elapsed_time:.2f} segundos.")
        # Traducir la respuesta al español
        respuesta_en_espanol = traducir(respuesta.message.content, "español", gemini_llm)
        logging.info("Respuesta traducida al español.")
        return respuesta_en_espanol
    except Exception as e:
        logging.error(f"Error al generar la respuesta: {e}")
        return "Lo siento, ocurrió un error al generar la respuesta."

def generar_hash(pregunta):
    """
    Genera un hash SHA-256 para una pregunta dada.
    
    Args:
        pregunta (str): Pregunta del usuario.
    
    Returns:
        str: Hash generado.
    """
    return hashlib.sha256(pregunta.encode('utf-8')).hexdigest()

def obtener_respuesta_cacheada(pregunta):
    """
    Obtiene una respuesta cacheada para una pregunta si existe.
    
    Args:
        pregunta (str): Pregunta del usuario.
    
    Returns:
        str o None: Respuesta cacheada o None si no existe.
    """
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
    """
    Guarda una respuesta en caché para una pregunta dada.
    
    Args:
        pregunta (str): Pregunta del usuario.
        respuesta (str): Respuesta generada.
    """
    hash_pregunta = generar_hash(pregunta)
    archivo_cache = f"cache/{hash_pregunta}.json"
    try:
        os.makedirs(os.path.dirname(archivo_cache), exist_ok=True)
        with open(archivo_cache, "w", encoding='utf-8') as f:
            json.dump({"pregunta": pregunta, "respuesta": respuesta}, f, ensure_ascii=False, indent=4)
        logging.info(f"Respuesta cacheada para la pregunta: '{pregunta}'")
    except Exception as e:
        logging.error(f"Error al guardar la respuesta en caché: {e}")

def responder_pregunta(pregunta, index, trozos, model, gemini_llm, embedding_cache):
    """
    Integra categorización, obtención de contexto y generación de respuesta.
    Incluye manejo de caché para respuestas repetidas.
    
    Args:
        pregunta (str): Pregunta del usuario.
        index (HNSWIndex): Índice de HNSWlib para búsqueda de contexto.
        trozos (list): Lista de `Document` relacionados.
        model (SentenceTransformer): Modelo para generar embeddings.
        gemini_llm (Gemini): Instancia del modelo Gemini.
        embedding_cache (dict): Caché de embeddings.
    
    Returns:
        str: Respuesta generada.
    """
    try:
        if index is None or not trozos:
            logging.warning("No se encontraron índices o trozos para esta pregunta.")
            return "No se encontró información para responder tu pregunta."

        # Verificar caché
        respuesta_cacheada = obtener_respuesta_cacheada(pregunta)
        if respuesta_cacheada:
            logging.info(f"Respuesta obtenida del caché para: '{pregunta}'")
            return respuesta_cacheada

        # Categorizar la pregunta
        categoria = categorizar_pregunta(pregunta)
        logging.info(f"Categoría de la pregunta: {categoria}")

        # Generar prompt específico
        prompt_especifico = generar_prompt(categoria, pregunta)
        logging.info(f"Prompt específico: {prompt_especifico}")

        # Obtener contexto relevante
        contexto = obtener_contexto(pregunta, index, trozos, model, gemini_llm, embedding_cache)
        if not contexto.strip():
            logging.warning("No se encontró contexto relevante.")
            respuesta = "No pude encontrar información relevante para responder tu pregunta."
            guardar_respuesta_cacheada(pregunta, respuesta)
            return respuesta

        # Generar la respuesta
        respuesta = generar_respuesta(pregunta, contexto, prompt_especifico, gemini_llm, model, embedding_cache)

        # Guardar la respuesta en caché
        guardar_respuesta_cacheada(pregunta, respuesta)
        return respuesta
    except Exception as e:
        logging.error(f"Error en el proceso de responder pregunta: {e}")
        return "Ocurrió un error al procesar tu pregunta."

def doc_enfermedad(pregunta):
    """
    Identifica el índice del documento más relevante para la enfermedad en la pregunta.
    Utiliza embeddings precomputados de los nombres de archivo.
    
    Args:
        pregunta (str): Pregunta del usuario.
    
    Returns:
        int: Índice del documento más relevante.
    """
    if not documentos:
        logging.warning("No se encontraron documentos. Índice por defecto: 0.")
        return 0

    # Generar embedding de la pregunta
    preg_embedding = model.encode(pregunta)

    # Calcular similitudes con los embeddings de los nombres de archivo
    similarities = [util.cos_sim(preg_embedding, emb).item() for emb in archivos_embeddings]

    # Obtener el índice con mayor similitud
    max_index = similarities.index(max(similarities))
    return max_index

# Cargar y procesar documentos usando funciones cacheadas
ruta_fuente = 'data'  # Asegúrate de tener una carpeta 'data' con los documentos
model = cargar_modelo_embeddings("all-MiniLM-L6-v2")
documentos, archivos, archivos_embeddings, trozos_archivos, index_archivos = cargar_y_procesar_documentos(ruta_fuente, model)

# Configurar la clave API de Gemini usando función cacheada
gemini_llm = configurar_gemini_cached()

# Inicializar cachés
embedding_cache = {}
translation_cache = {}

# Crear directorio de caché si no existe
os.makedirs("cache", exist_ok=True)

# Inicializar historial en el estado de Streamlit
if 'historial' not in st.session_state:
    st.session_state.historial = []

# Configurar la página para usar toda la anchura disponible
st.set_page_config(page_title="🤖 Chatbot de Ensayos Clínicos", layout="wide")

# Título de la aplicación
st.title("🤖 Chatbot de Ensayos Clínicos")

# Descripción
st.markdown("""
Bienvenido al **Chatbot de Ensayos Clínicos**.
Conversemos sobre ensayos clínicos en enfermedades neuromusculares 
(Distrofia Muscular de Duchenne o Becker, Enfermedad de Pompe, Distrofia Miotónica, etc.).

**Instrucciones:**
- Escribe tu pregunta en el campo de abajo, indicando la enfermedad sobre la que quieres información.
- El historial de la conversación aparecerá arriba.
""")

# Crear contenedores para el historial y el input
historial_container = st.container()
input_container = st.container()

with historial_container:
    st.markdown("### 🗨️ Historial de Conversación")
    st.write("")  # Espacio adicional

    # Añadir un contenedor con scroll para el historial
    scrollable_area = st.empty()

    # Función para renderizar el historial
    def render_historial():
        with scrollable_area.container():
            st.markdown('<div class="scrollable-content">', unsafe_allow_html=True)
            for sender, message in st.session_state.historial:
                if sender == "Usuario":
                    st.markdown(f"<p style='text-align: right;'><strong>🧑 Tú:</strong> {message}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='text-align: left;'><strong>🤖 Chatbot:</strong> {message}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Renderizar el historial inicial
    render_historial()

with input_container:
    st.markdown("---")  # Línea divisoria
    # Aplicar estilos CSS para el área de chat con scroll
    st.markdown("""
    <style>
    .scrollable-content {
        height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: #F5F5F5;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Entrada de usuario
    col1, col2 = st.columns([9, 1])
    with col1:
        pregunta = st.text_input("Tu pregunta:", key="input_pregunta")
    with col2:
        enviar = st.button("Enviar", use_container_width=True)

    if enviar:
        if not pregunta.strip():
            st.warning("⚠️ Por favor, ingresa una pregunta.")
        else:
            if es_saludo(pregunta):
                respuesta_saludo = responder_saludo()
                st.session_state.historial.append(("Usuario", pregunta))
                st.session_state.historial.append(("Chatbot", respuesta_saludo))
            else:
                # Identificar la enfermedad (documento más relevante)
                idn = doc_enfermedad(pregunta)
                index = index_archivos[idn] if idn < len(index_archivos) else None
                trozos = trozos_archivos[idn] if idn < len(trozos_archivos) else []

                # Responder la pregunta
                respuesta = responder_pregunta(pregunta, index, trozos, model, gemini_llm, embedding_cache)
                st.session_state.historial.append(("Usuario", pregunta))
                st.session_state.historial.append(("Chatbot", respuesta))

        # Limpiar el campo de entrada después de enviar
        st.session_state.input_pregunta = ""

        # Renderizar el historial actualizado
        render_historial()

    # Actualizar el scroll al final después de cada mensaje
    st.markdown("""
    <script>
    var scrollableDiv = document.querySelector('.scrollable-content');
    if(scrollableDiv){
        scrollableDiv.scrollTop = scrollableDiv.scrollHeight;
    }
    </script>
    """, unsafe_allow_html=True)
