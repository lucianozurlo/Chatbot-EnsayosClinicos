# %% [markdown]
# ### **PASO 1: Verificar versión de Python**
# Se valida que la versión de Python sea la requerida.
#
# Se agregan docstrings y type hints cuando corresponda.
# El manejo de errores en esta sección no es crítico, pero se deja más claro el log.

import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

REQUIRED_VERSION = (3, 10, 12)
current_version = sys.version_info

if (current_version.major, current_version.minor, current_version.micro) != REQUIRED_VERSION:
    logging.warning(f"Advertencia: Se está utilizando Python {current_version.major}.{current_version.minor}.{current_version.micro} en lugar de {REQUIRED_VERSION[0]}.{REQUIRED_VERSION[1]}.{REQUIRED_VERSION[2]}.")


# %%
# %% [markdown]
# ### **PASO 2: Instalación de Paquetes Necesarios**
# Se asume que ya están instalados.
# (No se modifica esta sección, ya que se solicitó solo type hints, docstrings, errores y doc_enfermedad)

# %%
# %pip install -r requirements.txt


# %%
# %% [markdown]
# ### **PASO 3: Importar Librerías y Configurar Logging**
# Se agregan docstrings en funciones más adelante.
# Se mejora manejo de excepciones en puntos críticos.

import hnswlib
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

logging.info("Librerías importadas correctamente.")

load_dotenv()
logging.info("Variables de entorno cargadas desde el archivo .env.")


# %%
# %% [markdown]
# ### **PASO 4: Cargar Documentos**
# Se agregan docstrings y type hints. Se mejora manejo de excepciones.

def extract_content(filepath: str) -> Optional[Any]:
    """
    Extrae el contenido del archivo según su tipo.
    
    Parámetros:
        filepath (str): Ruta del archivo.
    
    Retorna:
        Optional[Any]: Contenido del archivo extraído o None si falla.
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
        else:
            logging.warning(f"Formato de archivo no soportado: {filepath}")
            return None
    except Exception as e:
        logging.error(f"Error al extraer contenido de '{filepath}': {e}")
        return None

def load_documents(source: str, is_directory: bool=False) -> List[Dict[str, Any]]:
    """
    Carga documentos desde un archivo o un directorio.
    
    Parámetros:
        source (str): Ruta del archivo o directorio.
        is_directory (bool): Indica si 'source' es un directorio.
        
    Retorna:
        List[Dict[str, Any]]: Lista de diccionarios con 'filename' y 'content'.
    """
    if not os.path.exists(source):
        logging.error(f"La fuente '{source}' no existe.")
        raise FileNotFoundError(f"La fuente '{source}' no se encontró.")

    loaded_files = []
    try:
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
    except Exception as e:
        logging.error(f"Error al cargar documentos desde '{source}': {e}")
        return loaded_files

ruta_fuente = 'data'
documentos = load_documents(ruta_fuente, is_directory=True)


# %%
# %% [markdown]
# ### **PASO 5: Configurar la Clave API de Gemini**
# Se agrega manejo de excepciones con logs claros.

gemini_llm = None

def configure_gemini() -> Gemini:
    """
    Configura la instancia de Gemini usando la clave API.
    Lanza una excepción EnvironmentError si la clave no está configurada.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("La clave API de Gemini no está configurada en el archivo .env.")
        raise EnvironmentError("Configura GEMINI_API_KEY antes de continuar.")
    gemini = Gemini(api_key=api_key)
    logging.info("Gemini configurado correctamente.")
    return gemini

gemini_llm = configure_gemini()


# %%
# %% [markdown]
# ### **PASO 6: Configurar el Modelo de Embeddings**
# Se agrega docstring y type hints.

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
logging.info("Modelo de embeddings cargado.")


# %%
# %% [markdown]
# ### **PASO 7: Crear Clases para Documentos e Índices**
# Se agregan docstrings y type hints.

class Document:
    """
    Representa un documento con texto y metadatos.
    """
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]]=None):
        self.page_content: str = text
        self.metadata: Dict[str, Any] = metadata or {}
    
    def __str__(self) -> str:
        return (
            f"Título: {self.metadata.get('Title', 'N/A')}\n"
            f"Resumen: {self.metadata.get('Summary', 'N/A')}\n"
            f"Tipo de Estudio: {self.metadata.get('StudyType', 'N/A')}\n"
            f"Paises: {self.metadata.get('Countries', 'N/A')}\n"
            f"Fase: {self.metadata.get('Phases', 'N/A')}\n"
            f"IDestudio: {self.metadata.get('IDestudio', 'N/A')}\n"
        )

class HNSWIndex:
    """
    Índice HNSWlib para búsqueda por similitud.
    """
    def __init__(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], space: str='cosine', ef_construction:int=200, M:int=16):
        self.dimension: int = embeddings.shape[1]
        self.index = hnswlib.Index(space=space, dim=self.dimension)
        self.index.init_index(max_elements=embeddings.shape[0], ef_construction=ef_construction, M=M)
        self.index.add_items(embeddings, np.arange(embeddings.shape[0]))
        self.index.set_ef(50)
        self.metadata: List[Dict[str, Any]] = metadata
    
    def similarity_search(self, query_vector: np.ndarray, k: int=5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Realiza una búsqueda de similitud y devuelve los metadatos y distancias.
        
        Parámetros:
            query_vector (np.ndarray): Embedding de la consulta.
            k (int): Número de resultados a devolver.
        
        Retorna:
            List[Tuple[Dict[str, Any], float]]: Lista de tuplas (metadata, distancia).
        """
        try:
            labels, distances = self.index.knn_query(query_vector, k=k)
            return [(self.metadata[i], distances[0][j]) for j, i in enumerate(labels[0])]
        except Exception as e:
            logging.error(f"Error en similarity_search: {e}")
            return []


# %%
# %% [markdown]
# ### **PASO 8: Procesar Documentos y Crear Índices**
# Se agrega docstring y manejo de excepciones con logs.

def desdobla_doc(data2: Dict[str, Any]) -> Tuple[List[Document], Optional[HNSWIndex]]:
    """
    Desdobla el contenido del documento en varios Documents y crea un índice HNSW.
    Maneja JSON con ensayos clínicos y texto genérico.
    
    Parámetros:
        data2 (Dict[str, Any]): Diccionario con 'filename' y 'content'.
    
    Retorna:
        (List[Document], Optional[HNSWIndex]): Lista de Documents y el índice HNSW asociado.
    """
    documents = []
    contenido = data2['content']

    try:
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
                else:
                    texto = str(entry)
                    metadata = {"Summary": texto}
                    doc = Document(texto, metadata)
                    documents.append(doc)
        else:
            texto = str(contenido)
            metadata = {"Summary": texto}
            doc = Document(texto, metadata)
            documents.append(doc)

        if documents:
            embeddings = model.encode([d.page_content for d in documents], show_progress_bar=False)
            embeddings = np.array(embeddings).astype(np.float32)
            vector_store = HNSWIndex(embeddings, metadata=[d.metadata for d in documents])
        else:
            vector_store = None

        return documents, vector_store
    except Exception as e:
        logging.error(f"Error al desdoblar documento: {e}")
        return [], None

trozos_archivos = []
index_archivos = []
for i, doc_data in enumerate(documentos):
    trozos, index = desdobla_doc(doc_data)
    trozos_archivos.append(trozos)
    index_archivos.append(index)

logging.info("Índices HNSWlib creados para todos los documentos.")


# %%
# %% [markdown]
# ### Mejora 7: Mayor Precisión en doc_enfermedad()
# Calculamos embeddings a nivel de documento. Para ello promediamos los embeddings de todos los trozos del documento.
# Luego, en doc_enfermedad usamos estos embeddings para encontrar el documento más relevante.

doc_embeddings = []
for trozos in trozos_archivos:
    if trozos:
        # Obtenemos embeddings de todos los trozos y promediamos
        all_texts = [t.page_content for t in trozos]
        emb = model.encode(all_texts, show_progress_bar=False)
        emb_avg = np.mean(emb, axis=0)  # Vector promedio
        doc_embeddings.append(emb_avg)
    else:
        doc_embeddings.append(np.zeros((384,)))  # Vectores vacíos si no hay trozos


# %%
# %% [markdown]
# ### **PASO 9: Traducir Preguntas y Respuestas**
# Se agregan docstrings y manejo de excepciones.

def traducir(texto: str, idioma_destino: str) -> str:
    """
    Traduce el texto al idioma especificado usando Gemini.
    
    Parámetros:
        texto (str): Texto a traducir.
        idioma_destino (str): Idioma al que se traduce.
    
    Retorna:
        str: Texto traducido o el original si ocurre un error.
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
        return texto

def generate_embedding(texto: str) -> np.ndarray:
    """
    Genera un embedding para el texto dado.
    En caso de error, devuelve un vector vacío.
    
    Parámetros:
        texto (str): Texto a embeber.
        
    Retorna:
        np.ndarray: Embedding del texto.
    """
    try:
        embedding = model.encode([texto])
        logging.info(f"Embedding generado para el texto: {texto}")
        return embedding
    except Exception as e:
        logging.error(f"Error al generar el embedding: {e}")
        return np.zeros((1, 384))

def obtener_contexto(pregunta: str, index: HNSWIndex, trozos: List[Document], top_k: int=50) -> str:
    """
    Recupera los trozos más relevantes para responder la pregunta.
    Primero traduce la pregunta al inglés, genera el embedding, y busca en el índice.
    
    Parámetros:
        pregunta (str): Pregunta del usuario.
        index (HNSWIndex): Índice HNSW para búsqueda.
        trozos (List[Document]): Lista de documentos (trozos) a consultar.
        top_k (int): Número de resultados a retornar.
        
    Retorna:
        str: Contexto concatenado de los trozos más relevantes.
    """
    try:
        pregunta_en_ingles = traducir(pregunta, "inglés")
        logging.info(f"Pregunta traducida al inglés: {pregunta_en_ingles}")

        pregunta_emb = generate_embedding(pregunta_en_ingles)
        logging.info("Embedding de pregunta generado correctamente.")

        results = index.similarity_search(pregunta_emb, k=top_k)
        texto = ""
        for entry in results:
            resum = entry[0]["Summary"]
            texto += resum + "\n"

        logging.info("Contexto relevante recuperado con éxito.")
        return texto
    except Exception as e:
        logging.error(f"Error al obtener el contexto: {e}")
        return ""


# %%
# %% [markdown]
# ### **PASO 10: Generar Respuestas**
# Se agregan docstrings y type hints. Manejo de excepciones con logs.

def categorizar_pregunta(pregunta: str) -> str:
    """
    Clasifica la pregunta según palabras clave.
    
    Parámetros:
        pregunta (str): La pregunta del usuario.
    
    Retorna:
        str: Categoría de la pregunta ('tratamiento', 'ensayo', 'resultado', 'prevención', o 'general').
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

def generar_prompt(categoria: str, pregunta: str) -> str:
    """
    Genera un prompt específico según la categoría de la pregunta.
    
    Parámetros:
        categoria (str): Categoría de la pregunta.
        pregunta (str): Pregunta del usuario.
        
    Retorna:
        str: Prompt personalizado.
    """
    prompts = {
        "tratamiento": f"Proporciona información sobre tratamientos en ensayos clínicos relacionados con: {pregunta}.",
        "ensayo": f"Describe los ensayos clínicos actuales relacionados con: {pregunta}.",
        "resultado": f"Explica los resultados más recientes de ensayos clínicos sobre: {pregunta}.",
        "prevención": f"Ofrece información sobre prevención y ensayos clínicos para: {pregunta}."
    }
    return prompts.get(categoria, "Por favor, responde la pregunta sobre ensayos clínicos.")

def es_saludo(pregunta: str) -> bool:
    """
    Determina si la pregunta es un saludo.
    
    Parámetros:
        pregunta (str): Pregunta del usuario.
        
    Retorna:
        bool: True si es un saludo, False en caso contrario.
    """
    saludos = ["hola", "buen día", "buenas", "cómo estás", "cómo te llamas", "qué tal", "estás bien", "buenas tardes", "buenas noches"]
    return any(saludo in pregunta.lower() for saludo in saludos)

def responder_saludo() -> str:
    """
    Retorna una respuesta aleatoria a un saludo.
    
    Retorna:
        str: Respuesta a un saludo.
    """
    saludos_respuestas = [
        "¡Hola! Estoy para ayudarte con información sobre ensayos clínicos. ¿En qué puedo asistirte hoy?",
        "¡Buenas! Tenés alguna pregunta sobre ensayos clínicos en enfermedades neuromusculares?",
        "¡Hola! ¿Cómo puedo ayudarte con tus consultas sobre ensayos clínicos?"
    ]
    import random
    return random.choice(saludos_respuestas)

def generar_respuesta(pregunta: str, contexto: str, prompt_especifico: str) -> str:
    """
    Genera una respuesta en inglés usando Gemini, luego la traduce al español.
    
    Parámetros:
        pregunta (str): Pregunta del usuario.
        contexto (str): Contexto relevante de los documentos.
        prompt_especifico (str): Prompt específico según la categoría.
        
    Retorna:
        str: Respuesta final en español.
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
        respuesta_en_espanol = traducir(respuesta.message.content, "español")
        logging.info("Respuesta traducida al español.")
        return respuesta_en_espanol
    except Exception as e:
        logging.error(f"Error al generar la respuesta: {e}")
        return "Lo siento, ocurrió un error al generar la respuesta."


# %%
# %% [markdown]
# ### **PASO 11: Función Principal para Responder Preguntas**
# Se agregan docstrings y manejo de excepciones más claros.

def generar_hash(pregunta: str) -> str:
    """
    Genera un hash SHA-256 de la pregunta.
    """
    return hashlib.sha256(pregunta.encode('utf-8')).hexdigest()

def obtener_respuesta_cacheada(pregunta: str) -> Optional[str]:
    """
    Verifica si existe una respuesta cacheada para la pregunta.
    
    Parámetros:
        pregunta (str): Pregunta del usuario.
        
    Retorna:
        Optional[str]: Respuesta cacheada o None si no existe.
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

def guardar_respuesta_cacheada(pregunta: str, respuesta: str) -> None:
    """
    Guarda la respuesta en caché.
    
    Parámetros:
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

def responder_pregunta(pregunta: str, index, trozos: List[Document]) -> str:
    """
    Integra todos los pasos: categorización, contexto, generación de respuesta y caché.
    
    Parámetros:
        pregunta (str): Pregunta del usuario.
        index (HNSWIndex): Índice HNSW para búsqueda.
        trozos (List[Document]): Lista de trozos de un documento.
        
    Retorna:
        str: Respuesta final en español.
    """
    try:
        if index is None or not trozos:
            logging.warning("No se encontraron índices o trozos para esta pregunta.")
            return "No se encontró información para responder tu pregunta."

        respuesta_cacheada = obtener_respuesta_cacheada(pregunta)
        if respuesta_cacheada:
            logging.info(f"Respuesta obtenida del caché para: '{pregunta}'")
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
        logging.error(f"Error en responder_pregunta: {e}")
        return "Ocurrió un error al procesar tu pregunta."


# %%
# %% [markdown]
# ### Mejora 7 Aplicada en doc_enfermedad()
# Ahora doc_enfermedad usa `doc_embeddings` en lugar de los nombres de archivo.

def doc_enfermedad(pregunta: str) -> int:
    """
    Determina el índice del documento más relevante para la pregunta usando embeddings del contenido.
    
    Parámetros:
        pregunta (str): Pregunta del usuario.
    
    Retorna:
        int: Índice del documento más relevante.
    """
    if not documentos:
        logging.warning("No se encontraron documentos. Devolviendo índice 0 por defecto.")
        return 0
    
    try:
        preg_embedding = model.encode([pregunta])
        sims = [util.cos_sim(preg_embedding, emb.reshape(1,-1)).item() for emb in doc_embeddings]
        max_index = sims.index(max(sims))
        return max_index
    except Exception as e:
        logging.error(f"Error en doc_enfermedad: {e}")
        return 0


# %%
# %% [markdown]
# ### **PASO 12: Interfaz CLI**
# Se agregan logs más claros en caso de errores. Ya se cuenta con docstrings en funciones.

if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    
    if len(documentos) == 0:
        print("No se cargaron documentos. Por favor, verifica el directorio 'data'.")
        logging.error("No se encontraron documentos. Finalizando.")
    else:
        print("Bienvenido al Chatbot de Ensayos Clínicos")
        print("Conversemos sobre Ensayos Clínicos en enfermedades neuromusculares.")
        print("Escribí tu pregunta, indicando la enfermedad sobre la que quieres información. Escribí 'salir' para terminar.")
        
        while True:
            try:
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

                idn = doc_enfermedad(pregunta)
                index = index_archivos[idn] if idn < len(index_archivos) else None
                trozos = trozos_archivos[idn] if idn < len(trozos_archivos) else []

                respuesta = responder_pregunta(pregunta, index, trozos)
                print(f"Respuesta: {respuesta}")
            except Exception as e:
                logging.error(f"Error en el ciclo principal: {e}")
                print("Ocurrió un error inesperado. Por favor, intenta nuevamente.")
