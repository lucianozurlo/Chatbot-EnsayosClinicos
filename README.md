# Chatbot-EnsayosClinicos

### **PASO 0: Verificar versión de Python**

- **Verificación de la Versión de Python:** Se utiliza `print(sys.version)` para mostrar la versión actual de Python instalada.

### **PASO 1: Instalación de Paquetes Necesarios**
Se instalan las bibliotecas necesarias para que el chatbot funcione correctamente.

- **Transformers (`transformers`)**: Para el procesamiento de lenguaje natural.
- **Sentence Transformers (`sentence_transformers`)**: Para crear embeddings eficientes de texto.
- **HNSWlib (`hnswlib`)**: Realiza búsquedas rápidas de vecinos más cercanos.
- **Numpy (`numpy<2.0`)**: Utiliza una versión compatible para operaciones matemáticas.
- **PyPDF2 (`PyPDF2`)**: Manejo y extracción de texto desde archivos PDF.
- **Dotenv (`python-dotenv`)**: Gestiona variables de entorno desde un archivo `.env`.
- **Tenacity (`tenacity`)**: Manejo de reintentos con lógica exponencial.
- **Llama Index (`llama-index` y extensiones para Gemini)**: Proporciona integración con el modelo Gemini.
- **Tqdm (`tqdm`)**: Barra de progreso visual.

### **PASO 2: Importar Librerías y Configurar Logging**
Se importan todas las librerías necesarias y se configura un sistema de logs para monitorear el flujo del programa.

- **Importación de Librerías:** Incluye módulos estándar como `os`, `json`, y `logging`, y bibliotecas específicas del proyecto.
- **Configuración del Logging:** 
  - Configura un formato estándar para los mensajes de log.
  - Establece que los mensajes se impriman directamente en la consola.
  - Define el nivel de logging como `INFO` para capturar detalles esenciales del flujo.
- **Carga de Variables de Entorno:**
  - Usa `load_dotenv()` para cargar claves de API u otras configuraciones sensibles desde un archivo `.env`.

### **PASO 3: Cargar Documentos**
Se cargan documentos desde archivos o directorios para analizarlos y extraer contenido relevante.

- **Función `load_documents`:**
  - Permite cargar documentos de varios formatos (`.txt`, `.json`, `.pdf`).
  - Verifica si la fuente existe y lanza un error si no es válida.
  - Itera sobre los archivos en un directorio o procesa un único archivo.
- **Función `extract_content`:**
  - Maneja la lógica para extraer contenido dependiendo del formato:
    - **TXT:** Divide el contenido en bloques usando delimitadores.
    - **JSON:** Carga y devuelve el contenido en formato de diccionario.
    - **PDF:** Extrae texto de todas las páginas utilizando `PdfReader`.

### **PASO 4: Configurar la Clave API de Gemini**
Configura la conexión al modelo Gemini utilizando la clave API proporcionada en un archivo `.env`.

- **Función `configure_gemini`:**
  - Recupera la clave API desde las variables de entorno.
  - Inicializa la instancia de Gemini usando la clave recuperada.
  - Lanza un error si la clave no está configurada correctamente.

### **PASO 5: Configurar el Modelo de Embeddings**
Configura el modelo preentrenado de embeddings que se usará para calcular similitudes de texto.

- **Modelo Utilizado:** `"paraphrase-multilingual-MiniLM-L12-v2"`.
- **Carga del Modelo:** Se inicializa con `SentenceTransformer` para generar embeddings.
- **Función `doc_enfermedad`:**
  - Compara embeddings de la pregunta con los nombres de archivos cargados.
  - Devuelve el índice del archivo más relevante basado en la similitud de coseno.

### **PASO 5: Configurar el Modelo de Embeddings**
Configura el modelo preentrenado de embeddings que se usará para calcular similitudes de texto.

- **Modelo Utilizado:** `"paraphrase-multilingual-MiniLM-L12-v2"`.
- **Carga del Modelo:** Se inicializa con `SentenceTransformer` para generar embeddings.
- **Función `doc_enfermedad`:**
  - Compara embeddings de la pregunta con los nombres de archivos cargados.
  - Devuelve el índice del archivo más relevante basado en la similitud de coseno.

### **PASO 6: Crear Clases para Documentos e Índices**
Define clases para manejar documentos y realizar búsquedas eficientes en índices de texto.

- **Clase `Document`:**
  - Representa un documento con contenido (`page_content`) y metadatos.
  - Implementa un método `__str__` para mostrar información relevante del documento.
- **Clase `HNSWIndex`:**
  - Implementa un índice de vecinos más cercanos usando HNSWlib.
  - Admite búsquedas rápidas basadas en similitud de embeddings.

### **PASO 7: Procesar Documentos y Crear Índices**
Convierte documentos en bloques manejables y genera índices para búsquedas rápidas.

- **Función `desdobla_doc`:**
  - Extrae información relevante de cada documento.
  - Crea embeddings para los textos procesados.
  - Genera un índice HNSWlib con estos embeddings.
- **Proceso por Archivo:** Itera sobre los documentos cargados para crear bloques e índices.

### **PASO 8: Traducir Preguntas y Respuestas**
Se encarga de traducir texto entre idiomas usando el modelo Gemini.

- **Función `traducir`:**
  - Envía el texto a traducir como un mensaje al modelo Gemini.
  - Retorna la traducción y registra el tiempo tomado para la operación.

### **PASO 9: Generar Respuestas**
Genera respuestas específicas basadas en la pregunta del usuario y el contexto recuperado.

- **Función `categorizar_pregunta`:**
  - Clasifica la pregunta en categorías como "tratamiento", "ensayo", etc.
- **Función `generar_prompt`:**
  - Crea prompts personalizados según la categoría.
- **Función `es_saludo`:**
  - Detecta si la entrada del usuario es un saludo.
- **Función `responder_saludo`:**
  - Devuelve una respuesta aleatoria de saludo.

### **PASO 10: Función Principal para Responder Preguntas**
Integra todos los pasos anteriores para generar respuestas completas.

- **Función `obtener_respuesta_cacheada`:**
  - Verifica si una pregunta ya tiene respuesta en caché.
- **Función `guardar_respuesta_cacheada`:**
  - Almacena la respuesta generada en caché.
- **Función `responder_pregunta`:**
  - Traduce la pregunta.
  - Recupera el contexto relevante.
  - Genera la respuesta y la almacena en caché.

### **PASO 11: Interfaz CLI**
Proporciona una interfaz interactiva en línea de comandos para que los usuarios interactúen con el chatbot.

- **Inicialización del Caché:** Crea el directorio de caché si no existe.
- **Bucle Principal:**
  - Permite al usuario ingresar preguntas.
  - Responde saludos inmediatamente si se detectan.
  - Recupera el índice y bloques relacionados con la enfermedad mencionada.
  - Genera y muestra la respuesta al usuario.
