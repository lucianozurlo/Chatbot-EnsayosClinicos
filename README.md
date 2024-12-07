# Chatbot: Ensayos Clínicos

**Trabajo Final de Diplomatura en Inteligencia Artificial**

Grupo 10:

Dra. Viviana Lencina, 
vblencina@mail.austral.edu.ar 

L.S.I, Carlos Ponce, 
coponce@mail.austral.edu.ar  

T.E.N. Enzo Zapata, 
ezapata@mail.austral.edu.ar 

D.G. Luciano Zurlo, 
lzurlo@mail.austral.edu.ar

---
### **PASO 1: Verificar versión de Python**
Se valida que la versión de Python sea la requerida.

- **Líneas Clave:**
  - Importación de librerías para verificar la versión (`sys`, `os`).
  - Configuración de logging para advertencias y mensajes informativos.
  - Comparación de la versión actual con la requerida.

### **PASO 2: Instalación de Paquetes Necesarios**
Se listan las bibliotecas requeridas para el funcionamiento del chatbot.

- **Líneas Clave:**
  - Uso de `requirements.txt` para instalar dependencias.
  - Listado de las principales bibliotecas usadas.

### **PASO 3: Importar Librerías y Configurar Logging**
Importa todas las librerías necesarias y configura un sistema de logs.

- **Líneas Clave:**
  - Importación de librerías estándar (`os`, `json`, `logging`).
  - Importación de librerías específicas (`hnswlib`, `sentence_transformers`).
  - Configuración del sistema de logs.
  - Carga de variables de entorno desde un archivo `.env`.

### **PASO 4: Cargar Documentos**
Carga documentos desde archivos o carpetas.

- **Líneas Clave:**
  - `load_documents`: Carga archivos de tipo `.txt`, `.json`, y `.pdf`.
  - `extract_content`: Procesa el contenido de los documentos dependiendo de su tipo.

### **PASO 5: Configurar la Clave API de Gemini**
- Obtiene la clave API desde las variables de entorno.
- Crea una instancia del modelo Gemini para uso posterior.

### **PASO 6: Configurar el Modelo de Embeddings**
- Se utiliza SentenceTransformer para generar embeddings de texto.
- `doc_enfermedad(pregunta)`: Determina el índice del documento más relevante 
  según la similitud con el nombre del archivo.

### **PASO 7: Crear Clases para Documentos e Índices**
- Clase `Document`: Representa un documento con su texto y metadatos.
- Clase `HNSWIndex`: Crea un índice para recuperación rápida de información mediante embeddings.

### **PASO 8: Procesar Documentos y Crear Índices**
- `desdobla_doc(data2)`: Crea objetos `Document` a partir del contenido.
- Para JSON con ensayos clínicos, crea un `Document` por ensayo.
- Para TXT/PDF, un `Document` genérico.
- Genera embeddings y construye el índice `HNSWlib`.

### **PASO 9: Traducir Preguntas y Respuestas**
- `traducir(texto, idioma_destino)`: Usa Gemini para traducir el texto solicitado.
- `generate_embedding(texto)`: Genera embeddings para la pregunta en inglés.

### **PASO 10: Generar Respuestas**
- `categorizar_pregunta(pregunta)`: Usa palabras clave para clasificar la pregunta (ej: 'ensayo', 'tratamiento').
- `generar_prompt(categoria, pregunta)`: Crea una instrucción específica según la categoría.
- `generar_respuesta(pregunta, contexto, prompt_especifico)`: 
  - Envía el prompt y el contexto a Gemini.
  - Traduce la respuesta al español.

### **PASO 11: Función Principal para Responder Preguntas**
- Utiliza caché para no recalcular respuestas idénticas.
- Integra categorización, obtención de contexto y generación de respuesta.

### **PASO 12: Interfaz CLI**
Ofrece una interfaz de línea de comando para interactuar con el chatbot.

- **Líneas Clave:**
  - Espera una pregunta del usuario.
  - Responde saludos.
  - Detecta si el usuario quiere salir.
  - Llama a `responder_pregunta()` con la información necesaria.