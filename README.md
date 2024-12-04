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