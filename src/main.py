import os
from dotenv import load_dotenv
from src.model.patient_status_classifier import PatientStatusClassifier
from src.model.recommendation_generator import RecommendationGenerator
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Inicializar el clasificador de estatus del paciente
classifier = PatientStatusClassifier(
    model_file="../models/XGBoost_model.pkl",
    embedding_model_name='all-MiniLM-L6-v2'
)

# Simular la entrada del paciente
patient_text = "Me siento muy ansioso y no puedo dormir."
status = classifier.classify_status(patient_text)

# Mapear el estatus a una descripción legible
status_mapping = {
    1: "Ligeramente Ansioso",
    2: "Moderadamente Ansioso",
    3: "Muy Ansioso",
    4: "Depresión Leve",
    5: "Depresión Moderada",
    6: "Depresión Severa"
}
status_description = status_mapping.get(status, "Estatus desconocido")

# Configurar el modelo de embeddings de OpenAI
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

# Crear un documento de ejemplo (debes reemplazar esto con documentos reales)
sample_document = Document(page_content="Este es un ejemplo de contenido", metadata={"source": "ejemplo"})

# Crear el vectorstore usando la función from_documents
vectorstore = Chroma.from_documents(documents=[sample_document], embedding=embedding_model)

# Configurar el retriever usando el vectorstore
llm = Ollama(model="gdisney/phi-uncensored", temperature=0.7)  # LLM de Ollama
retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Inicializar el generador de recomendaciones
report_path = "../data/raw_data/txt_directory/LauraGInforme.txt"
guide_path = "../data/guide.txt"  # Ruta a la guía
recommender = RecommendationGenerator(llm=llm, report_path=report_path, retriever=retriever, guide_path=guide_path)

# Seleccionar el modo: "Good Mode" o "Bad Mode"
mode = "Good Mode"  # Puedes cambiarlo a "Bad Mode" para ver cómo afecta la recomendación

# Generar la recomendación basada en la descripción del estatus, el informe y el modo seleccionado
recommendation = recommender.generate_recommendation(status, patient_text, mode)

# Actualizar el informe del paciente con la recomendación
recommender.update_report(patient_text, recommendation)

print(f"Estatus: {status_description}")
print(f"Recomendación para el paciente: {recommendation}")
print("Informe actualizado con la recomendación.")