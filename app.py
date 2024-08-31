import os
import streamlit as st
from src.model.patient_status_classifier import PatientStatusClassifier
from src.model.recommendation_generator import RecommendationGenerator
from src.utils.document_manager import DocumentManager
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

# Cargar variables de entorno
load_dotenv()

# Configuración del título y descripción de la app
st.title("Mental Health Assessment App")
st.write("Describe tu estado actual y completa el formulario para recibir una recomendación personalizada.")

# Selector de modo: Good Mode o Bad Mode
mode = st.selectbox("Seleccione el modo de operación:", ["Good Mode", "Bad Mode"])

# Mostrar advertencia si se selecciona "Bad Mode"
if mode == "Bad Mode":
    st.warning("Estás en Bad Mode. Las respuestas pueden ser inapropiadas o peligrosas. Use con precaución.")

# Recoger información del paciente
nombre = st.text_input("Nombre")
apellido = st.text_input("Apellido")
nombre_completo = f"{nombre} {apellido}".strip()

# Crear una instancia del DocumentManager
doc_manager = DocumentManager(base_directory='./data/raw_data')

# Verificar si el informe del paciente ya existe
informe_path = f"./data/raw_data/txt_directory/{nombre_completo.replace(' ', '')}Informe.txt"
guide_path = "./data/guide.txt"
informe_existente = os.path.exists(informe_path)

if informe_existente:
    st.write(f"Informe encontrado para {nombre_completo}.")
else:
    st.write(f"No se encontró un informe para {nombre_completo}. Se creará un nuevo informe.")

# Permitir al paciente adjuntar un informe en formato .txt
uploaded_file = st.file_uploader("Adjunta tu informe en formato .txt", type=["txt"])

if uploaded_file is not None:
    # Leer el contenido del archivo adjuntado
    report_content = uploaded_file.read().decode("utf-8")

    # Guardar el informe adjuntado en el directorio correspondiente
    with open(informe_path, 'w', encoding='utf-8') as file:
        file.write(report_content)
    st.success(f"Se ha cargado y guardado el informe adjuntado para {nombre_completo}.")
else:
    # Si no se adjuntó ningún archivo, continuar con el proceso habitual
    report_content = None

# Requerir input libre
input_text = st.text_area("Describe cómo te sientes hoy (requerido):")

# Formulario detallado para recoger información adicional (también requerido)
st.subheader("Por favor, completa el siguiente formulario (requerido):")
ansiedad = st.slider("Nivel de ansiedad (0-10):", 0, 10, 5)
insomnio = st.slider("Nivel de insomnio (0-10):", 0, 10, 5)
fatiga = st.slider("Nivel de fatiga (0-10):", 0, 10, 5)
estado_animo = st.selectbox("¿Cómo describirías tu estado de ánimo general?",
                            ["Muy bajo", "Bajo", "Neutral", "Alto", "Muy alto"])
apetito = st.selectbox("¿Cómo ha sido tu apetito últimamente?",
                       ["Muy bajo", "Bajo", "Normal", "Alto", "Muy alto"])
concentracion = st.selectbox("¿Cómo calificarías tu capacidad de concentración?",
                             ["Muy baja", "Baja", "Normal", "Alta", "Muy alta"])
actividad_fisica = st.slider(
    "¿Con qué frecuencia has realizado actividad física en la última semana? (días por semana)",
    0, 7, 3)
pensamientos_suicidas = st.selectbox("¿Has tenido pensamientos suicidas recientemente?",
                                     ["No", "Sí, ocasionalmente", "Sí, con frecuencia"])
apoyo_social = st.selectbox("¿Tienes un buen sistema de apoyo social (amigos, familia)?",
                            ["Sí", "No", "Parcialmente"])
dificultad_respirar = st.selectbox("¿Has experimentado dificultad para respirar en situaciones de ansiedad?",
                                   ["No", "Sí, ocasionalmente", "Sí, con frecuencia"])
ataques_panico = st.selectbox("¿Has experimentado ataques de pánico?",
                              ["No", "Sí, en el pasado", "Sí, recientemente"])

# Combinar las respuestas del formulario en un solo texto
form_data = (
    f"Ansiedad: {ansiedad}/10, Insomnio: {insomnio}/10, Fatiga: {fatiga}/10, "
    f"Estado de ánimo: {estado_animo}, Apetito: {apetito}, Concentración: {concentracion}, "
    f"Actividad física: {actividad_fisica} días por semana, "
    f"Pensamientos suicidas: {pensamientos_suicidas}, Apoyo social: {apoyo_social}, "
    f"Dificultad para respirar: {dificultad_respirar}, Ataques de pánico: {ataques_panico}"
)

# Botón para procesar el input
if st.button("Enviar"):
    if not input_text:
        st.warning("Por favor, describe cómo te sientes hoy.")
    elif not form_data:
        st.warning("Por favor, completa el formulario.")
    else:
        # Cargar el clasificador de estatus
        classifier = PatientStatusClassifier(
            model_file="./models/XGBoost_model.pkl",
            embedding_model_name='all-MiniLM-L6-v2'
        )
        status = classifier.classify_status(input_text)

        # Mapa del estatus a descripción legible
        status_mapping = {
            1: "Ligeramente Ansioso",
            2: "Moderadamente Ansioso",
            3: "Muy Ansioso",
            4: "Depresión Leve",
            5: "Depresión Moderada",
            6: "Depresión Severa"
        }
        status_description = status_mapping.get(status, "Estatus desconocido")

        # Configurar el modelo de embeddings y el retriever
        embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
        sample_document = Document(page_content="Este es un ejemplo de contenido", metadata={"source": "ejemplo"})
        vectorstore = Chroma.from_documents(documents=[sample_document], embedding=embedding_model)

        # Usar el LLM gdisney/phi-uncensored
        llm = Ollama(model="dolphin2.2-mistral", temperature=0.7)

        retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

        # Generar la recomendación usando la guía, el input actual del paciente y el modo seleccionado
        recommender = RecommendationGenerator(llm=llm, report_path=informe_path, retriever=retriever,
                                              guide_path=guide_path)

        # Generar la recomendación pasando el modo como parte del prompt
        recommendation = recommender.generate_recommendation(status, f"{input_text}\n{form_data}", mode=mode)

        # Remover posibles repeticiones en la recomendación
        recommendation = "\n".join(dict.fromkeys(recommendation.split("\n")))

        # Actualizar o crear el informe del paciente
        recommender.update_report(f"{input_text}\n{form_data}", recommendation)

        # Mostrar los resultados
        st.write(f"Estatus: {status_description}")
        st.write("Recomendación para el paciente:")
        st.write(recommendation)

        # Notificar al usuario si se creó un nuevo informe
        if not informe_existente:
            st.success(f"Se creó un nuevo informe para {nombre_completo}.")
        else:
            st.success(f"Se actualizó el informe para {nombre_completo}.")
