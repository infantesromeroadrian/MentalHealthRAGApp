import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from dotenv import load_dotenv
from src.utils.logger import log_function_call, logger

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

class PatientStatusClassifier:
    """
    Clasificador de estatus del paciente basado en el texto proporcionado.
    """

    def __init__(self, model_file: str, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Inicializa la clase cargando el modelo de clasificación y el modelo de embeddings.

        Args:
            model_file (str): Ruta al archivo del modelo entrenado.
            embedding_model_name (str): Nombre del modelo de embeddings a utilizar.
        """
        with open(model_file, 'rb') as f:
            self.model = joblib.load(f)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    @log_function_call
    def classify_status(self, text: str) -> int:
        """
        Clasifica el estatus del paciente en función del texto proporcionado.

        Args:
            text (str): Texto proporcionado por el paciente.

        Returns:
            int: Estatus clasificado (entero).
        """
        if not text:
            logger.warning("El texto proporcionado está vacío.")
            return -1  # Devuelve un valor predeterminado para texto vacío

        # Procesamiento del texto
        polarity = TextBlob(text).sentiment.polarity
        length = len(text.split())
        text_embedding = self.embedding_model.encode([text])

        # Convertir polarity y length en arrays bidimensionales
        polarity = np.array([[polarity]])
        length = np.array([[length]])

        # Concatenar las características (polarity, length) con los embeddings
        features = np.hstack((polarity, length, text_embedding))

        # Realizar la predicción
        status = self.model.predict(features)
        logger.info(f"Estatus clasificado: {status[0]} para el texto proporcionado.")
        return int(status[0])

