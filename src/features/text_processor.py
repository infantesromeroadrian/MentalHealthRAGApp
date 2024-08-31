import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
from src.utils.logger import log_function_call, logger

# Descarga los recursos necesarios para NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TextProcessor:
    """
    Procesa texto utilizando técnicas de NLP, como limpieza, tokenización y lematización.
    """

    def __init__(self):
        """
        Inicializa la clase TextProcessor con las herramientas de NLP necesarias.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    @log_function_call
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto eliminando caracteres especiales, números y múltiples espacios.
        """
        if pd.isna(text):  # Verificar si el texto es NaN
            logger.warning("Texto recibido es NaN, devolviendo cadena vacía.")
            return ''

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @log_function_call
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza el texto en palabras individuales.
        """
        return word_tokenize(text)

    @log_function_call
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Elimina las stopwords (palabras vacías) de los tokens.
        """
        return [word for word in tokens if word not in self.stop_words]

    @log_function_call
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Aplica lematización a los tokens para reducirlos a su forma base.
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    @log_function_call
    def process(self, text: str) -> str:
        """
        Realiza todo el procesamiento de texto: limpieza, tokenización, eliminación de stopwords y lematización.
        """
        clean_text = self.clean_text(text)
        if clean_text == '':
            return ''  # Retornar cadena vacía si el texto es NaN o se queda vacío tras la limpieza

        tokens = self.tokenize(clean_text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)