from langchain_community.embeddings import OpenAIEmbeddings
import pandas as pd
from src.utils.logger import log_function_call, logger

class EmbeddingGenerator:
    """
    Clase para generar embeddings de la columna 'processed_statement' en un DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, api_key: str):
        """
        Inicializa la clase con un DataFrame y el modelo de OpenAI para generar embeddings.

        Args:
            dataframe (pd.DataFrame): DataFrame que contiene la columna 'processed_statement'.
            api_key (str): Clave API para acceder al modelo de OpenAI.
        """
        self.dataframe = dataframe
        self.model = OpenAIEmbeddings(openai_api_key=api_key)

    @log_function_call
    def clean_statements(self):
        """
        Asegura que todos los valores en 'processed_statement' sean cadenas válidas.
        """
        self.dataframe['processed_statement'] = self.dataframe['processed_statement'].astype(str).fillna('')
        logger.info("Se han limpiado las declaraciones en la columna 'processed_statement'.")

    @log_function_call
    def generate_embeddings(self) -> pd.DataFrame:
        """
        Genera embeddings para cada fila en la columna 'processed_statement'.

        Returns:
            pd.DataFrame: DataFrame con una nueva columna 'embeddings' que contiene los embeddings generados.
        """
        self.clean_statements()  # Asegurarse de que todas las declaraciones sean cadenas
        logger.info("Generando embeddings para las declaraciones procesadas.")
        embeddings = self.model.embed_documents(self.dataframe['processed_statement'].tolist())
        self.dataframe['embeddings'] = embeddings
        logger.info("Embeddings generados y añadidos al DataFrame.")
        return self.dataframe

    @log_function_call
    def save_embeddings(self, output_file: str):
        """
        Guarda el DataFrame con los embeddings generados en un archivo CSV.

        Args:
            output_file (str): Ruta del archivo donde se guardará el DataFrame.
        """
        # Para guardar los embeddings como una columna de listas, puedes utilizar el formato pickle de pandas
        self.dataframe.to_pickle(output_file)
        logger.info(f"Embeddings guardados en: {output_file}")
