from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from src.utils.logger import log_function_call, logger

class FeatureExtractor:
    """
    Clase para extraer, procesar y balancear características de un conjunto de datos de texto procesado.
    """

    def __init__(self, dataframe: pd.DataFrame, balance_method: str = None):
        """
        Inicializa la clase con un DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame que contiene las columnas 'status' y 'processed_statement'.
            balance_method (str): Método para balancear las clases ('smote', 'undersample', 'smoteenn').
        """
        self.dataframe = dataframe
        self.balance_method = balance_method

    @log_function_call
    def calculate_polarity(self, text: str) -> float:
        """
        Calcula la polaridad de un texto usando TextBlob.

        Args:
            text (str): Texto procesado.

        Returns:
            float: Polaridad del texto. Si el texto es NaN, devuelve 0.0.
        """
        if isinstance(text, float) and np.isnan(text):
            logger.warning("Texto recibido es NaN, devolviendo 0.0 como polaridad.")
            return 0.0
        return TextBlob(text).sentiment.polarity

    @log_function_call
    def calculate_length(self, text: str) -> int:
        """
        Calcula la longitud de un texto en número de palabras.

        Args:
            text (str): Texto procesado.

        Returns:
            int: Número de palabras en el texto. Si el texto es NaN, devuelve 0.
        """
        if isinstance(text, float) and np.isnan(text):
            logger.warning("Texto recibido es NaN, devolviendo 0 como longitud.")
            return 0
        return len(text.split())

    @log_function_call
    def extract_features(self) -> pd.DataFrame:
        """
        Extrae características del DataFrame y añade columnas correspondientes.

        Returns:
            pd.DataFrame: DataFrame con columnas adicionales para las características.
        """
        # Calcular la polaridad y la longitud de cada 'processed_statement'
        self.dataframe['polarity'] = self.dataframe['processed_statement'].apply(self.calculate_polarity)
        self.dataframe['length'] = self.dataframe['processed_statement'].apply(self.calculate_length)

        return self.dataframe

    @log_function_call
    def balance_classes(self):
        """
        Balancea las clases en el DataFrame usando el método especificado.

        Returns:
            pd.DataFrame: DataFrame con las clases balanceadas.
        """
        X = self.dataframe[['polarity', 'length']].values
        y = self.dataframe['status_encoded']

        if self.balance_method == 'smote':
            balancer = SMOTE()
        elif self.balance_method == 'undersample':
            balancer = RandomUnderSampler()
        elif self.balance_method == 'smoteenn':
            balancer = SMOTEENN()
        else:
            logger.info("No se especificó un método de balanceo, devolviendo el DataFrame original.")
            return self.dataframe  # Si no se especifica un método, devolver el DataFrame original

        X_balanced, y_balanced = balancer.fit_resample(X, y)
        balanced_df = pd.DataFrame(X_balanced, columns=['polarity', 'length'])
        balanced_df['status_encoded'] = y_balanced

        # Mantener solo las columnas balanceadas y reemplazar en el dataframe original
        self.dataframe = pd.concat([balanced_df, self.dataframe.drop(columns=['polarity', 'length', 'status_encoded'])],
                                   axis=1)
        logger.info("Clases balanceadas correctamente.")
        return self.dataframe

    @log_function_call
    def scale_features(self) -> pd.DataFrame:
        """
        Escala las columnas numéricas del DataFrame.

        Returns:
            pd.DataFrame: DataFrame con las columnas numéricas escaladas.
        """
        scaler = StandardScaler()
        self.dataframe[['polarity', 'length']] = scaler.fit_transform(self.dataframe[['polarity', 'length']])
        logger.info("Características numéricas escaladas correctamente.")
        return self.dataframe

    @log_function_call
    def encode_target(self, method: str = 'label') -> pd.DataFrame:
        """
        Codifica la columna target 'status' en etiquetas.

        Args:
            method (str): Método de codificación ('label' para Label Encoding, 'onehot' para One-Hot Encoding).

        Returns:
            pd.DataFrame: DataFrame con la columna 'status' codificada.
        """
        if method == 'label':
            encoder = LabelEncoder()
            self.dataframe['status_encoded'] = encoder.fit_transform(self.dataframe['status'])
            logger.info("Codificación Label Encoding aplicada correctamente.")
        elif method == 'onehot':
            encoder = OneHotEncoder(sparse=False)
            one_hot_encoded = encoder.fit_transform(self.dataframe[['status']])
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['status']))
            self.dataframe = pd.concat([self.dataframe, one_hot_df], axis=1)
            logger.info("Codificación One-Hot Encoding aplicada correctamente.")
        else:
            logger.error("Método de codificación no soportado. Use 'label' o 'onehot'.")
            raise ValueError("Método de codificación no soportado. Use 'label' o 'onehot'.")

        return self.dataframe

    @log_function_call
    def save_features(self, output_file: str):
        """
        Guarda el DataFrame con las características extraídas en un archivo CSV.

        Args:
            output_file (str): Ruta del archivo donde se guardará el DataFrame.
        """
        self.dataframe.to_csv(output_file, index=False)
        logger.info(f"Características guardadas en: {output_file}")
