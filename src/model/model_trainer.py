import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.utils.logger import log_function_call, logger

class ModelTrainer:
    """
    Clase para gestionar el pipeline de machine learning: desde la preparación de datos
    hasta el entrenamiento, búsqueda de hiperparámetros y evaluación del modelo.
    """

    def __init__(self, dataframe, target_column='status_encoded', feature_columns=None):
        """
        Inicializa la clase con el DataFrame y las columnas relevantes.

        Args:
            dataframe (pd.DataFrame): DataFrame con los datos para el entrenamiento.
            target_column (str): Nombre de la columna objetivo (etiquetas).
            feature_columns (list): Lista de nombres de las columnas con las características.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.feature_columns = feature_columns or ['polarity', 'length', 'embeddings']
        self.model = None
        self.best_params_ = None

    @log_function_call
    def prepare_features(self):
        """
        Prepara las características para el entrenamiento.

        Returns:
            np.ndarray: Matriz de características.
        """
        features = []

        # Si los embeddings están en la lista de columnas de características, extraer y combinar
        if 'embeddings' in self.feature_columns:
            embeddings = np.array(self.dataframe['embeddings'].tolist())
            features.append(embeddings)

        # Extraer las otras características numéricas
        for col in self.feature_columns:
            if col != 'embeddings':
                feature_col = self.dataframe[col].values.reshape(-1, 1)
                features.append(feature_col)

        # Concatenar todas las características en una sola matriz
        X = np.hstack(features)
        return X

    @log_function_call
    def split_data(self, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        Args:
            test_size (float): Proporción del conjunto de datos que se usará como conjunto de prueba.
            random_state (int): Semilla para asegurar la reproducibilidad.

        Returns:
            X_train, X_test, y_train, y_test: Conjuntos de entrenamiento y prueba.
        """
        X = self.prepare_features()
        y = self.dataframe[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        logger.info(f'Datos divididos: {len(X_train)} en entrenamiento y {len(X_test)} en prueba.')
        return X_train, X_test, y_train, y_test

    @log_function_call
    def train_model(self, X_train, y_train, model_type='RandomForest', param_grid=None, cv=5):
        """
        Entrena el modelo seleccionado con Grid Search y los datos de entrenamiento.

        Args:
            X_train: Características de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            model_type (str): Tipo de modelo a entrenar ('RandomForest', 'XGBoost').
            param_grid (dict): Diccionario con los parámetros para el Grid Search.
            cv (int): Número de particiones para la validación cruzada.

        Returns:
            model: Mejor modelo entrenado.
        """
        if model_type == 'RandomForest':
            model = RandomForestClassifier()
            param_grid = param_grid or {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'XGBoost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            param_grid = param_grid or {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            raise ValueError(f'Modelo {model_type} no soportado.')

        grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        logger.info(f'Modelo entrenado con éxito. Mejores parámetros: {self.best_params_}')
        return self.model

    @log_function_call
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo entrenado utilizando el conjunto de prueba.

        Args:
            X_test: Características del conjunto de prueba.
            y_test: Etiquetas del conjunto de prueba.

        Returns:
            dict: Métricas de evaluación del modelo.
        """
        if self.model is None:
            raise ValueError('El modelo no ha sido entrenado aún.')

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f'Accuracy: {accuracy}')
        logger.info(f'Informe de clasificación: {report}')

        return {"accuracy": accuracy, "classification_report": report}

    def get_best_params(self):
        """
        Devuelve los mejores parámetros encontrados por Grid Search.

        Returns:
            dict: Mejores parámetros.
        """
        return self.best_params_
