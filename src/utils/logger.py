import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Llamada a la función: {func.__name__} con argumentos: {args} y {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"Salida de la función: {func.__name__} con resultado: {result}")
        return result
    return wrapper
