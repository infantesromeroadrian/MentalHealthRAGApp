# 🧠 Mental Health Assessment App

Bienvenido a **Mental Health Assessment App** 🩺, una herramienta diseñada para evaluar el estado mental de los pacientes y proporcionar recomendaciones personalizadas basadas en su estado actual y su historial clínico. Esta aplicación utiliza modelos de aprendizaje automático y procesamiento de lenguaje natural para analizar la información proporcionada por el paciente y generar recomendaciones adecuadas.

## 🚀 Características

- **Clasificación de Estado del Paciente**: Analiza el texto proporcionado por el paciente para determinar su nivel de ansiedad, insomnio, y otros factores importantes.
- **Generación de Recomendaciones**: Basado en el estado del paciente y un informe clínico previo, la app sugiere tratamientos o pasos a seguir.
- **Modos de Operación**:
  - **Good Mode**: Proporciona recomendaciones éticas y profesionales.
  - **Bad Mode**: Ofrece recomendaciones no convencionales, sin restricciones éticas (con advertencia).
- **Carga de Informes**: Los pacientes pueden adjuntar sus informes en formato .txt para un análisis más completo.
- **Formulario Detallado**: Los pacientes completan un formulario detallado para ofrecer una visión más clara de su estado actual.

## 🛠️ Instalación

Sigue estos pasos para configurar y ejecutar la aplicación en tu entorno local.

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/mental-health-assessment-app.git
cd mental-health-assessment-app
```

## 2. Configurar el entorno virtual
Asegúrate de tener `poetry` instalado. Luego, crea un entorno virtual e instala las dependencias:

```bash
poetry install
```

### 3. Configurar variables de entorno

Crea un archivo .env en el directorio raíz y añade tus claves API necesarias:
OPENAI_API_KEY=tu_clave_de_openai


```bash
OPENAI_API_KEY=tu_clave_de_openai
```

4. Ejecutar la aplicación

```bash
streamlit run app.py
```

## 4. Estructura del proyecto

mental-health-assessment-app/
│
├── data/                   # Directorio para datos crudos y procesados
│   ├── raw_data/           # Datos originales de entrada
│   ├── processed_data/     # Datos procesados para entrenamiento y análisis
│   └── guide.txt           # Guía para psicólogos
│
├── models/                 # Modelos entrenados
│   ├── XGBoost_model.pkl
│   └── vectorizer.pkl
│
├── src/                    # Código fuente de la aplicación
│   ├── model/              # Modelos de ML y generador de recomendaciones
│   ├── features/           # Extracción de características y generación de embeddings
│   └── utils/              # Utilidades como el gestor de documentos y logger
│
├── notebooks/              # Notebooks para experimentación y análisis
│
├── app.py                  # Script principal para ejecutar la aplicación
├── README.md               # Archivo que estás leyendo ahora
└── .env.example            # Ejemplo de archivo .env


## 🧑‍💻 Uso

Ingresar Información

- Describe cómo te sientes: Escribe un breve resumen de tu estado actual.
- Completa el formulario: Proporciona más detalles sobre tu estado mental y físico.
- Adjunta tu informe: Si tienes un informe clínico previo, puedes adjuntarlo en formato .txt.

Seleccionar Modo

- Good Mode: Recibe recomendaciones éticas.
- Bad Mode: Recibe recomendaciones sin restricciones éticas (⚠️ Úselo bajo su propio riesgo).
- Generar Recomendación

Haz clic en "Enviar" y la aplicación te proporcionará una recomendación basada en tu estado actual y tu historial clínico.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.