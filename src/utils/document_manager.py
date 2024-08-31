import os
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .logger import log_function_call, logger

class DocumentManager:
    """
    Gestiona la carga y procesamiento de documentos PDF, archivos CSV y textos simples desde subdirectorios específicos.
    """

    def __init__(self, base_directory: str):
        """
        Inicializa el DocumentManager con la ruta del directorio base.
        """
        self.base_directory = base_directory
        self.csv_directory = os.path.join(base_directory, 'csv_directory')
        self.pdf_directory = os.path.join(base_directory, 'pdf_directory')
        self.txt_directory = os.path.join(base_directory, 'txt_directory')

    @log_function_call
    def load_pdfs(self) -> List[Document]:
        """
        Carga todos los archivos PDF en el subdirectorio especificado.
        """
        pdf_docs = []
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                loader = PyMuPDFLoader(os.path.join(self.pdf_directory, filename))
                pdf_docs.extend(loader.load())
        return pdf_docs

    @log_function_call
    def load_texts(self) -> List[Document]:
        """
        Carga todos los archivos de texto (.txt) en el subdirectorio especificado.
        """
        text_docs = []
        for filename in os.listdir(self.txt_directory):
            if filename.endswith('.txt'):
                with open(os.path.join(self.txt_directory, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    text_docs.append(Document(page_content=text, metadata={"source": filename}))
        return text_docs

    @log_function_call
    def load_csv(self) -> pd.DataFrame:
        """
        Carga el archivo CSV en el subdirectorio especificado y lo devuelve como un DataFrame de pandas.
        """
        csv_files = [file for file in os.listdir(self.csv_directory) if file.endswith('.csv')]
        if not csv_files:
            logger.warning("No se encontraron archivos CSV en el directorio especificado.")
            return pd.DataFrame()  # Retorna un DataFrame vacío si no hay archivos CSV
        return pd.read_csv(os.path.join(self.csv_directory, csv_files[0]))

    @log_function_call
    def preview_csv(self, preview_lines: int = 5) -> pd.DataFrame:
        """
        Muestra una vista previa de las primeras líneas del CSV.
        """
        csv_data = self.load_csv()
        return csv_data.head(preview_lines)

    @log_function_call
    def process_pdfs(self, chunk_size: int = 500) -> List[Document]:
        """
        Procesa los documentos PDF cargados, dividiéndolos en fragmentos más pequeños.
        """
        pdf_docs = self.load_pdfs()
        return self.split_documents(pdf_docs, chunk_size)

    @log_function_call
    def process_texts(self, chunk_size: int = 500) -> List[Document]:
        """
        Procesa los documentos TXT cargados, dividiéndolos en fragmentos más pequeños.
        """
        text_docs = self.load_texts()
        return self.split_documents(text_docs, chunk_size)

    @log_function_call
    def split_documents(self, documents: List[Document], chunk_size: int = 500) -> List[Document]:
        """
        Divide los documentos en fragmentos más pequeños.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
        split_docs = []
        for doc in documents:
            split_texts = text_splitter.split_text(doc.page_content)
            for text in split_texts:
                split_docs.append(Document(page_content=text, metadata=doc.metadata))
        return split_docs
