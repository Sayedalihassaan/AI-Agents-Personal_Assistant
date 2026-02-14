import os
import openai
import logging
from typing import List
from dotenv import load_dotenv
from backend.config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (TextLoader, PyPDFLoader, Docx2txtLoader)

# Load environment variables
_ = load_dotenv(override=True)
openai.api_key=OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def splitting_documents(documents: list[Document], chunk_size: int = 2000, chunk_overlap: int = 100) -> list[Document]:
    """
    Splits a list of documents into smaller sub-documents based on a specified character chunk size.

    Args:
        documents (list[Document]): The list of documents to split.
        chunk_size (int): The maximum number of characters in each sub-document. Defaults to 2000.
        chunk_overlap (int): The number of characters to keep the same between sub-documents. Defaults to 100.

    Returns:
        list[Document]: A list of the sub-documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)


def summarize_document(documents: List[Document]) -> str:
    """
    Summarizes a collection of documents using a language model.
    """
    llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0
    )

    
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.invoke(documents)
    return summary['output_text']


def loading_url(url: str=None) -> list[Document]:
    """
    Loads a document from a URL and returns it as a list of Document objects.

    Args:
        url (str): The URL of the document to load.

    Returns:
        list[Document]: A list of Document objects representing the loaded document.
    """
    logger.info(f"🌐 Loading document from URL: {url}")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"❌ Error loading document from URL {url}: {e}", exc_info=True)
        return []

def loading_audio(file_paths: List[str]) -> List[Document]:
    """
    Loads audio files, converts speech to text, and returns as Document objects.

    Supported formats: .mp3, .wav, .m4a

    Args:
        file_paths (List[str]): List of audio file paths

    Returns:
        List[Document]: List of transcribed documents
    """
    audio_docs = []

    for path in file_paths:
        filename = os.path.basename(path)
        logging.info(f"🎵 Processing audio file: {filename}")

        try:
            # Only support certain audio formats
            if not path.lower().endswith((".mp3", ".wav", ".m4a")):
                logging.warning(f"✖️ Unsupported audio format: {path}")
                continue

            # Use OpenAI Whisper for transcription
            with open(path, "rb") as audio_file:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            text = transcription.text
            audio_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": filename,
                        "Duration": transcription.usage.seconds
                    }
                )
            )

        except Exception as e:
            logging.error(f"❌ Error processing audio file {path}: {e}", exc_info=True)
            continue

    return audio_docs

def loading_documents(file_paths: List[str]) -> list[Document]:    
    """
    Loads a collection of documents from a list of file paths and returns them as a list of Document objects.

    Args:
        file_paths (List[str]): A list of file paths to load.

    Returns:
        list[Document]: A list of Document objects representing the loaded documents.

    Supported file formats are .txt, .pdf, .docx
    """
    full_document = []
    for path in file_paths:
        filename = os.path.basename(path)
        logger.info(f"📁 Processing file: {filename}")
        try:
            if path.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            elif path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                logger.warning(f"✖️ Unsupported file format: {path}")
                continue
            
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
            # secure_text, replacements = prepare_for_embedding(text)
            # print(replacements)
            # print(secure_text)
            full_document.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": filename,
                        "summary": summarize_document(documents) if documents else "Summary not available."
                    }
                )
            )
        
        except Exception as e:
            logger.error(f"❌ Error processing file {path}: {e}", exc_info=True)
            continue
    
    return full_document

def loading_data(file_paths: List[str] = None, url: str = None) -> list[Document]:
    full_doc = []
    if url:
        full_doc.extend(loading_url(url))
    if file_paths:
        # Split documents for text/pdf/docx
        text_doc_paths = [f for f in file_paths if f.lower().endswith((".txt", ".pdf", ".docx"))]
        if text_doc_paths:
            splitting_doc = splitting_documents(loading_documents(text_doc_paths))
            full_doc.extend(splitting_doc)

        # Process audio files separately
        audio_doc_paths = [f for f in file_paths if f.lower().endswith((".mp3", ".wav", ".m4a"))]
        if audio_doc_paths:
            full_doc.extend(loading_audio(audio_doc_paths))

    return full_doc