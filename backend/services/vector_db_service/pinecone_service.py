import logging
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from backend.config import PINECONE_API_KEY, OPENAI_API_KEY

_ = load_dotenv(override=True)

shared_executor = ThreadPoolExecutor(max_workers=5)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_index(index_name: str, vect_length: int = 1536):
    """
    Create a Pinecone index with the specified name and vector length.

    Args:
        index_name (str): Name of the Pinecone index to create.
        vect_length (int, optional): Number of dimensions for the vector embeddings. Defaults to 1536.

    Returns:
        None
    """
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in [idx["name"] for idx in pinecone.list_indexes()]:
        logger.info(f'Creating Index: {index_name}')
        pinecone.create_index(
            name=index_name,
            dimension=vect_length,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        logger.info(f'✅ Done Creating Index: {index_name}')


def add_documents_to_pinecone(index_name: str, documents: List[Document], vect_length: int = 1536, batch_size: int = 20):
    """
    Add a list of documents to the specified Pinecone index.

    Args:
        index_name (str): Name of the Pinecone index to add the documents to.
        documents (List[Document]): List of documents to add to the index.
        vect_length (int, optional): Number of dimensions for the vector embeddings. Defaults to 1536.
        batch_size (int, optional): Number of documents to process in each batch. Defaults to 20.

    Returns:
        None
    """
    if not documents:
        logger.warning("⚠️ No valid documents to process.")
        return

    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    pinecone = Pinecone(api_key=PINECONE_API_KEY)

    if index_name not in [idx["name"] for idx in pinecone.list_indexes()]:
        create_index(index_name=index_name, vect_length=vect_length)

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model,
        pinecone_api_key=PINECONE_API_KEY
    )

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(documents=batch)
        logger.info(f"✅ Added batch {i//batch_size+1} ({len(batch)} docs) to Pinecone")
        
    logger.info("🎉 All documents added to Pinecone.")


async def delete_index_from_pinecone(index_name: str):
    """
    Delete the entire Pinecone index and remove all metadata from Postgres asynchronously.
    """
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        if index_name in [idx["name"] for idx in pinecone.list_indexes()]:
            logger.info(f"Deleting Pinecone Index: {index_name}")
            pinecone.delete_index(name=index_name)
            logger.info(f"✅ Pinecone Index {index_name} deleted")
        else:
            logger.info(f"Index {index_name} does not exist in Pinecone")

        # Recreate index
        create_index(index_name=index_name)

    except Exception as e:
        logger.error("❌ Error deleting index", exc_info=True)


async def delete_vectors_by_source(source_name: str, index_name: str, namespace: str = "__default__"):
    """
    Delete all vectors from the specified Pinecone index that match the given source name asynchronously.

    Args:
        source_name (str): Name of the source to delete vectors for.
        index_name (str): Name of the Pinecone index to delete vectors from.
        namespace (str, optional): Namespace to use in the Pinecone index. Defaults to "__default__".

    Raises:
        Exception: If an error occurs while deleting vectors from the index.

    Returns:
        None
    """
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        if index_name not in [idx["name"] for idx in pinecone.list_indexes()]:
            logger.warning(f"Index {index_name} does not exist")
            return

        index = pinecone.Index(index_name)
        all_ids = [i for ids in index.list(namespace=namespace) for i in ids]

        matching_ids = []
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            fetched = index.fetch(ids=batch_ids, namespace=namespace)
            for vec_id, data in fetched.vectors.items():
                if data.metadata.get("source") == source_name or data.metadata.get("filename") == source_name:
                    matching_ids.append(vec_id)

        if matching_ids:
            index.delete(ids=matching_ids, namespace=namespace)
            logger.info(f"💥 Deleted {len(matching_ids)} vectors from Pinecone with source '{source_name}'")
        else:
            logger.info("ℹ️ No vectors found to delete in Pinecone")
    except Exception as e:
        logger.error("❌ Error deleting vectors by source", exc_info=True)
        
        
async def delete_vectors_by_user(user_anon_id: str, index_name: str, namespace: str = "__default__"):
    """
    Delete all vectors whose metadata.user_id == user_anon_id from the given Pinecone index.
    """
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        if index_name not in [idx["name"] for idx in pinecone.list_indexes()]:
            logger.warning(f"Index {index_name} does not exist")
            return 0

        index = pinecone.Index(index_name)

        # gather all ids in the namespace
        # index.list returns a nested structure; handle depending on SDK version
        all_ids = [i for ids in index.list(namespace=namespace) for i in ids] if hasattr(index, "list") else []

        matching_ids = []
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            fetched = index.fetch(ids=batch_ids, namespace=namespace)
            # fetched might be an object with .vectors or a dict
            vectors = getattr(fetched, "vectors", None) or (fetched.get("vectors") if isinstance(fetched, dict) else None)
            if not vectors:
                continue

            for vec_id, vec_data in vectors.items():
                # vec_data shape depends on SDK: it may have .metadata or be dict with 'metadata'
                metadata = None
                if isinstance(vec_data, dict):
                    metadata = vec_data.get("metadata") or vec_data.get("meta")
                else:
                    # object with .metadata attribute
                    metadata = getattr(vec_data, "metadata", None)

                if metadata and metadata.get("user_id") == user_anon_id:
                    matching_ids.append(vec_id)

        if matching_ids:
            index.delete(ids=matching_ids, namespace=namespace)
            logger.info(f"Deleted {len(matching_ids)} vectors for user {user_anon_id}")
            return len(matching_ids)
        else:
            logger.info(f"No vectors found for user {user_anon_id}")
            return 0

    except Exception as e:
        logger.error("❌ Error deleting vectors by user", exc_info=True)
        return 0