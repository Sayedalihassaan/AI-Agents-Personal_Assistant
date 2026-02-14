from langchain_community.tools import Tool
from langchain_pinecone import PineconeVectorStore
from backend.config import OPENAI_API_KEY, PINECONE_API_KEY
from langchain_openai.embeddings import OpenAIEmbeddings

def pinecone_retriever_tool(index_name: str, top_k: int = 5) -> Tool:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Wrapper function (Tools require simple string → output)
    def retrieve(query: str):
        docs = retriever.invoke(query)   # <-- correct way in LangChain 0.1+
        return [d.page_content for d in docs]

    return Tool(
        name="pinecone_retriever",
        description="Retrieves relevant documents from Pinecone vector database.",
        func=retrieve
    )