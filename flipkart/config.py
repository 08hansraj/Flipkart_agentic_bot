import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # =========================
    # AstraDB
    # =========================
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "flipkart_database")

    # =========================
    # LLM Provider (Groq)
    # =========================
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # =========================
    # Embeddings + RAG Model
    # =========================
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RAG_MODEL = os.getenv("RAG_MODEL", "groq:qwen/qwen3-32b")

    # =========================
    # Dataset Path (for ingestion)
    # =========================
    DATA_PATH = os.getenv(
        "DATA_PATH",
        "data/processed/flipkart_products_prepared_25k.jsonl"
    )