from tqdm import tqdm
from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from flipkart.data_converter import DataConverter
from flipkart.config import Config


class DataIngestor:
    def __init__(self):
        # Local embeddings (fast + free, but needs RAM/CPU)
        self.embedding = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name=Config.ASTRA_DB_COLLECTION,
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE,
        )

    def ingest(self, load_existing: bool = True):
        """
        If load_existing=True, just returns the vector store.
        If False, loads documents and ingests into AstraDB with stable IDs.
        """

        if load_existing:
            print("Loading existing vector store (no ingestion).")
            return self.vstore

        print("Embedding model:", Config.EMBEDDING_MODEL)

        docs = DataConverter(Config.DATA_PATH).convert()

        if not docs:
            raise ValueError("No documents were loaded from the dataset.")

        print("Total docs loaded:", len(docs))
        print("Sample doc metadata:", docs[0].metadata)

        # ✅ Stable IDs (prevents duplicates forever)
        ids = []
        clean_docs = []

        for doc in docs:
            pid = doc.metadata.get("id")

            # Skip invalid IDs
            if pid is None:
                continue

            pid = str(pid).strip()
            if not pid:
                continue

            clean_docs.append(doc)
            ids.append(pid)

        print(f"Docs with valid IDs: {len(clean_docs)}")

        batch_size = 64
        for i in tqdm(range(0, len(clean_docs), batch_size), desc="Ingesting batches"):
            batch = clean_docs[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            # ✅ IMPORTANT: ids passed here prevents duplicates
            self.vstore.add_documents(batch, ids=batch_ids)

        print("Ingestion finished!")
        return self.vstore


if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.ingest(load_existing=False)