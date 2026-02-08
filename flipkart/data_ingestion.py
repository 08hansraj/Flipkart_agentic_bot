from tqdm import tqdm
from langchain_astradb import AstraDBVectorStore
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from flipkart.data_converter import DataConverter
from flipkart.config import Config

class DataIngestor:
    def __init__(self):
        # self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)
        self.embedding = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )


# load_existing - if docs already exits just load that 
   

    def ingest(self, load_existing=True):
        if load_existing:
            print("Loading existing vector store (no ingestion).")
            return self.vstore

        print("Embedding model:", Config.EMBEDDING_MODEL)

        docs = DataConverter(
            "data/processed/flipkart_products_prepared_25k.jsonl"
        ).convert()
        docs = DataConverter("data/processed/flipkart_products_prepared_25k.jsonl").convert()

        # ids = [doc.metadata["id"] for doc in docs]

        # self.vstore.add_documents(docs, ids=ids)

        print("Total docs loaded:", len(docs))
        print("Sample doc metadata:", docs[0].metadata if docs else "No docs")

        batch_size = 64
        for i in tqdm(range(0, len(docs), batch_size), desc="Ingesting batches"):
            batch = docs[i:i+batch_size]
            self.vstore.add_documents(batch)

        print("Ingestion finished!")
        return self.vstore
        
if __name__=="__main__":
    ingestor = DataIngestor()
    ingestor.ingest(load_existing=False)