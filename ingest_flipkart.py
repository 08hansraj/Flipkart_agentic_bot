from flipkart.data_ingestion import DataIngestor

if __name__ == "__main__":
    DataIngestor().ingest(load_existing=False)