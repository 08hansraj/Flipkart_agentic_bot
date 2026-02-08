# Flipkart Agentic Chatbot (RAG + AstraDB + Groq Qwen)

A Flipkart-style product recommendation chatbot built using **RAG (Retrieval-Augmented Generation)** with:

* **HuggingFace embeddings** (`BAAI/bge-small-en-v1.5`)
* **AstraDB Vector Database** for semantic search
* **Qwen LLM via Groq** (`qwen3-32b`)
* **LangChain + LangGraph** for tool usage, memory, and agent behavior
* **Flask** frontend + backend (local deployment)
* **Prometheus + Grafana** monitoring setup

The bot retrieves relevant Flipkart products from the vector database and shows results as clean product cards (image, rating, price, review snippet, and Flipkart link).

---

## Project Structure

```
FLIPKART_AGENT_CHATBOT/
│
├── app.py                          # Flask app entrypoint
├── requirements.txt                # Python dependencies
├── README.md
├── setup.py
├── .env                            # Environment variables (not committed)
├── .gitignore
│
├── flipkart/                       # Core chatbot package
│   ├── __init__.py
│   ├── config.py                   # Config + env variables
│   ├── data_converter.py           # Converts dataset → LangChain Documents
│   ├── data_ingestion.py           # Embedding + AstraDB ingestion
│   └── rag_agent.py                # LangGraph agent + retriever tool
│
├── data/
│   ├── raw/                        # Original dataset (optional)
│   └── processed/
│       └── flipkart_products_prepared_25k.jsonl
│
├── frontend/
│   ├── templates/
│   │   └── index.html              # Chat UI
│   └── static/
│       └── style.css               # UI styling + product cards
│
├── scripts/
│   └── prepare_flipkart_dataset.py # Script used to clean/prepare dataset
│
├── prometheus/                     # Prometheus config (monitoring)
├── grafana/                        # Grafana dashboards (monitoring)
│
└── amznbot/                        # (Local venv folder name, not part of project code)
```

> Note: `amznbot` is the local virtual environment folder name (venv).
> It is not part of the application logic.

---

## How It Works (Architecture)

### 1) Data Ingestion (Offline)

* The Flipkart dataset is converted into LangChain `Document` objects.
* Each document contains:

  * `page_content`: embedding text (product description + keywords)
  * `metadata`: product_name, brand, category_path, price, rating, url, image, etc.
* HuggingFace embeddings are generated locally.
* Embeddings are stored in **AstraDB Vector Store**.

### 2) RAG Agent (Runtime)

* User asks a question (example: *"best office chair under 5000"*).
* LangGraph agent decides to call the retriever tool.
* The retriever fetches top products from AstraDB using semantic search.
* Tool returns formatted product cards.
* The assistant responds with those cards in the UI.

### 3) Memory

* Each user session is tracked using a `thread_id` stored in browser localStorage.
* LangGraph uses this thread_id to maintain conversation memory.
* Summarization middleware prevents memory from growing too large.

---

## Tech Stack

* **Backend:** Flask
* **RAG Framework:** LangChain + LangGraph
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **Vector Database:** AstraDB
* **LLM:** Qwen 3 32B via Groq
* **Monitoring:** Prometheus + Grafana

---

## Dependencies

This project uses the following key packages:

* `langchain==1.2.1`
* `langchain-astradb==1.0.0`
* `langchain-huggingface==1.2.0`
* `langchain-groq==1.1.1`
* `langchain-community==0.4.1`
* `datasets==4.4.2`
* `pypdf==6.5.0`
* `python-dotenv==1.2.1`
* `pandas==2.3.3`
* `flask==3.1.2`
* `prometheus_client==0.23.1`
* `setuptools==80.9.0`
* `streamlit==1.52.2`

---

## Setup Instructions (Local)

### Requirements

* **Python 3.10**
* **uv** (recommended) or pip

### 1) Clone the repo

```bash
git clone https://github.com/08hansraj/Flipkart_agentic_bot.git
cd Flipkart_agentic_bot
```

### 2) Create a virtual environment

Your local venv name can be anything. Example:

```bash
python -m venv amznbot
```

Activate:

**Windows (PowerShell)**

```bash
amznbot\Scripts\activate
```

**Mac/Linux**

```bash
source amznbot/bin/activate
```

---

## Install Dependencies (uv)

This project can be installed using **uv**.

### Option A: Install from requirements.txt

```bash
uv pip install -r requirements.txt
```

### Option B: Normal pip

```bash
pip install -r requirements.txt
```

---

## Environment Variables (.env)

Create a `.env` file in the project root:

```env
# AstraDB
ASTRA_DB_API_ENDPOINT=your_astra_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_astra_token
ASTRA_DB_KEYSPACE=your_keyspace
ASTRA_DB_COLLECTION=flipkart_database

# Groq
GROQ_API_KEY=your_groq_api_key

# Models
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RAG_MODEL=groq:qwen/qwen3-32b

# Dataset
DATA_PATH=data/processed/flipkart_products_prepared_25k.jsonl
```

---

## Step 1: Ingest Data into AstraDB (One-time)

⚠️ Important: If you accidentally ingest multiple times without stable IDs, you may get duplicates.

This project ingests with stable IDs (`metadata["id"]`) so duplicates do not happen after the fix.

### Run ingestion

```bash
python -m flipkart.data_ingestion
```

You should see output like:

* total docs loaded
* ingestion batches progress
* ingestion finished

---

## Step 2: Run the Flask App

Start the chatbot locally:

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

---

## API Endpoints

### `GET /`

Chat UI

### `POST /get`

Chat endpoint used by the frontend.

Payload:

* `msg` (user message)
* `thread_id` (stored in browser localStorage)

### `GET /health`

Health check endpoint.

### `GET /metrics`

Prometheus metrics endpoint.

---

## Monitoring (Prometheus + Grafana)

This repo includes Prometheus and Grafana folders.

### Metrics exposed:

* `http_requests_total`
* `model_predictions_total`
* `model_errors_total`

Once deployed, Prometheus can scrape `/metrics` and Grafana can visualize it.

---

## Common Issues & Fixes

### 1) Duplicates in AstraDB

If you see document count increasing unexpectedly (example: 19800 → 20400), it means ingestion ran multiple times.

Fix:

* Delete AstraDB collection `flipkart_database`
* Re-run ingestion using stable IDs

---

### 2) AstraDB connection errors

Make sure:

* ASTRA_DB_API_ENDPOINT is correct
* ASTRA_DB_APPLICATION_TOKEN is valid
* ASTRA_DB_KEYSPACE exists

---

### 3) Groq model not responding

Make sure:

* GROQ_API_KEY is set correctly
* `RAG_MODEL=groq:qwen/qwen3-32b`

---

### 4) Slow ingestion

HuggingFace embeddings run locally.
If CPU is slow, ingestion will take time.

---

## Deployment Plan (Upcoming)

This project is currently deployed locally using Flask.

Next planned deployment:

* Docker containerization
* Kubernetes deployment on GCP VM
* Public endpoint exposure via LoadBalancer / Ingress
* Persistent memory store (Redis) for LangGraph

---

## Future Improvements

* Add reranking for better product quality
* Add filters: budget, brand, rating, category
* Store conversation memory in Redis/Postgres instead of InMemorySaver
* Improve UI and mobile responsiveness
* Add streaming responses (SSE)

---

## Author

**Hansraj**
GitHub: [https://github.com/08hansraj](https://github.com/08hansraj)
