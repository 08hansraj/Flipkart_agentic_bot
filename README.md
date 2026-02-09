# Flipkart Agentic Chatbot (RAG + AstraDB + Groq Qwen)

A Flipkart-style product recommendation chatbot built using **RAG (Retrieval-Augmented Generation)** with:

* **HuggingFace embeddings** (`BAAI/bge-small-en-v1.5`)
* **AstraDB Vector Database** for semantic search
* **Qwen LLM via Groq** (`qwen3-32b`)
* **LangChain + LangGraph** for tool usage, memory, and agent behavior
* **Flask** frontend + backend (local deployment)
* **Prometheus + Grafana** monitoring setup

The bot retrieves relevant Flipkart products from the vector database and displays them as clean product cards (image, price, rating, description, and Flipkart link).

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
│   ├── data_ingestion.py           # Embedding + AstraDB ingestion logic
│   └── rag_agent.py                # LangGraph agent + retriever tool (JSON output)
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
│   ├── prepare_flipkart_dataset.py # Script used to clean/prepare dataset
│   └── ingest_flipkart.py          # One-time ingestion runner
│
├── prometheus/                     # Prometheus config (monitoring)
├── grafana/                        # Grafana dashboards (monitoring)
│
└── amznbot/                        # Local venv folder name (not part of code)
```

> Note: `amznbot` is only the local virtual environment folder name (venv).

---

## How It Works (Architecture)

### 1) Data Ingestion (Offline / One-time)

* The Flipkart dataset is converted into LangChain `Document` objects.
* Each document contains:

  * `page_content`: compact high-signal embedding text
  * `metadata`: product_name, brand, category_path, price, rating, url, image, etc.
* HuggingFace embeddings are generated locally.
* Vectors are stored in **AstraDB Vector Store**.

---

### 2) RAG Agent (Runtime)

* User asks a question (example: **"men tshirt under 500"**).
* LangGraph agent always calls the retriever tool for product intent.
* Retriever performs **MMR semantic search** in AstraDB.
* Results are lightly reranked for better stability.
* Tool returns a structured JSON payload:

```json
{
  "reply": "Here are the best matches I found:",
  "products": [
    {
      "title": "...",
      "brand": "...",
      "discounted_price": 549,
      "retail_price": 999,
      "image": "http://...",
      "url": "http://..."
    }
  ]
}
```

* Flask frontend renders product cards using HTML + CSS.

---

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

## Setup Instructions (Local)

### Requirements

* **Python 3.10**
* **uv** (recommended) or pip

---

## 1) Clone the repo

```bash
git clone https://github.com/08hansraj/Flipkart_agentic_bot.git
cd Flipkart_agentic_bot
```

---

## 2) Create a virtual environment

Example:

```bash
python -m venv amznbot
```

Activate:

### Windows (PowerShell)

```bash
amznbot\Scripts\activate
```

### Mac/Linux

```bash
source amznbot/bin/activate
```

---

## 3) Install dependencies

### Using uv

```bash
uv pip install -r requirements.txt
```

### Using pip

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
ASTRA_DB_KEYSPACE=default_keyspace
ASTRA_DB_COLLECTION=flipkart_products_v2

# HuggingFace
HF_TOKEN=your_hf_token

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

⚠️ Important:
Ingestion should be run **only once**.
The project uses stable IDs (`metadata["id"]`) so duplicates do not occur.

Run ingestion:

```bash
python scripts/ingest_flipkart.py
```

You should see:

* Total docs loaded
* Batch ingestion progress
* Ingestion finished

---

## Step 2: Run the Flask App

Start the chatbot locally:

```bash
python app.py
```

Open in browser:

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

Response:

```json
{
  "reply": "...",
  "products": [...],
  "thread_id": "..."
}
```

### `GET /health`

Health check endpoint.

### `GET /metrics`

Prometheus metrics endpoint.

---

## Monitoring (Prometheus + Grafana)

This repo includes Prometheus and Grafana folders.

Metrics exposed:

* `http_requests_total`
* `model_predictions_total`
* `model_errors_total`

Prometheus can scrape `/metrics` and Grafana can visualize it.

---

## Common Issues & Fixes

### 1) AstraDB record count keeps increasing

Cause: ingestion was run multiple times.

Fix:

* Use a new AstraDB collection (recommended):

  * Example: `flipkart_products_v2`
* Or delete the old collection and ingest once.

---

### 2) Groq 429 Rate Limit Errors

Cause: too many tokens per minute.

Fix:

* This project uses JSON tool output (instead of HTML) to reduce tokens.
* If you still hit limits, reduce:

  * number of retrieved docs
  * number of returned products

---

### 3) AstraDB connection errors

Make sure:

* ASTRA_DB_API_ENDPOINT is correct
* ASTRA_DB_APPLICATION_TOKEN is valid
* ASTRA_DB_KEYSPACE exists

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

* Add stronger reranking for better product quality
* Add filters: budget, brand, rating, category
* Add hybrid retrieval (BM25 + embeddings)
* Store conversation memory in Redis/Postgres instead of InMemorySaver
* Improve UI and mobile responsiveness
* Add streaming responses (SSE)

---

## Author

**Hansraj**
GitHub: [https://github.com/08hansraj](https://github.com/08hansraj)
