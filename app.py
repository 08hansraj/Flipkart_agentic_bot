from flask import Flask, render_template, request, Response, jsonify
from prometheus_client import Counter, generate_latest

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_agent import RAGAgentBuilder

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")
PREDICTION_COUNT = Counter("model_predictions_total", "Total Model Predictions")


def create_app():
    app = Flask(
        __name__,
        template_folder="frontend/templates",
        static_folder="frontend/static"
    )

    # ✅ Load vector store (already ingested)
    vector_store = DataIngestor().ingest(load_existing=True)

    # ✅ Build RAG Agent
    rag_agent = RAGAgentBuilder(vector_store).build_agent()

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")

    @app.route("/get", methods=["POST"])
    def get_response():
        REQUEST_COUNT.inc()

        user_input = request.form.get("msg", "").strip()
        thread_id = request.form.get("thread_id", "").strip()

        if not user_input:
            return "Please type something."

        # thread_id MUST come from browser localStorage
        # If missing, fallback to a simple fixed thread (not ideal but safe)
        if not thread_id:
            thread_id = "default_thread"

        response = rag_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        PREDICTION_COUNT.inc()

        if not response.get("messages"):
            return "Sorry, I couldn't find relevant product information."

        return response["messages"][-1].content

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy"}), 200

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
    