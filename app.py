from flask import Flask, render_template, request, Response, jsonify
from prometheus_client import Counter, generate_latest
from dotenv import load_dotenv
import uuid

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_agent import RAGAgentBuilder

# Load environment variables
load_dotenv()

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")
PREDICTION_COUNT = Counter("model_predictions_total", "Total Model Predictions")
ERROR_COUNT = Counter("model_errors_total", "Total Model Errors")


def create_app():
    app = Flask(
        __name__,
        template_folder="frontend/templates",
        static_folder="frontend/static"
    )

    # Load vector store (already ingested)
    try:
        vector_store = DataIngestor().ingest(load_existing=True)
    except Exception as e:
        print("❌ Failed to load vector store:", str(e))
        vector_store = None

    # Build RAG Agent
    rag_agent = None
    if vector_store is not None:
        try:
            rag_agent = RAGAgentBuilder(vector_store).build_agent()
        except Exception as e:
            print("❌ Failed to build RAG agent:", str(e))
            rag_agent = None

    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")

    @app.route("/get", methods=["POST"])
    def get_response():
        REQUEST_COUNT.inc()

        if rag_agent is None:
            return "Bot is not ready. Please check server logs and AstraDB connection."

        user_input = request.form.get("msg", "").strip()
        thread_id = request.form.get("thread_id", "").strip()

        if not user_input:
            return "Please type something."

        # ✅ Generate a thread_id if missing (prevents shared memory across users)
        if not thread_id:
            thread_id = str(uuid.uuid4())

        try:
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

            final_msg = response["messages"][-1]
            final_text = getattr(final_msg, "content", None)

            if not final_text:
                return "Sorry, I couldn't generate a valid response."

            # Optional: attach thread_id so frontend can store it
            # But your frontend likely already stores it in localStorage.
            return final_text

        except Exception as e:
            ERROR_COUNT.inc()
            print("❌ Agent invoke error:", str(e))
            return "Something went wrong while generating the response. Please try again."

    @app.route("/health")
    def health():
        if vector_store is None or rag_agent is None:
            return jsonify({"status": "unhealthy"}), 500
        return jsonify({"status": "healthy"}), 200

    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)