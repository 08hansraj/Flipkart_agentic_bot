from flask import Flask, render_template, request, Response, jsonify
from prometheus_client import Counter, generate_latest
from dotenv import load_dotenv
import uuid
import json

from flipkart.data_ingestion import DataIngestor
from flipkart.rag_agent import RAGAgentBuilder

load_dotenv()

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests")
PREDICTION_COUNT = Counter("model_predictions_total", "Total Model Predictions")
ERROR_COUNT = Counter("model_errors_total", "Total Model Errors")


def create_app():
    app = Flask(
        __name__,
        template_folder="frontend/templates",
        static_folder="frontend/static",
    )

    vector_store = None
    rag_agent = None

    try:
        print("🔹 Loading AstraDB vector store (no ingestion)...")
        vector_store = DataIngestor().ingest(load_existing=True)
        print("✅ Vector store loaded.")
    except Exception as e:
        print("❌ Failed to load vector store:", str(e))
        vector_store = None

    if vector_store is not None:
        try:
            print("🔹 Building RAG agent...")
            rag_agent = RAGAgentBuilder(vector_store).build_agent()
            print("✅ RAG agent ready.")
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
            return jsonify({"reply": "Bot is not ready. Check server logs.", "products": []}), 500

        user_input = request.form.get("msg", "").strip()
        thread_id = request.form.get("thread_id", "").strip()

        if not user_input:
            return jsonify({"reply": "Please type something.", "products": []})

        if not thread_id:
            thread_id = str(uuid.uuid4())

        try:
            response = rag_agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": thread_id}},
            )

            PREDICTION_COUNT.inc()

            if not response.get("messages"):
                return jsonify({"reply": "Sorry, no response generated.", "products": []})

            final_msg = response["messages"][-1]
            final_text = getattr(final_msg, "content", "")

            try:
                payload = json.loads(final_text)
            except Exception:
                payload = {"reply": final_text, "products": []}

            payload["thread_id"] = thread_id
            return jsonify(payload)

        except Exception as e:
            ERROR_COUNT.inc()
            print("❌ Agent invoke error:", str(e))
            return jsonify({"reply": "Something went wrong. Please try again.", "products": []}), 500

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
    app.run(host="0.0.0.0", port=5000, debug=False)