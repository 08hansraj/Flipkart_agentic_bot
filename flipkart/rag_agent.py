from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

from flipkart.config import Config


def build_flipkart_retriever_tool(retriever):
    @tool
    def flipkart_retriever_tool(query: str) -> str:
        """
        Retrieve top products related to the user query.
        Returns formatted product cards with image, rating, price, and link.
        """
        docs = retriever.invoke(query)

        # If no docs found
        if not docs:
            return """
I couldn’t find an exact match for that.

Can you tell me:
- your budget range, and
- what you’ll use it for?

For example, I can recommend:
• Office chairs  
• Gaming chairs  
• Ergonomic chairs
"""

        formatted = []

        for d in docs:
            m = d.metadata or {}

            title = m.get("product_name") or "Unknown Product"
            brand = m.get("brand") or "N/A"

            # Ratings
            product_rating = m.get("product_rating")
            overall_rating = m.get("overall_rating")

            # Prices
            retail_price = m.get("retail_price")
            discounted_price = m.get("discounted_price")

            # URL
            url = m.get("product_url") or ""

            # Image
            image = m.get("image") or ""
            first_image = ""

            # If stored as string like ["url1","url2"]
            if isinstance(image, str) and image.strip().startswith("["):
                try:
                    import json
                    imgs = json.loads(image)
                    if isinstance(imgs, list) and len(imgs) > 0:
                        first_image = imgs[0]
                except Exception:
                    first_image = ""
            elif isinstance(image, str):
                first_image = image

            # Clean rating display
            def safe_rating(x):
                if x is None:
                    return "N/A"
                try:
                    if str(x).lower() == "nan":
                        return "N/A"
                    return str(round(float(x), 1))
                except Exception:
                    return str(x)

            product_rating = safe_rating(product_rating)
            overall_rating = safe_rating(overall_rating)

            # Clean prices
            def safe_price(x):
                if x is None:
                    return "N/A"
                try:
                    if str(x).lower() == "nan":
                        return "N/A"
                    return str(int(float(x)))
                except Exception:
                    return str(x)

            retail_price = safe_price(retail_price)
            discounted_price = safe_price(discounted_price)

            # Review snippet from text
            review_snippet = d.page_content.replace("\n", " ").strip()
            if len(review_snippet) > 220:
                review_snippet = review_snippet[:220] + "..."

            # Product card HTML
            card = f"""
<div class="fk-card">
  <img class="fk-img" src="{first_image}" onerror="this.style.display='none'"/>
  <div class="fk-info">
    <div class="fk-title">{title}</div>

    <div class="fk-meta"><b>Brand:</b> {brand}</div>
    <div class="fk-meta"><b>Product Rating:</b> {product_rating}</div>
    <div class="fk-meta"><b>Overall Rating:</b> {overall_rating}</div>

    <div class="fk-price">
      <span class="fk-price-new">₹{discounted_price}</span>
      <span class="fk-price-old">₹{retail_price}</span>
    </div>

    <div class="fk-meta"><b>Review:</b> {review_snippet}</div>

    <a class="fk-btn" href="{url}" target="_blank" rel="noopener noreferrer">
      View on Flipkart
    </a>
  </div>
</div>
"""
            formatted.append(card)

        return "\n".join(formatted)

    return flipkart_retriever_tool


class RAGAgentBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = init_chat_model(Config.RAG_MODEL)

    def build_agent(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        flipkart_tool = build_flipkart_retriever_tool(retriever)

        agent = create_agent(
            model=self.model,
            tools=[flipkart_tool],
            system_prompt="""
You are a Flipkart product recommendation chatbot.

Rules:
1. If the user says hi/hello, greet politely and ask what product they want.
2. If the user asks for a product OR continues a product conversation
   (example: "yes office chair"), ALWAYS call flipkart_retriever_tool.
3. If results are weak or missing, ask 1 clarification question instead of refusing.
4. Keep answers short and show products as cards.
""",
            checkpointer=InMemorySaver(),
            middleware=[
                SummarizationMiddleware(
                    model=self.model,
                    trigger=("messages", 10),
                    keep=("messages", 4),
                )
            ],
        )

        return agent