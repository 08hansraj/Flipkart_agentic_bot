import json
import re
from typing import List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool

from flipkart.config import Config


def detect_category_hint(query: str) -> str:
    """
    Returns a dataset-aligned category prefix like:
    'Clothing >> Men's Clothing'
    """
    q = query.lower().strip()

    # Men's clothing
    if any(x in q for x in ["tshirt", "t-shirt", "tee"]):
        if "women" not in q and "girl" not in q:
            return "Clothing >> Men's Clothing"

    if any(x in q for x in ["men shirt", "mens shirt", "shirt for men"]):
        return "Clothing >> Men's Clothing"

    # Women's clothing
    if any(x in q for x in ["women", "girls", "lady", "ladies", "dress", "kurti", "top", "bra", "lingerie"]):
        return "Clothing >> Women's Clothing"

    # Jewellery
    if any(x in q for x in ["jewellery", "jewelry", "necklace", "ring", "bangle", "bracelet", "earring"]):
        return "Jewellery >>"

    # Footwear
    if any(x in q for x in ["shoes", "footwear", "heels", "wedges", "boots", "sandals", "slippers", "loafers"]):
        return "Footwear >>"

    # Watches
    if "watch" in q or "watches" in q:
        return "Watches >>"

    # Home furnishing / decor
    if any(x in q for x in ["curtain", "bedsheet", "blanket", "quilt", "pillow", "furnishing"]):
        return "Home Furnishing >>"

    if any(x in q for x in ["wall sticker", "wall decor", "showpiece", "decor", "decoration"]):
        return "Home Decor & Festive Needs >>"

    # Automotive
    if any(x in q for x in ["car", "auto", "automotive", "seat cover", "car mat", "steering"]):
        return "Automotive >>"

    return ""


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def keyword_score(query: str, title: str, brand: str, category: str) -> int:
    """
    Lightweight reranking.
    Helps with stability a LOT for ecommerce.
    """
    q = normalize_text(query)
    if not q:
        return 0

    q_tokens = set(q.split())
    score = 0

    t = normalize_text(title)
    b = normalize_text(brand)
    c = normalize_text(category)

    # Brand match
    if b and b in q:
        score += 5

    # Token overlap
    if t:
        t_tokens = set(t.split())
        score += min(6, len(q_tokens.intersection(t_tokens)) * 2)

    # Category overlap
    if c:
        c_tokens = set(c.split())
        score += min(4, len(q_tokens.intersection(c_tokens)))

    return score


def safe_rating(x):
    if x is None:
        return None
    try:
        if str(x).lower() == "nan":
            return None
        return round(float(x), 1)
    except Exception:
        return None


def safe_price(x):
    if x is None:
        return None
    try:
        if str(x).lower() == "nan":
            return None
        return int(float(x))
    except Exception:
        return None


def parse_first_image(image_field) -> str:
    if not image_field:
        return ""

    if isinstance(image_field, str) and image_field.strip().startswith("["):
        try:
            imgs = json.loads(image_field)
            if isinstance(imgs, list) and len(imgs) > 0:
                return str(imgs[0]).strip()
        except Exception:
            return ""

    if isinstance(image_field, str):
        return image_field.strip()

    return ""


def build_flipkart_retriever_tool(retriever):
    @tool
    def flipkart_retriever_tool(query: str) -> str:
        """
        Returns JSON string:
        {
          "reply": "...",
          "products": [ { ... }, ... ]
        }
        """

        category_hint = detect_category_hint(query)

        # IMPORTANT: don't pollute the query with "Flipkart products..."
        biased_query = f"{query} {category_hint}".strip()

        # Pull more docs, then rerank
        docs = retriever.invoke(biased_query) or []
        docs = docs[:20]

        if not docs:
            payload = {
                "reply": "I couldn’t find a strong match in the dataset. Try a simpler product name like 'necklace', 'men tshirt', or 'loafers'.",
                "products": [],
            }
            return json.dumps(payload, ensure_ascii=False)

        products: List[Dict[str, Any]] = []

        for d in docs:
            m = d.metadata or {}

            title = m.get("product_name") or "Unknown Product"
            brand = m.get("brand") or "N/A"
            category_path = m.get("category_path") or ""

            retail_price = safe_price(m.get("retail_price"))
            discounted_price = safe_price(m.get("discounted_price"))

            product_rating = safe_rating(m.get("product_rating"))
            overall_rating = safe_rating(m.get("overall_rating"))

            url = m.get("product_url") or ""
            first_image = parse_first_image(m.get("image"))

            # Use a short snippet from page_content as description
            # Since your page_content is now compact embedding text,
            # we extract the "Description:" line if present.
            desc = ""
            txt = (d.page_content or "").replace("\n", " ")
            match = re.search(r"Description:\s*(.*?)(Specs:|$)", txt, flags=re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
            if len(desc) > 180:
                desc = desc[:180].rsplit(" ", 1)[0] + "..."

            products.append(
                {
                    "id": m.get("id"),
                    "title": title,
                    "brand": brand,
                    "category_path": category_path,
                    "discounted_price": discounted_price,
                    "retail_price": retail_price,
                    "product_rating": product_rating,
                    "overall_rating": overall_rating,
                    "image": first_image,
                    "url": url,
                    "description": desc,
                }
            )

        # Rerank
        for p in products:
            p["_score"] = keyword_score(
                query=query,
                title=p.get("title", ""),
                brand=p.get("brand", ""),
                category=p.get("category_path", ""),
            )

            # Extra boost if category hint matches
            if category_hint and p.get("category_path", "").startswith(category_hint):
                p["_score"] += 4

        products.sort(key=lambda x: x["_score"], reverse=True)

        # Keep top 4
        products = products[:4]

        # Remove internal scoring
        for p in products:
            p.pop("_score", None)

        payload = {
            "reply": "Here are the best matches I found:",
            "products": products,
        }

        return json.dumps(payload, ensure_ascii=False)

    return flipkart_retriever_tool


class RAGAgentBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = init_chat_model(Config.RAG_MODEL)

    def build_agent(self):
        # MMR for better diversity + stability
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 60, "lambda_mult": 0.75},
        )

        flipkart_tool = build_flipkart_retriever_tool(retriever)

        agent = create_agent(
            model=self.model,
            tools=[flipkart_tool],
            system_prompt="""
You are a Flipkart product recommendation assistant.

Rules:
1) If user greets (hi/hello), respond politely and ask what they want to buy.
2) For any shopping/product intent, ALWAYS call flipkart_retriever_tool(query).
3) Do NOT ask clarification before calling the tool.
4) After tool returns, respond ONLY with the JSON returned by the tool.
5) Never mention internal tools or vector database.
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