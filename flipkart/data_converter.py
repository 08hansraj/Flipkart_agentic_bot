import pandas as pd
from langchain_core.documents import Document
import math
import json
import re


class DataConverter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @staticmethod
    def clean_value(v):
        if v is None:
            return None

        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None

        # Clean strings
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() in ["nan", "none", "null"]:
                return None
            return v

        return v

    @staticmethod
    def safe_int(x):
        try:
            if x is None:
                return None
            if isinstance(x, str) and x.lower() == "nan":
                return None
            return int(float(x))
        except Exception:
            return None

    @staticmethod
    def safe_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, str) and x.lower() == "nan":
                return None
            return float(x)
        except Exception:
            return None

    @staticmethod
    def clean_text(s: str) -> str:
        if not s:
            return ""
        s = str(s)
        s = s.replace("â€™", "'").replace("â€“", "-").replace("â€œ", '"').replace("â€", '"')

        # Remove common ecommerce boilerplate
        boilerplate_patterns = [
            r"Only Genuine Products\.*",
            r"30 Day Replacement Guarantee\.*",
            r"Free Shipping\.*",
            r"Cash On Delivery\!*",
            r"from Flipkart\.com\.*",
        ]
        for pat in boilerplate_patterns:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)

        # Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def extract_description_from_embedding_text(embedding_text: str) -> str:
        """
        Your embedding_text contains:
        Product: ...
        Brand: ...
        Category: ...
        Description: ...
        Specifications: ...

        We'll extract only the Description part (short).
        """
        if not embedding_text:
            return ""

        t = str(embedding_text)

        # Extract between "Description:" and "Specifications:"
        m = re.search(r"Description:\s*(.*?)(Specifications:|$)", t, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""

        desc = m.group(1)
        desc = DataConverter.clean_text(desc)

        # Hard truncate (very important)
        if len(desc) > 450:
            desc = desc[:450].rsplit(" ", 1)[0] + "..."

        return desc

    @staticmethod
    def extract_specs_from_embedding_text(embedding_text: str, max_specs: int = 8) -> dict:
        """
        Extract specs JSON from the end if present.
        Keep only a few key-value pairs.
        """
        if not embedding_text:
            return {}

        t = str(embedding_text)

        # Find "Specifications:" section
        m = re.search(r"Specifications:\s*(.*)$", t, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return {}

        spec_raw = m.group(1).strip()

        # Specs sometimes contain JSON-like string
        # We'll try to parse it safely.
        try:
            spec_json = json.loads(spec_raw)
            ps = spec_json.get("product_specification", [])
            out = {}

            for item in ps:
                if not isinstance(item, dict):
                    continue
                k = item.get("key")
                v = item.get("value")
                if not k or not v:
                    continue
                k = str(k).strip()
                v = str(v).strip()

                # Skip useless keys
                if k.lower() in ["sales package", "pack of", "number of contents in sales package"]:
                    continue

                out[k] = v
                if len(out) >= max_specs:
                    break

            return out
        except Exception:
            return {}

    @staticmethod
    def build_embedding_text(
        product_name: str,
        brand: str,
        category_path: str,
        retail_price,
        discounted_price,
        product_rating,
        overall_rating,
        embedding_text_raw: str,
    ) -> str:
        """
        Build a compact, high-signal text for embeddings.
        """
        desc = DataConverter.extract_description_from_embedding_text(embedding_text_raw)
        specs = DataConverter.extract_specs_from_embedding_text(embedding_text_raw, max_specs=8)

        rp = DataConverter.safe_int(retail_price)
        dp = DataConverter.safe_int(discounted_price)

        pr = DataConverter.safe_float(product_rating)
        orr = DataConverter.safe_float(overall_rating)

        lines = []

        if product_name:
            lines.append(f"Product: {product_name}")
        if brand:
            lines.append(f"Brand: {brand}")
        if category_path:
            # Make it embedding-friendly
            lines.append(f"Category: {category_path.replace('>>', '>')}")

        if dp is not None:
            lines.append(f"Discounted Price: {dp}")
        if rp is not None:
            lines.append(f"Retail Price: {rp}")

        if pr is not None:
            lines.append(f"Product Rating: {pr}")
        if orr is not None:
            lines.append(f"Overall Rating: {orr}")

        if desc:
            lines.append(f"Description: {desc}")

        if specs:
            # Convert to compact string
            spec_str = ", ".join([f"{k}={v}" for k, v in specs.items()])
            lines.append(f"Specs: {spec_str}")

        # Final cleanup
        text = "\n".join(lines).strip()

        # Hard length cap (VERY IMPORTANT)
        if len(text) > 1600:
            text = text[:1600]

        return text

    def convert(self):
        df = pd.read_json(self.file_path, lines=True)[
            [
                "id",
                "embedding_text",
                "product_name",
                "brand",
                "category_path",
                "product_url",
                "image",
                "retail_price",
                "discounted_price",
                "product_rating",
                "overall_rating",
                "is_FK_Advantage_product",
            ]
        ]

        docs = []
        for _, row in df.iterrows():
            pid = self.clean_value(row["id"])
            if pid is not None:
                pid = str(pid).strip()

            product_name = self.clean_value(row["product_name"])
            brand = self.clean_value(row["brand"])
            category_path = self.clean_value(row["category_path"])

            retail_price = self.clean_value(row["retail_price"])
            discounted_price = self.clean_value(row["discounted_price"])
            product_rating = self.clean_value(row["product_rating"])
            overall_rating = self.clean_value(row["overall_rating"])

            raw_embedding_text = self.clean_value(row["embedding_text"]) or ""

            # ✅ Build clean embedding text
            final_embedding_text = self.build_embedding_text(
                product_name=product_name,
                brand=brand,
                category_path=category_path,
                retail_price=retail_price,
                discounted_price=discounted_price,
                product_rating=product_rating,
                overall_rating=overall_rating,
                embedding_text_raw=raw_embedding_text,
            )

            doc = Document(
                page_content=final_embedding_text,
                metadata={
                    "id": pid,
                    "product_name": product_name,
                    "brand": brand,
                    "category_path": category_path,
                    "product_url": self.clean_value(row["product_url"]),
                    "image": self.clean_value(row["image"]),
                    "retail_price": retail_price,
                    "discounted_price": discounted_price,
                    "product_rating": product_rating,
                    "overall_rating": overall_rating,
                    "is_FK_Advantage_product": self.clean_value(row["is_FK_Advantage_product"]),
                },
            )
            docs.append(doc)

        return docs