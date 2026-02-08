import pandas as pd
from langchain_core.documents import Document
import math


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

        return v

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

        docs = [
            Document(
                page_content=row["embedding_text"],
                metadata={
                    "id": self.clean_value(row["id"]),
                    "product_name": self.clean_value(row["product_name"]),
                    "brand": self.clean_value(row["brand"]),
                    "category_path": self.clean_value(row["category_path"]),
                    "product_url": self.clean_value(row["product_url"]),
                    "image": self.clean_value(row["image"]),
                    "retail_price": self.clean_value(row["retail_price"]),
                    "discounted_price": self.clean_value(row["discounted_price"]),
                    "product_rating": self.clean_value(row["product_rating"]),
                    "overall_rating": self.clean_value(row["overall_rating"]),
                    "is_FK_Advantage_product": self.clean_value(row["is_FK_Advantage_product"]),
                },
            )
            for _, row in df.iterrows()
        ]

        return docs