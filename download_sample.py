from datasets import load_dataset
import pandas as pd

SAMPLE_SIZE = 25000
SEED = 42

print("Loading Amazon US Reviews (Electronics)...")

ds = load_dataset("amazon_us_reviews", "Electronics_v1_00", split="train")

print("Total rows:", len(ds))

print(f"Sampling {SAMPLE_SIZE} rows...")
sample = ds.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

df = sample.to_pandas()

print("Columns found:")
print(df.columns)

# Clean and prepare embedding text
df = df.dropna(subset=["review_body"])
df["review_body"] = df["review_body"].astype(str).str.strip()
df = df[df["review_body"].str.len() > 20]

df["embedding_text"] = (
    df["product_title"].fillna("").astype(str).str.strip()
    + "\n\n"
    + df["review_body"].fillna("").astype(str).str.strip()
).str.strip()

final = df[[
    "product_id",
    "product_title",
    "star_rating",
    "review_headline",
    "review_body",
    "verified_purchase",
    "helpful_votes",
    "total_votes",
    "embedding_text"
]].copy()

print("Final rows:", len(final))

final.to_csv("amazon_electronics_25k.csv", index=False)
final.to_json("amazon_electronics_25k.jsonl", orient="records", lines=True)

print("Saved:")
print("- amazon_electronics_25k.csv")
print("- amazon_electronics_25k.jsonl")