import pandas as pd
import numpy as np
import ast
import re
import orjson
from tqdm import tqdm

INPUT_CSV = r"data\raw\org_data.csv"   # change if your name differs
OUTPUT_JSONL = "flipkart_products_prepared_25k.jsonl"
OUTPUT_CSV = "flipkart_products_prepared_25k.csv"

SAMPLE_SIZE = 25000
SEED = 42


# ---------------------------
# Helpers
# ---------------------------

def fix_bad_encoding(text: str) -> str:
    """Fix common mojibake artifacts from older crawls."""
    if not isinstance(text, str):
        return ""
    text = text.replace("â€¢", "•")
    text = text.replace("â€™", "'")
    text = text.replace("â€œ", '"').replace("â€", '"')
    text = text.replace("â€“", "-").replace("â€”", "-")
    text = text.replace("Â", " ")
    return text


def safe_parse_category_tree(cat_tree):
    """
    Input looks like:
    ["Clothing >> Women's Clothing >> Lingerie... >> Shorts >> ..."]
    We'll extract the first path and clean it.
    """
    if not isinstance(cat_tree, str):
        return ""

    cat_tree = cat_tree.strip()
    if not cat_tree:
        return ""

    try:
        parsed = ast.literal_eval(cat_tree)
        if isinstance(parsed, list) and len(parsed) > 0:
            return str(parsed[0]).strip()
    except:
        pass

    return cat_tree


def safe_parse_specifications(spec_str):
    """
    product_specifications looks like:
    {"product_specification"=>[{"key":"Fabric","value":"Cotton"}, ...]}
    But sometimes it is malformed Ruby hash style.

    We'll try best-effort extraction into readable text.
    """
    if not isinstance(spec_str, str):
        return ""

    s = spec_str.strip()
    if not s:
        return ""

    # Fix Ruby hash rocket => to :
    s = s.replace("=>", ":")

    # Remove weird wrapper keys
    # {"product_specification":[{...}]}
    try:
        # Some rows are not valid JSON due to single quotes.
        # We'll do a rough conversion.
        s2 = s.replace("'", '"')

        # Remove trailing invalid characters
        s2 = re.sub(r",\s*}", "}", s2)
        s2 = re.sub(r",\s*]", "]", s2)

        obj = orjson.loads(s2.encode("utf-8"))

        # Expected: {"product_specification": [ {"key": "...", "value": "..."}, ...]}
        if isinstance(obj, dict):
            ps = obj.get("product_specification")
            if isinstance(ps, list):
                parts = []
                for item in ps:
                    if not isinstance(item, dict):
                        continue
                    k = item.get("key")
                    v = item.get("value")

                    if k and v:
                        parts.append(f"{k}: {v}")
                    elif v:
                        parts.append(str(v))
                return " | ".join(parts).strip()

    except:
        pass

    # fallback: keep raw but cleaned
    return s[:1500]


def normalize_price(x):
    if pd.isna(x):
        return None
    try:
        x = str(x).strip()
        if x.lower() in ["", "nan", "none"]:
            return None
        return float(x)
    except:
        return None


def normalize_rating(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x.lower() in ["no rating available", "", "nan", "none"]:
        return None
    try:
        return float(x)
    except:
        return None


def clean_text_basic(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = fix_bad_encoding(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# ---------------------------
# Main
# ---------------------------

print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# Basic cleaning
for col in ["product_name", "description", "brand", "product_url"]:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(clean_text_basic)

# Parse category tree
df["category_path"] = df["product_category_tree"].apply(safe_parse_category_tree)
df["category_path"] = df["category_path"].apply(clean_text_basic)

# Parse specs
tqdm.pandas()
df["spec_text"] = df["product_specifications"].progress_apply(safe_parse_specifications)
df["spec_text"] = df["spec_text"].apply(clean_text_basic)

# Normalize price and rating
df["retail_price_num"] = df["retail_price"].apply(normalize_price)
df["discounted_price_num"] = df["discounted_price"].apply(normalize_price)

df["product_rating_num"] = df["product_rating"].apply(normalize_rating)
df["overall_rating_num"] = df["overall_rating"].apply(normalize_rating)

# Keep only rows with meaningful text
df["description"] = df["description"].fillna("").astype(str)
df["product_name"] = df["product_name"].fillna("").astype(str)

df = df[(df["product_name"].str.len() > 3)]

# Create embedding text
def build_embedding_text(row):
    name = row.get("product_name", "")
    brand = row.get("brand", "")
    cat = row.get("category_path", "")
    desc = row.get("description", "")
    specs = row.get("spec_text", "")

    chunks = []
    if name:
        chunks.append(f"Product: {name}")
    if brand and brand.lower() != "nan":
        chunks.append(f"Brand: {brand}")
    if cat:
        chunks.append(f"Category: {cat}")
    if desc:
        chunks.append(f"Description: {desc}")
    if specs:
        chunks.append(f"Specifications: {specs}")

    return "\n".join(chunks).strip()


print("Building embedding_text...")
df["embedding_text"] = df.apply(build_embedding_text, axis=1)

# Remove very short embedding_text
df = df[df["embedding_text"].str.len() > 80]

# Remove duplicates (same pid or same product_url)
if "pid" in df.columns:
    df = df.drop_duplicates(subset=["pid"])
else:
    df = df.drop_duplicates(subset=["product_url"])

# Sample 25k
df = df.reset_index(drop=True)

print("Final rows:", len(df))

# Prepare final export dataframe
final = pd.DataFrame({
    "id": df["pid"].fillna(df["uniq_id"]).astype(str),
    "embedding_text": df["embedding_text"],

    # Metadata
    "product_name": df["product_name"],
    "brand": df["brand"],
    "category_path": df["category_path"],
    "product_url": df["product_url"],
    "image": df["image"],
    "retail_price": df["retail_price_num"],
    "discounted_price": df["discounted_price_num"],
    "product_rating": df["product_rating_num"],
    "overall_rating": df["overall_rating_num"],
    "is_FK_Advantage_product": df["is_FK_Advantage_product"],
})

# Save CSV
final.to_csv(OUTPUT_CSV, index=False)

# Save JSONL (best for ingestion)
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for _, row in final.iterrows():
        record = row.to_dict()
        f.write(orjson.dumps(record).decode("utf-8") + "\n")

print("Saved:")
print("-", OUTPUT_CSV)
print("-", OUTPUT_JSONL)

print("Total rows:", len(df))
print("Rows with embedding_text >= 300:", (df["embedding_text"].str.len() >= 300).sum())
print("Rows with embedding_text >= 200:", (df["embedding_text"].str.len() >= 200).sum())