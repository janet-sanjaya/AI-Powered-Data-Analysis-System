import pandas as pd
import random
import os

random.seed(42)

def make_messy(df):
    df = df.copy()

    # Messy column names only
    df.columns = [
        random.choice([c, c.upper(), c + " ", c.capitalize()])
        for c in df.columns
    ]

    # Whitespace and casing only
    for col in df.columns:
        if col.strip().lower() in ["branch", "region", "category"]:
            df[col] = df[col].apply(
                lambda x: f" {x} " if pd.notna(x) and random.random() < 0.3 else x
            )
            df[col] = df[col].apply(
                lambda x: x.upper() if pd.notna(x) and random.random() < 0.5 else x
            )

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_path = os.path.join(base_dir, "data", "clean_sales_data.csv")
    messy_path = os.path.join(base_dir, "data", "messy_sales_data.csv")

    clean_df = pd.read_csv(clean_path)
    messy_df = make_messy(clean_df)
    messy_df.to_csv(messy_path, index=False)

    print("Saved messy dataset to:", messy_path)