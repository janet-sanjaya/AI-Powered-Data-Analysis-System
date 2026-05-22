import pandas as pd
import numpy as np
import re


EXPECTED_COLUMNS = {
    "date": ["date", "transaction_date", "order_date", "datetime", "day"],
    "branch": ["branch", "store", "store_branch", "location_branch"],
    "region": ["region", "area", "zone", "territory"],
    "category": ["category", "product_category", "product", "cat"],
    "sales": ["sales", "sale", "revenue", "amount", "total_sales"],
    "profit": ["profit", "gross_profit", "net_profit"],
    "quantity": ["quantity", "qty", "units", "unit_sold"],
    "orders": ["orders", "order_count", "ordercount", "num_orders", "order_number"],
}


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def _standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    original_cols = list(df.columns)
    normalized_map = {_normalize_text(c): c for c in original_cols}

    rename_map = {}
    for canonical, aliases in EXPECTED_COLUMNS.items():
        found = None

        for alias in [canonical] + aliases:
            alias_norm = _normalize_text(alias)
            if alias_norm in normalized_map:
                found = normalized_map[alias_norm]
                break

        if found is None:
            for col in original_cols:
                col_norm = _normalize_text(col)
                if canonical in col_norm or any(alias in col_norm for alias in aliases):
                    found = col
                    break

        if found is not None:
            rename_map[found] = canonical

    df = df.rename(columns=rename_map)

    keep_cols = [c for c in EXPECTED_COLUMNS.keys() if c in df.columns]
    df = df[keep_cols]

    return df, rename_map


def _clean_string_column(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"nan": pd.NA, "None": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
    )


def _standardize_text_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "branch" in df.columns:
        s = _clean_string_column(df["branch"]).fillna("Unknown")
        s = s.str.upper()
        s = s.replace({"UNKNOWN": "Unknown"})
        df["branch"] = s

    if "region" in df.columns:
        s = _clean_string_column(df["region"]).fillna("Unknown")
        s = s.str.title()
        df["region"] = s

    if "category" in df.columns:
        s = _clean_string_column(df["category"]).fillna("Unknown")
        s = s.str.title()
        df["category"] = s

    return df


def preprocess_csv(path_or_df):
    """
    Load and lightly standardize a CSV without changing the underlying values.

    Returns:
        cleaned_df, report
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    report = {
        "original_rows": len(df),
        "original_columns": list(df.columns),
        "renamed_columns": {},
        "rows_dropped_missing_date": 0,
        "duplicates_removed": 0,
        "missing_values_filled": {},
        "outliers_capped": [],
    }

    df, rename_map = _standardize_columns(df)
    report["renamed_columns"] = rename_map

    required = ["date", "branch", "region", "category", "sales", "profit", "quantity", "orders"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = df[required]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["sales", "profit", "quantity", "orders"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["branch", "region", "category"]:
        df[col] = _clean_string_column(df[col])

    df = _standardize_text_values(df)

    df["sales"] = df["sales"].astype(float)
    df["profit"] = df["profit"].astype(float)

    # Keep nullable integer types so missing values do not get forced to 0.
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").round().astype("Int64")
    df["orders"] = pd.to_numeric(df["orders"], errors="coerce").round().astype("Int64")

    df = df.sort_values("date").reset_index(drop=True)

    report["final_rows"] = len(df)
    report["final_columns"] = list(df.columns)

    return df, report


if __name__ == "__main__":
    cleaned_df, report = preprocess_csv("data/messy_sales_data.csv")
    print("Preprocessing complete")
    print(report)
    print(cleaned_df.head())