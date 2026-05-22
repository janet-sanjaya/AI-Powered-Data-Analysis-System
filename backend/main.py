from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import uuid
import re
import traceback
import calendar

import pandas as pd
import numpy as np

from backend.query_cleaner import normalize_query
from backend.preprocessing import preprocess_csv
from backend.llm import (
    chat_completion,
    build_code_prompt,
    build_explanation_prompt,
)
from backend.sandbox_executor import execute_pandas_code_in_sandbox
from backend.executor import result_to_dataframe
from backend.recommendation import generate_recommendation


def clean_llm_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


FALLBACK_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

DIMENSION_ORDER = {
    "branch": ["A", "B", "C", "D", "E"],
    "region": ["West", "East", "Central", "South"],
    "category": ["Electronics", "Clothing", "Furniture", "Grocery"],
}

app = FastAPI(title="AInsight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASTORE = {}


class AnalyzeRequest(BaseModel):
    file_id: str
    question: str
    model: str = "openai/gpt-4o-mini"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        cleaned_df, report = preprocess_csv(temp_path)
        os.remove(temp_path)

        file_id = str(uuid.uuid4())
        DATASTORE[file_id] = cleaned_df
        print("FILE_ID:", file_id)

        return {
            "file_id": file_id,
            "preview": cleaned_df.head(10).to_dict(orient="records"),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _models_to_try(requested_model: str):
    return [requested_model] + [m for m in FALLBACK_MODELS if m != requested_model]


def _generate_code(messages, models):
    last_error = None
    for m in models:
        try:
            print("Trying:", m)
            text = chat_completion(m, messages)
            return clean_llm_code(text), m
        except Exception as e:
            last_error = e
    raise Exception(f"All models failed: {last_error}")


def _month_num_from_question(q: str):
    q = q.lower()
    for month_name, month_num in MONTH_MAP.items():
        if month_name in q:
            return month_num
    return None


def _metric_from_question(q: str) -> str:
    q = q.lower()
    if "margin" in q:
        return "margin"
    if "profit" in q:
        return "profit"
    if "quantity" in q:
        return "quantity"
    if "orders" in q:
        return "orders"
    if "revenue" in q:
        return "sales"
    return "sales"


def _dimension_from_question(q: str):
    q = q.lower()
    if "branch" in q:
        return "branch"
    if "region" in q:
        return "region"
    if "category" in q:
        return "category"
    return None


def _grouping_phrase_present(q: str) -> bool:
    q = q.lower()
    return any(
        phrase in q
        for phrase in [
            " by branch",
            " by region",
            " by category",
            "each branch",
            "each region",
            "each category",
            "per branch",
            "per region",
            "per category",
            "for each branch",
            "for each region",
            "for each category",
        ]
    )


def _analysis_year(df: pd.DataFrame) -> int:
    years = pd.to_datetime(df["date"], errors="coerce").dt.year.dropna()
    if len(years) == 0:
        return 2024
    return int(years.mode().iat[0])


def _period_label(month_num: int, year: int) -> str:
    return f"{calendar.month_name[month_num]} {year}"


def _normalize_text_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].astype(str).str.strip().str.lower()


def _sort_dimension_frame(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if dimension not in frame.columns:
        return frame

    if dimension in DIMENSION_ORDER:
        order = DIMENSION_ORDER[dimension]
        frame = frame.copy()
        frame[dimension] = pd.Categorical(frame[dimension], categories=order, ordered=True)
        return frame.sort_values(dimension)

    return frame.sort_values(dimension)


def _should_exclude_unknown(q: str) -> bool:
    q = q.lower()
    return any(
        phrase in q
        for phrase in [
            "excluding unknown",
            "exclude unknown",
            "without unknown",
            "ignore unknown",
            "except unknown",
            "remove unknown",
            "not include unknown",
            "no unknown",
        ]
    )


def _ensure_margin_column(temp: pd.DataFrame) -> pd.DataFrame:
    temp = temp.copy()
    temp["margin"] = np.where(
        temp["sales"] != 0,
        (temp["profit"] / temp["sales"]) * 100,
        np.nan,
    )
    return temp


def _build_filter_code_block(q: str, temp_name: str = "temp", include_month: bool = True) -> str:
    q = q.lower()
    lines = []

    if _should_exclude_unknown(q):
        lines.append(
            f'{temp_name} = {temp_name}[{temp_name}["branch"].astype(str).str.strip().str.lower() != "unknown"]'
        )

    if include_month:
        month_num = _month_num_from_question(q)
        if month_num is not None:
            lines.append(f'{temp_name} = {temp_name}[{temp_name}["date"].dt.month == {month_num}]')

    for cat in ["electronics", "clothing", "furniture", "grocery"]:
        if cat in q:
            lines.append(
                f'{temp_name} = {temp_name}[{temp_name}["category"].astype(str).str.strip().str.lower() == "{cat}"]'
            )

    for region in ["west", "east", "central", "south"]:
        if region in q:
            lines.append(
                f'{temp_name} = {temp_name}[{temp_name}["region"].astype(str).str.strip().str.lower() == "{region}"]'
            )

    for branch in ["a", "b", "c", "d", "e"]:
        if f"branch {branch}" in q or f"branch is {branch}" in q:
            lines.append(
                f'{temp_name} = {temp_name}[{temp_name}["branch"].astype(str).str.strip().str.upper() == "{branch.upper()}"]'
            )

    return "\n".join(lines)


def _apply_filter_rules(temp: pd.DataFrame, q: str, include_month: bool = True) -> pd.DataFrame:
    q = q.lower()

    if _should_exclude_unknown(q):
        temp = temp[temp["branch"].astype(str).str.strip().str.lower() != "unknown"]

    if include_month:
        month_num = _month_num_from_question(q)
        if month_num is not None:
            temp = temp[temp["date"].dt.month == month_num]

    for cat in ["electronics", "clothing", "furniture", "grocery"]:
        if cat in q:
            temp = temp[temp["category"].astype(str).str.strip().str.lower() == cat]

    for region in ["west", "east", "central", "south"]:
        if region in q:
            temp = temp[temp["region"].astype(str).str.strip().str.lower() == region]

    for branch in ["a", "b", "c", "d", "e"]:
        if f"branch {branch}" in q or f"branch is {branch}" in q:
            temp = temp[temp["branch"].astype(str).str.strip().str.upper() == branch.upper()]

    return temp


def _extract_months_in_order(q: str):
    q = q.lower()
    found = []
    for month_name, month_num in MONTH_MAP.items():
        if month_name in q:
            found.append((q.index(month_name), month_num))
    found.sort(key=lambda x: x[0])
    return [m for _, m in found]


def _previous_month(month_num: int) -> int:
    return 12 if month_num == 1 else month_num - 1


def _code_prefix(q: str, include_month: bool = True, temp_name: str = "temp") -> list[str]:
    lines = [
        f'{temp_name} = df.copy()',
        f'{temp_name}["date"] = pd.to_datetime({temp_name}["date"], errors="coerce")',
    ]
    filter_block = _build_filter_code_block(q, temp_name=temp_name, include_month=include_month)
    if filter_block:
        lines.extend(filter_block.splitlines())
    return lines


def _margin_scalar(temp: pd.DataFrame) -> float:
    sales = temp["sales"].sum()
    profit = temp["profit"].sum()
    return float((profit / sales) * 100) if sales != 0 else np.nan


def _month_metric_table(temp: pd.DataFrame, metric: str) -> pd.DataFrame:
    temp = temp.copy()
    temp["month"] = temp["date"].dt.month

    if metric == "margin":
        grouped = temp.groupby("month", as_index=False).agg(
            sales_sum=("sales", "sum"),
            profit_sum=("profit", "sum"),
        )
        grouped["margin"] = np.where(
            grouped["sales_sum"] != 0,
            (grouped["profit_sum"] / grouped["sales_sum"]) * 100,
            np.nan,
        )
        return grouped[["month", "margin"]]

    return temp.groupby("month", as_index=False)[metric].sum()


def _dimension_metric_table(temp: pd.DataFrame, dimension: str, metric: str) -> pd.DataFrame:
    temp = temp.copy()
    temp["month"] = temp["date"].dt.month

    if metric == "margin":
        grouped = temp.groupby([dimension, "month"], as_index=False).agg(
            sales_sum=("sales", "sum"),
            profit_sum=("profit", "sum"),
        )
        grouped["margin"] = np.where(
            grouped["sales_sum"] != 0,
            (grouped["profit_sum"] / grouped["sales_sum"]) * 100,
            np.nan,
        )
        return grouped[[dimension, "month", "margin"]]

    return temp.groupby([dimension, "month"], as_index=False)[metric].sum()

def _aov_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()

    if not (
        (("order" in q or "orders" in q) and "value" in q)
        or "aov" in q
    ):
        return None, None

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = _apply_filter_rules(temp, q, include_month=True)

    dimension = _dimension_from_question(q)

    if dimension:
        grouped = temp.groupby(dimension, as_index=False).agg(
            sales_sum=("sales", "sum"),
            orders_sum=("orders", "sum"),
        )
        grouped["result"] = np.where(
            grouped["orders_sum"] != 0,
            grouped["sales_sum"] / grouped["orders_sum"],
            np.nan,
        )
        grouped["result"] = grouped["result"].round(2)
        grouped = grouped[[dimension, "result"]].reset_index(drop=True)

        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=True)
            + f'\nresult = temp.groupby("{dimension}", as_index=False).agg(sales_sum=("sales", "sum"), orders_sum=("orders", "sum"))\n'
            + 'result["result"] = np.where(result["orders_sum"] != 0, result["sales_sum"] / result["orders_sum"], np.nan)\n'
            + 'result["result"] = result["result"].round(2)\n'
            + f'result = result[["{dimension}", "result"]]'
        )
        return grouped, code

    sales_sum = temp["sales"].sum()
    orders_sum = temp["orders"].sum()
    value = (sales_sum / orders_sum) if orders_sum != 0 else np.nan
    result_df = pd.DataFrame([{"result": round(float(value), 2)}])

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        + _build_filter_code_block(q, "temp", include_month=True)
        + '\nresult = round(temp["sales"].sum() / temp["orders"].sum(), 2)'
    )
    return result_df, code


def _group_by_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()

    months = _extract_months_in_order(q)
    if len(months) >= 2:
        return None, None

    if any(k in q for k in ["change", "compare", "compared", "difference", "month-over-month", "mom"]):
        return None, None

    if not _grouping_phrase_present(q):
        return None, None

    dimension = _dimension_from_question(q)
    if dimension is None:
        return None, None

    metric = _metric_from_question(q)
    agg = "mean" if any(k in q for k in ["average", "avg"]) else "sum"

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = _apply_filter_rules(temp, q, include_month=True)

    if temp.empty:
        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=True)
            + f'\nresult = temp.groupby("{dimension}", as_index=False)["{metric}"].{agg}()'
        )
        return pd.DataFrame(), code

    if metric == "margin":
        grouped = temp.groupby(dimension, as_index=False).agg(
            sales_sum=("sales", "sum"),
            profit_sum=("profit", "sum"),
        )
        grouped["margin"] = np.where(
            grouped["sales_sum"] != 0,
            (grouped["profit_sum"] / grouped["sales_sum"]) * 100,
            np.nan,
        )
        grouped = grouped[[dimension, "margin"]]
        grouped = _sort_dimension_frame(grouped, dimension).reset_index(drop=True)

        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=True)
            + f'\nresult = temp.groupby("{dimension}", as_index=False).agg(sales_sum=("sales", "sum"), profit_sum=("profit", "sum"))\n'
            + 'result["margin"] = np.where(result["sales_sum"] != 0, (result["profit_sum"] / result["sales_sum"]) * 100, np.nan)\n'
            + f'result = result[["{dimension}", "margin"]]\n'
            + f'result = result.sort_values("{dimension}").reset_index(drop=True)'
        )
        return grouped, code

    grouped = temp.groupby(dimension, as_index=False)[metric].agg(agg)
    grouped = _sort_dimension_frame(grouped, dimension).reset_index(drop=True)

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        + _build_filter_code_block(q, "temp", include_month=True)
        + f'\nresult = temp.groupby("{dimension}", as_index=False)["{metric}"].{agg}()\n'
        + f'result = result.sort_values("{dimension}").reset_index(drop=True)'
    )
    return grouped, code


def _top_bottom_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()

    if not any(k in q for k in ["highest", "most", "largest", "lowest", "least"]):
        return None, None

    dimension = _dimension_from_question(q)
    if dimension is None:
        return None, None

    metric = _metric_from_question(q)
    ascending = any(k in q for k in ["lowest", "least"])

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")

    if _should_exclude_unknown(q):
        temp = temp[temp["branch"].astype(str).str.strip().str.lower() != "unknown"]

    month_num = _month_num_from_question(q)
    if month_num is not None:
        temp = temp[temp["date"].dt.month == month_num]

    for cat in ["electronics", "clothing", "furniture", "grocery"]:
        if re.search(rf"\b{cat}\b", q):
            temp = temp[temp["category"].astype(str).str.strip().str.lower() == cat]

    region_patterns = {
        "west": [r"\bwest\s+region\b", r"\bregion\s+west\b", r"\bin\s+west\b", r"\bin\s+the\s+west\b", r"\bwest\s+area\b"],
        "east": [r"\beast\s+region\b", r"\bregion\s+east\b", r"\bin\s+east\b", r"\bin\s+the\s+east\b", r"\beast\s+area\b"],
        "central": [r"\bcentral\s+region\b", r"\bregion\s+central\b", r"\bin\s+central\b", r"\bin\s+the\s+central\b", r"\bcentral\s+area\b"],
        "south": [r"\bsouth\s+region\b", r"\bregion\s+south\b", r"\bin\s+south\b", r"\bin\s+the\s+south\b", r"\bsouth\s+area\b"],
    }
    for region, patterns in region_patterns.items():
        if any(re.search(p, q) for p in patterns):
            temp = temp[temp["region"].astype(str).str.strip().str.lower() == region]

    for branch in ["a", "b", "c", "d", "e"]:
        if re.search(rf"\bbranch\s+{branch}\b", q) or re.search(rf"\bbranch\s+is\s+{branch}\b", q):
            temp = temp[temp["branch"].astype(str).str.strip().str.upper() == branch.upper()]

    if metric == "margin":
        grouped = temp.groupby(dimension, as_index=False).agg(
            sales_sum=("sales", "sum"),
            profit_sum=("profit", "sum"),
        )
        grouped["margin"] = np.where(
            grouped["sales_sum"] != 0,
            (grouped["profit_sum"] / grouped["sales_sum"]) * 100,
            np.nan,
        )
        grouped = grouped.sort_values("margin", ascending=ascending).head(1).reset_index(drop=True)

        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + ('temp = temp[temp["branch"].astype(str).str.strip().str.lower() != "unknown"]\n' if _should_exclude_unknown(q) else '')
            + (f'temp = temp[temp["date"].dt.month == {month_num}]\n' if month_num is not None else '')
            + ''.join(
                f'temp = temp[temp["category"].astype(str).str.strip().str.lower() == "{cat}"]\n'
                for cat in ["electronics", "clothing", "furniture", "grocery"]
                if re.search(rf"\\b{cat}\\b", q)
            )
            + ''.join(
                f'temp = temp[temp["region"].astype(str).str.strip().str.lower() == "{region}"]\n'
                for region, patterns in region_patterns.items()
                if any(re.search(p, q) for p in patterns)
            )
            + ''.join(
                f'temp = temp[temp["branch"].astype(str).str.strip().str.upper() == "{branch.upper()}"]\n'
                for branch in ["a", "b", "c", "d", "e"]
                if re.search(rf"\\bbranch\\s+{branch}\\b", q) or re.search(rf"\\bbranch\\s+is\\s+{branch}\\b", q)
            )
            + f'result = temp.groupby("{dimension}", as_index=False).agg(sales_sum=("sales", "sum"), profit_sum=("profit", "sum"))\n'
            + 'result["margin"] = np.where(result["sales_sum"] != 0, (result["profit_sum"] / result["sales_sum"]) * 100, np.nan)\n'
            + f'result = result.sort_values("margin", ascending={ascending}).head(1).reset_index(drop=True)'
        )
        return grouped, code

    grouped = temp.groupby(dimension, as_index=False)[metric].sum().sort_values(metric, ascending=ascending).head(1)
    grouped = grouped.reset_index(drop=True)

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        + ('temp = temp[temp["branch"].astype(str).str.strip().str.lower() != "unknown"]\n' if _should_exclude_unknown(q) else '')
        + (f'temp = temp[temp["date"].dt.month == {month_num}]\n' if month_num is not None else '')
        + ''.join(
            f'temp = temp[temp["category"].astype(str).str.strip().str.lower() == "{cat}"]\n'
            for cat in ["electronics", "clothing", "furniture", "grocery"]
            if re.search(rf"\\b{cat}\\b", q)
        )
        + ''.join(
            f'temp = temp[temp["region"].astype(str).str.strip().str.lower() == "{region}"]\n'
            for region, patterns in region_patterns.items()
            if any(re.search(p, q) for p in patterns)
        )
        + ''.join(
            f'temp = temp[temp["branch"].astype(str).str.strip().str.upper() == "{branch.upper()}"]\n'
            for branch in ["a", "b", "c", "d", "e"]
            if re.search(rf"\\bbranch\\s+{branch}\\b", q) or re.search(rf"\\bbranch\\s+is\\s+{branch}\\b", q)
        )
        + f'result = temp.groupby("{dimension}", as_index=False)["{metric}"].sum().sort_values("{metric}", ascending={ascending}).head(1).reset_index(drop=True)'
    )
    return grouped, code


def _trend_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()
    if not any(k in q for k in ["trend", "over time", "over the year", "change over the year"]):
        return None, None

    metric = _metric_from_question(q)

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = _apply_filter_rules(temp, q, include_month=False)

    if temp.empty:
        return pd.DataFrame(), None

    temp["month_num"] = temp["date"].dt.month

    if metric == "margin":
        monthly = temp.groupby("month_num", as_index=False).agg(
            sales_sum=("sales", "sum"),
            profit_sum=("profit", "sum"),
        )
        monthly["margin"] = np.where(
            monthly["sales_sum"] != 0,
            (monthly["profit_sum"] / monthly["sales_sum"]) * 100,
            np.nan,
        )
        monthly = monthly.sort_values("month_num")
        monthly["month"] = monthly["month_num"].apply(lambda m: _period_label(int(m), _analysis_year(temp)))
        result = monthly[["month", "margin"]].reset_index(drop=True)

        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + '\ntemp["month_num"] = temp["date"].dt.month\n'
            + 'monthly = temp.groupby("month_num", as_index=False).agg(sales_sum=("sales", "sum"), profit_sum=("profit", "sum"))\n'
            + 'monthly["margin"] = np.where(monthly["sales_sum"] != 0, (monthly["profit_sum"] / monthly["sales_sum"]) * 100, np.nan)\n'
            + 'monthly = monthly.sort_values("month_num")\n'
            + f'monthly["month"] = monthly["month_num"].apply(lambda m: calendar.month_name[int(m)] + " " + str({_analysis_year(temp)}))\n'
            + 'result = monthly[["month", "margin"]].reset_index(drop=True)'
        )
        return result, code

    grouped = temp.groupby("month_num", as_index=False)[metric].sum().sort_values("month_num")
    grouped["month"] = grouped["month_num"].apply(lambda m: _period_label(int(m), _analysis_year(temp)))
    grouped = grouped[["month", metric]].reset_index(drop=True)

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        + _build_filter_code_block(q, "temp", include_month=False)
        + '\ntemp["month_num"] = temp["date"].dt.month\n'
        + f'result = temp.groupby("month_num", as_index=False)["{metric}"].sum().sort_values("month_num")\n'
        + f'result["month"] = result["month_num"].apply(lambda m: calendar.month_name[int(m)] + " " + str({_analysis_year(temp)}))\n'
        + f'result = result[["month", "{metric}"]]'
    )
    return grouped, code


def _month_compare_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()
    months = _extract_months_in_order(q)
    if len(months) < 2:
        return None, None

    if not any(k in q for k in ["change", "compare", "compared", "difference", "month-over-month", "mom", "how did"]):
        return None, None

    m1, m2 = months[0], months[1]
    metric = _metric_from_question(q)
    dimension = _dimension_from_question(q)
    year = _analysis_year(df)

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = _apply_filter_rules(temp, q, include_month=False)

    if metric == "margin":
        temp = _ensure_margin_column(temp)

    temp = temp[temp["date"].dt.month.isin([m1, m2])].copy()

    if temp.empty:
        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + f'\ntemp = temp[temp["date"].dt.month.isin([{m1}, {m2}])]'
        )
        return pd.DataFrame(), code

    if dimension:
        temp["month"] = temp["date"].dt.month
        monthly = _dimension_metric_table(temp, dimension, metric)
        pivot = monthly.pivot(index=dimension, columns="month", values=metric).reset_index()
        pivot.columns.name = None

        if m1 not in pivot.columns:
            pivot[m1] = 0.0
        if m2 not in pivot.columns:
            pivot[m2] = 0.0

        pivot[m1] = pivot[m1].fillna(0)
        pivot[m2] = pivot[m2].fillna(0)
        pivot["change"] = pivot[m2] - pivot[m1]
        pivot["pct_change"] = np.where(pivot[m1] != 0, (pivot["change"] / pivot[m1]) * 100, 0)

        result = pivot[[dimension, m1, m2, "change", "pct_change"]].copy()
        for col in [m1, m2, "change", "pct_change"]:
            result[col] = pd.to_numeric(result[col], errors="coerce").round(2)

        if any(k in q for k in ["drop", "decrease", "decline", "biggest"]):
            result = result.sort_values("change")
        else:
            result = _sort_dimension_frame(result, dimension)

        result = result.reset_index(drop=True)

        if metric == "margin":
            code = (
                'temp = df.copy()\n'
                'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
                + _build_filter_code_block(q, "temp", include_month=False)
                + '\ntemp = temp[temp["date"].notna()].copy()\n'
                + f'temp = temp[temp["date"].dt.month.isin([{m1}, {m2}])]\n'
                + 'temp["month"] = temp["date"].dt.month\n'
                + f'monthly = temp.groupby(["{dimension}", "month"], as_index=False).agg(sales_sum=("sales", "sum"), profit_sum=("profit", "sum"))\n'
                + 'monthly["margin"] = np.where(monthly["sales_sum"] != 0, (monthly["profit_sum"] / monthly["sales_sum"]) * 100, np.nan)\n'
                + f'pivot = monthly.pivot(index="{dimension}", columns="month", values="margin").reset_index()\n'
                + f'pivot["change"] = pivot[{m2}] - pivot[{m1}]\n'
                + f'pivot["pct_change"] = np.where(pivot[{m1}] != 0, (pivot["change"] / pivot[{m1}]) * 100, 0)\n'
                + f'result = pivot[["{dimension}", {m1}, {m2}, "change", "pct_change"]]'
            )
        else:
            code = (
                'temp = df.copy()\n'
                'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
                + _build_filter_code_block(q, "temp", include_month=False)
                + '\ntemp = temp[temp["date"].notna()].copy()\n'
                + f'temp = temp[temp["date"].dt.month.isin([{m1}, {m2}])]\n'
                + 'temp["month"] = temp["date"].dt.month\n'
                + f'monthly = temp.groupby(["{dimension}", "month"], as_index=False)["{metric}"].sum()\n'
                + f'pivot = monthly.pivot(index="{dimension}", columns="month", values="{metric}").reset_index()\n'
                + f'pivot["change"] = pivot[{m2}] - pivot[{m1}]\n'
                + f'pivot["pct_change"] = np.where(pivot[{m1}] != 0, (pivot["change"] / pivot[{m1}]) * 100, 0)\n'
                + f'result = pivot[["{dimension}", {m1}, {m2}, "change", "pct_change"]]'
            )

        return result, code

    if metric == "margin":
        month1_df = temp[temp["date"].dt.month == m1]
        month2_df = temp[temp["date"].dt.month == m2]
        m1_sales = month1_df["sales"].sum()
        m2_sales = month2_df["sales"].sum()
        m1_profit = month1_df["profit"].sum()
        m2_profit = month2_df["profit"].sum()
        m1_val = (m1_profit / m1_sales * 100) if m1_sales != 0 else 0.0
        m2_val = (m2_profit / m2_sales * 100) if m2_sales != 0 else 0.0
    else:
        m1_val = float(temp.loc[temp["date"].dt.month == m1, metric].sum())
        m2_val = float(temp.loc[temp["date"].dt.month == m2, metric].sum())

    change = m2_val - m1_val
    pct_change = (change / m1_val * 100) if m1_val != 0 else 0.0

    result = pd.DataFrame([{
        "month_1": _period_label(m1, year),
        "month_2": _period_label(m2, year),
        "value_1": round(m1_val, 2),
        "value_2": round(m2_val, 2),
        "change": round(change, 2),
        "pct_change": round(pct_change, 2),
    }])

    if metric == "margin":
        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + '\ntemp = temp[temp["date"].notna()].copy()\n'
            + f'temp = temp[temp["date"].dt.month.isin([{m1}, {m2}])]\n'
            + f'm1_df = temp[temp["date"].dt.month == {m1}]\n'
            + f'm2_df = temp[temp["date"].dt.month == {m2}]\n'
            + 'm1_sales = m1_df["sales"].sum()\n'
            + 'm2_sales = m2_df["sales"].sum()\n'
            + 'm1_profit = m1_df["profit"].sum()\n'
            + 'm2_profit = m2_df["profit"].sum()\n'
            + 'm1_val = (m1_profit / m1_sales * 100) if m1_sales != 0 else 0.0\n'
            + 'm2_val = (m2_profit / m2_sales * 100) if m2_sales != 0 else 0.0\n'
            + 'change = m2_val - m1_val\n'
            + 'pct_change = (change / m1_val * 100) if m1_val != 0 else 0\n'
            + f'result = pd.DataFrame([{{"month_1": "{_period_label(m1, year)}", "month_2": "{_period_label(m2, year)}", "value_1": round(float(m1_val), 2), "value_2": round(float(m2_val), 2), "change": round(float(change), 2), "pct_change": round(float(pct_change), 2)}}])'
        )
    else:
        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + f'\ntemp = temp[temp["date"].dt.month.isin([{m1}, {m2}])]\n'
            + f'm1_val = temp.loc[temp["date"].dt.month == {m1}, "{metric}"].sum()\n'
            + f'm2_val = temp.loc[temp["date"].dt.month == {m2}, "{metric}"].sum()\n'
            + 'change = m2_val - m1_val\n'
            + 'pct_change = (change / m1_val * 100) if m1_val != 0 else 0\n'
            + f'result = pd.DataFrame([{{"month_1": "{_period_label(m1, year)}", "month_2": "{_period_label(m2, year)}", "value_1": round(float(m1_val), 2), "value_2": round(float(m2_val), 2), "change": round(float(change), 2), "pct_change": round(float(pct_change), 2)}}])'
        )

    return result, code
def _branch_last_month_change_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()

    if "last month" not in q:
        return None, None

    if "branch" not in q:
        return None, None

    months = _extract_months_in_order(q)
    if len(months) == 0:
        return None, None

    target_month = months[0]
    prev_month = _previous_month(target_month)

    branch = None
    for b in ["a", "b", "c", "d", "e"]:
        if re.search(rf"\bbranch\s+{b}\b", q) or re.search(rf"\bbranch\s+is\s+{b}\b", q):
            branch = b.upper()
            break

    if branch is None:
        return None, None

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = temp[temp["date"].notna()].copy()
    temp = temp[temp["branch"].astype(str).str.strip().str.upper() == branch]
    temp = temp[temp["date"].dt.month.isin([prev_month, target_month])].copy()

    if temp.empty:
        return pd.DataFrame(), None

    metrics = ["sales", "profit", "quantity"]
    rows = []

    for metric in metrics:
        prev_val = float(temp.loc[temp["date"].dt.month == prev_month, metric].sum())
        curr_val = float(temp.loc[temp["date"].dt.month == target_month, metric].sum())
        change = curr_val - prev_val
        pct_change = (change / prev_val * 100) if prev_val != 0 else 0.0

        rows.append({
            "metric": metric,
            "month_1": _period_label(prev_month, _analysis_year(df)),
            "month_2": _period_label(target_month, _analysis_year(df)),
            "value_1": round(prev_val, 2),
            "value_2": round(curr_val, 2),
            "change": round(change, 2),
            "pct_change": round(pct_change, 2),
        })

    result = pd.DataFrame(rows)

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        f'temp = temp[temp["branch"].astype(str).str.strip().str.upper() == "{branch}"]\n'
        f'temp = temp[temp["date"].dt.month.isin([{prev_month}, {target_month}])]\n'
        f'rows = []\n'
        f'for metric in ["sales", "profit", "quantity"]:\n'
        f'    prev_val = float(temp.loc[temp["date"].dt.month == {prev_month}, metric].sum())\n'
        f'    curr_val = float(temp.loc[temp["date"].dt.month == {target_month}, metric].sum())\n'
        f'    change = curr_val - prev_val\n'
        f'    pct_change = (change / prev_val * 100) if prev_val != 0 else 0.0\n'
        f'    rows.append({{"metric": metric, "month_1": "{_period_label(prev_month, _analysis_year(df))}", "month_2": "{_period_label(target_month, _analysis_year(df))}", "value_1": round(prev_val, 2), "value_2": round(curr_val, 2), "change": round(change, 2), "pct_change": round(pct_change, 2)}})\n'
        f'result = pd.DataFrame(rows)'
    )

    return result, code

def _last_month_change_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()
    if not any(phrase in q for phrase in ["compared to last month", "what changed compared to last month", "last month"]):
        return None, None

    metric = _metric_from_question(q)
    year = _analysis_year(df)

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = _apply_filter_rules(temp, q, include_month=False)
    temp = temp[temp["date"].notna()].copy()

    if metric == "margin":
        temp = _ensure_margin_column(temp)

    months = sorted(temp["date"].dt.month.dropna().unique().tolist())
    if len(months) < 2:
        return None, None

    m1, m2 = int(months[-2]), int(months[-1])

    if metric == "margin":
        m1_df = temp[temp["date"].dt.month == m1]
        m2_df = temp[temp["date"].dt.month == m2]
        m1_sales = m1_df["sales"].sum()
        m2_sales = m2_df["sales"].sum()
        m1_profit = m1_df["profit"].sum()
        m2_profit = m2_df["profit"].sum()
        m1_val = (m1_profit / m1_sales * 100) if m1_sales != 0 else 0.0
        m2_val = (m2_profit / m2_sales * 100) if m2_sales != 0 else 0.0
    else:
        overall = temp.groupby(temp["date"].dt.month)[metric].sum()
        m1_val = float(overall.get(m1, 0.0))
        m2_val = float(overall.get(m2, 0.0))

    change = m2_val - m1_val
    pct_change = (change / m1_val * 100) if m1_val != 0 else 0.0

    rows = [{
        "segment": "Overall",
        "month_1": _period_label(m1, year),
        "month_2": _period_label(m2, year),
        "value_1": round(m1_val, 2),
        "value_2": round(m2_val, 2),
        "change": round(change, 2),
        "pct_change": round(pct_change, 2),
    }]

    for category in DIMENSION_ORDER["category"]:
        cat_temp = temp[_normalize_text_series(temp, "category") == category.lower()]
        if cat_temp.empty:
            continue

        if metric == "margin":
            c1_df = cat_temp[cat_temp["date"].dt.month == m1]
            c2_df = cat_temp[cat_temp["date"].dt.month == m2]
            c1_sales = c1_df["sales"].sum()
            c2_sales = c2_df["sales"].sum()
            c1_profit = c1_df["profit"].sum()
            c2_profit = c2_df["profit"].sum()
            c1 = (c1_profit / c1_sales * 100) if c1_sales != 0 else 0.0
            c2 = (c2_profit / c2_sales * 100) if c2_sales != 0 else 0.0
        else:
            cat_group = cat_temp.groupby(cat_temp["date"].dt.month)[metric].sum()
            c1 = float(cat_group.get(m1, 0.0))
            c2 = float(cat_group.get(m2, 0.0))

        c_change = c2 - c1
        c_pct = (c_change / c1 * 100) if c1 != 0 else 0.0
        rows.append({
            "segment": category,
            "month_1": _period_label(m1, year),
            "month_2": _period_label(m2, year),
            "value_1": round(c1, 2),
            "value_2": round(c2, 2),
            "change": round(c_change, 2),
            "pct_change": round(c_pct, 2),
        })

    result = pd.DataFrame(rows)

    if metric == "margin":
        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + '\ntemp = temp[temp["date"].notna()].copy()\n'
            + f'temp = temp[temp["date"].dt.month.isin([{m1}, {m2}])]\n'
            + f'm1_df = temp[temp["date"].dt.month == {m1}]\n'
            + f'm2_df = temp[temp["date"].dt.month == {m2}]\n'
            + 'm1_sales = m1_df["sales"].sum()\n'
            + 'm2_sales = m2_df["sales"].sum()\n'
            + 'm1_profit = m1_df["profit"].sum()\n'
            + 'm2_profit = m2_df["profit"].sum()\n'
            + 'm1_val = (m1_profit / m1_sales * 100) if m1_sales != 0 else 0.0\n'
            + 'm2_val = (m2_profit / m2_sales * 100) if m2_sales != 0 else 0.0\n'
            + 'change = m2_val - m1_val\n'
            + 'pct_change = (change / m1_val * 100) if m1_val != 0 else 0\n'
            + f'result = pd.DataFrame([{{"segment": "Overall", "month_1": "{_period_label(m1, year)}", "month_2": "{_period_label(m2, year)}", "value_1": round(m1_val, 2), "value_2": round(m2_val, 2), "change": round(change, 2), "pct_change": round(pct_change, 2)}}])'
        )
        return result, code

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        + _build_filter_code_block(q, "temp", include_month=False)
        + '\ntemp = temp[temp["date"].notna()].copy()\n'
        + f'temp = temp[temp["date"].dt.month.isin([{m1}, {m2}])]\n'
        + f'overall = temp.groupby(temp["date"].dt.month)["{metric}"].sum()\n'
        + f'm1_val = float(overall.get({m1}, 0.0))\n'
        + f'm2_val = float(overall.get({m2}, 0.0))\n'
        + 'change = m2_val - m1_val\n'
        + 'pct_change = (change / m1_val * 100) if m1_val != 0 else 0\n'
        + f'result = pd.DataFrame([{{"segment": "Overall", "month_1": "{_period_label(m1, year)}", "month_2": "{_period_label(m2, year)}", "value_1": round(m1_val, 2), "value_2": round(m2_val, 2), "change": round(change, 2), "pct_change": round(pct_change, 2)}}])'
    )
    return result, code


def _root_cause_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()
    if not any(k in q for k in ["drop", "decrease", "decline", "biggest drop", "largest drop", "caused"]):
        return None, None

    months = _extract_months_in_order(q)
    if not months:
        return None, None

    if len(months) == 1:
        target_month = months[0]
        prev_month = _previous_month(target_month)
    else:
        prev_month = months[-2]
        target_month = months[-1]

    metric = _metric_from_question(q)
    dimension = _dimension_from_question(q) or "branch"
    year = _analysis_year(df)

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = _apply_filter_rules(temp, q, include_month=False)
    temp = temp[temp["date"].dt.month.isin([prev_month, target_month])].copy()

    if metric == "margin":
        temp = _ensure_margin_column(temp)

    if temp.empty:
        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + f'\ntemp = temp[temp["date"].dt.month.isin([{prev_month}, {target_month}])]'
        )
        return pd.DataFrame(), code

    monthly = _dimension_metric_table(temp, dimension, metric)
    pivot = monthly.pivot(index=dimension, columns="month", values=metric).reset_index()
    pivot.columns.name = None

    if prev_month not in pivot.columns:
        pivot[prev_month] = 0.0
    if target_month not in pivot.columns:
        pivot[target_month] = 0.0

    pivot[prev_month] = pivot[prev_month].fillna(0)
    pivot[target_month] = pivot[target_month].fillna(0)
    pivot["change"] = pivot[target_month] - pivot[prev_month]
    pivot["pct_change"] = np.where(pivot[prev_month] != 0, (pivot["change"] / pivot[prev_month]) * 100, 0)

    result = pivot[[dimension, prev_month, target_month, "change", "pct_change"]].copy()
    for col in [prev_month, target_month, "change", "pct_change"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").round(2)

    result = result.sort_values("change").head(1).reset_index(drop=True)

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        + _build_filter_code_block(q, "temp", include_month=False)
        + '\ntemp = temp[temp["date"].notna()].copy()\n'
        + f'temp = temp[temp["date"].dt.month.isin([{prev_month}, {target_month}])]\n'
        + 'temp["month"] = temp["date"].dt.month\n'
        + ( 'temp["margin"] = np.where(temp["sales"] != 0, (temp["profit"] / temp["sales"]) * 100, np.nan)\n' if metric == "margin" else '' )
        + f'monthly = temp.groupby(["{dimension}", "month"], as_index=False)["{metric}"].sum()\n'
        + f'pivot = monthly.pivot(index="{dimension}", columns="month", values="{metric}").reset_index()\n'
        + f'pivot["change"] = pivot[{target_month}] - pivot[{prev_month}]\n'
        + f'pivot["pct_change"] = np.where(pivot[{prev_month}] != 0, (pivot["change"] / pivot[{prev_month}]) * 100, 0)\n'
        + f'result = pivot[["{dimension}", {prev_month}, {target_month}, "change", "pct_change"]].sort_values("change").head(1).reset_index(drop=True)'
    )
    return result, code


def _scalar_total_router(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()

    if "average order value" in q or "aov" in q:
        return None, None

    if not any(k in q for k in ["total", "sum", "average", "avg"]):
        return None, None

    if _grouping_phrase_present(q):
        return None, None

    metric = _metric_from_question(q)
    agg = "mean" if any(k in q for k in ["average", "avg"]) else "sum"

    temp = df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = temp[temp["date"].notna()].copy()
    temp = _apply_filter_rules(temp, q, include_month=False)

    months = _extract_months_in_order(q)
    range_words = ["from", "between", "through", "thru", "until", "till"]
    has_month_range = len(months) >= 2 and any(word in q for word in range_words)

    if has_month_range:
        start_m, end_m = months[0], months[-1]
        if start_m <= end_m:
            temp = temp[temp["date"].dt.month.between(start_m, end_m)]
        else:
            temp = temp[
                (temp["date"].dt.month >= start_m) | (temp["date"].dt.month <= end_m)
            ]
    else:
        month_num = _month_num_from_question(q)
        if month_num is not None:
            temp = temp[temp["date"].dt.month == month_num]

    if metric == "margin":
        sales_sum = temp["sales"].sum()
        profit_sum = temp["profit"].sum()
        value = (profit_sum / sales_sum * 100) if sales_sum != 0 else np.nan
        result_df = pd.DataFrame([{"result": f"{value:.2f}%"}])

        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            'temp = temp[temp["date"].notna()].copy()\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + (
                f'\n'
                f'start_m, end_m = {months[0]}, {months[-1]}\n'
                f'if start_m <= end_m:\n'
                f'    temp = temp[temp["date"].dt.month.between(start_m, end_m)]\n'
                f'else:\n'
                f'    temp = temp[(temp["date"].dt.month >= start_m) | (temp["date"].dt.month <= end_m)]\n'
                if has_month_range
                else (
                    f'\nmonth_num = {_month_num_from_question(q)}\n'
                    f'if month_num is not None:\n'
                    f'    temp = temp[temp["date"].dt.month == month_num]\n'
                )
            )
            + '\nm_sales = temp["sales"].sum()\n'
            + 'm_profit = temp["profit"].sum()\n'
            + 'm_value = (m_profit / m_sales * 100) if m_sales != 0 else np.nan\n'
            + 'result = f"{m_value:.2f}%"'
        )
        return result_df, code

    if temp.empty:
        code = (
            'temp = df.copy()\n'
            'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
            'temp = temp[temp["date"].notna()].copy()\n'
            + _build_filter_code_block(q, "temp", include_month=False)
            + (
                f'\nstart_m, end_m = {months[0]}, {months[-1]}\n'
                f'if start_m <= end_m:\n'
                f'    temp = temp[temp["date"].dt.month.between(start_m, end_m)]\n'
                f'else:\n'
                f'    temp = temp[(temp["date"].dt.month >= start_m) | (temp["date"].dt.month <= end_m)]\n'
                if has_month_range
                else (
                    f'\nmonth_num = {_month_num_from_question(q)}\n'
                    f'if month_num is not None:\n'
                    f'    temp = temp[temp["date"].dt.month == month_num]\n'
                )
            )
            + f'\nresult = temp["{metric}"].{agg}()'
        )
        return pd.DataFrame(), code

    value = temp[metric].mean() if agg == "mean" else temp[metric].sum()

    if agg == "sum" and metric in ["quantity", "orders"]:
        value = int(value)
        result_df = pd.DataFrame([{"result": value}])
    else:
        result_df = pd.DataFrame([{"result": round(float(value), 2)}])

    code = (
        'temp = df.copy()\n'
        'temp["date"] = pd.to_datetime(temp["date"], errors="coerce")\n'
        'temp = temp[temp["date"].notna()].copy()\n'
        + _build_filter_code_block(q, "temp", include_month=False)
        + (
            f'\nstart_m, end_m = {months[0]}, {months[-1]}\n'
            f'if start_m <= end_m:\n'
            f'    temp = temp[temp["date"].dt.month.between(start_m, end_m)]\n'
            f'else:\n'
            f'    temp = temp[(temp["date"].dt.month >= start_m) | (temp["date"].dt.month <= end_m)]\n'
            if has_month_range
            else (
                f'\nmonth_num = {_month_num_from_question(q)}\n'
                f'if month_num is not None:\n'
                f'    temp = temp[temp["date"].dt.month == month_num]\n'
            )
        )
        + f'\nresult = temp["{metric}"].{agg}()'
    )
    return result_df, code


def _rule_based_answer_and_code(df: pd.DataFrame, question: str):
    q = normalize_query(question).lower()

    for router in (
        _branch_last_month_change_router,
        _aov_router,
        _month_compare_router,
        _last_month_change_router,
        _root_cause_router,
        _top_bottom_router,
        _trend_router,
        _group_by_router,
        _scalar_total_router,
        ):

        result_df, code = router(df, q)
        if result_df is not None and not result_df.empty:
            return result_df, code

    return None, None


def _apply_fallbacks(df: pd.DataFrame, question: str, result_df: pd.DataFrame):
    if not result_df.empty:
        return result_df, None

    fallback_df, fallback_code = _rule_based_answer_and_code(df, question)
    if fallback_df is not None and not fallback_df.empty:
        return fallback_df, fallback_code

    return result_df, None


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    df = DATASTORE.get(req.file_id)
    if df is None:
        raise HTTPException(404, "Upload again")

    try:
        clean_q = normalize_query(req.question)

        direct_result, direct_code = _rule_based_answer_and_code(df, clean_q)

        if direct_result is not None and not direct_result.empty:
            result_df = direct_result
            code = direct_code or "# rule-based answer router used"
            analysis_model = req.model
            meta = {"error": None, "traceback": None}
        else:
            msgs = build_code_prompt(clean_q, df.head().to_string())
            code, analysis_model = _generate_code(msgs, _models_to_try(req.model))

            result, meta = execute_pandas_code_in_sandbox(code, df)

            if isinstance(result, (int, float, str, np.integer, np.floating, np.bool_)):
                if isinstance(result, (float, np.floating)):
                    scalar_value = round(float(result), 2)
                elif isinstance(result, (np.integer, np.bool_)):
                    scalar_value = int(result)
                else:
                    scalar_value = result
                result_df = pd.DataFrame([{"result": scalar_value}])
            else:
                result_df = result_to_dataframe(result)

            result_df, fallback_code = _apply_fallbacks(df, clean_q, result_df)
            if fallback_code:
                code = fallback_code

        if result_df.empty:
            return {
                "success": False,
                "generated_code": code,
                "error": meta.get("error") or "Generated code returned an empty result.",
                "traceback": meta.get("traceback"),
            }

        result_df = result_df.copy()
        for col in result_df.columns:
            if pd.api.types.is_float_dtype(result_df[col]):
                result_df[col] = result_df[col].round(2)

        answer_text = result_df.to_string(index=False)

        try:
            analysis = chat_completion(
                analysis_model,
                build_explanation_prompt(clean_q, code, answer_text),
                temperature=0.3,
            )
            if not analysis or not analysis.strip():
                analysis = answer_text
        except Exception:
            analysis = answer_text

        try:
            rec = generate_recommendation(clean_q, result_df, df)
            if not rec or not rec.strip():
                rec = "No recommendation."
        except Exception:
            rec = "No recommendation."

        return {
            "success": True,
            "generated_code": code,
            "answer_preview": answer_text,
            "analysis": analysis,
            "recommendation": rec,
            "result_table": result_df.to_dict(orient="records"),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))