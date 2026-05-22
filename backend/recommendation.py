import pandas as pd
from backend.llm import chat_completion, build_recommendation_prompt
import os


def result_to_text(result) -> str:
    """
    Convert result into readable text for the LLM.
    """
    if result is None:
        return "No result available."

    if isinstance(result, pd.DataFrame):
        return result.head(10).to_string(index=False)

    if isinstance(result, pd.Series):
        return result.to_string()

    return str(result)


def generate_recommendation(question: str, result, df: pd.DataFrame) -> str:
    """
    Generate business recommendation using LLM.
    """
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    answer_text = result_to_text(result)

    # Basic dataset summary
    summary = f"""
Rows: {len(df)}
Columns: {list(df.columns)}
Total Sales: {df['sales'].sum():.2f}
Total Profit: {df['profit'].sum():.2f}
"""

    messages = build_recommendation_prompt(
        question=question,
        answer_text=answer_text,
        dataframe_summary=summary
    )

    response = chat_completion(model, messages, temperature=0.3)

    return response