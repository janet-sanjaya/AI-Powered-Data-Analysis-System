import os
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in your environment.")

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "AInsight"),
        },
    )


def chat_completion(model: str, messages: list, temperature: float = 0.0, max_tokens: int = 1200) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


def build_code_prompt(question: str, dataframe_preview: str) -> list:
    system_msg = """
You are an expert data analyst writing Pandas code for a business analytics system.

STRICT RULES:
- Return ONLY Python code.
- Do NOT explain anything.
- Do NOT use markdown fences.
- Do NOT use imports.
- Do NOT use os, sys, subprocess, requests, eval, exec, open, or any network calls.
- Use ONLY the dataframe named df.
- Assign the final answer to a variable named result.
- The result must directly answer the question.
- Do NOT guess or fabricate values.
- Do NOT reuse values from other questions.
- If data is missing, return a grounded result instead of guessing.

IMPORTANT DATA ASSUMPTIONS:
- df["date"] is already a datetime column.
- Numeric columns are already converted.
- Text columns may contain inconsistent casing or whitespace.
- Ignore rows where branch == "Unknown" unless explicitly requested.

OUTPUT REQUIREMENTS:
- Output must be either:
  (1) a scalar value (int/float/string), OR
  (2) a clean DataFrame
- Always use reset_index() after groupby.
- Do NOT return multi-index results.
- Output must be clean and directly interpretable.

DATA HANDLING RULES:
- Always treat string filters as case-insensitive:
  use .str.lower().str.strip()
- Always strip whitespace when filtering text columns
- Use sales for revenue-type questions

CRITICAL MONTH RULES:
- If a month is mentioned (e.g., May), map it correctly:
  January=1, February=2, ..., December=12
- NEVER substitute one month for another
- NEVER reuse a value from another month
- ALWAYS compute from df

LOGIC RULES:

AGGREGATION:
- "total" → sum
- "average" → mean
- Apply filters first, then aggregate

FILTERING:
- Apply all filters explicitly:
  category, region, branch, month

COMPARISON:
- Use groupby + sum
- Use sort_values(descending)
- Return top result or full sorted table

TREND:
- Use df["date"].dt.month or df["date"].dt.to_period("M")
- Aggregate by time

ROOT CAUSE:
- Compare change across groups
- Return group with largest negative change

MONTH-OVER-MONTH:
1. Create month column: df["date"].dt.month
2. Filter requested months
3. Group by branch + month
4. Sum sales
5. Pivot month into columns
6. Compute change and pct_change

QUESTION INTERPRETATION:
1. Identify the requested metric (sales, profit, quantity, etc.).
2. Identify filters (month, region, category, branch).
3. Identify aggregation (sum, average, comparison, trend).
4. Build the simplest correct Pandas code.

IMPORTANT SIMPLIFICATION RULE:
- If the question is a simple aggregation (e.g., "total sales in May"):
  → filter first → then compute directly (no complex logic)
- Do NOT create unnecessary columns
- Do NOT use pivot/groupby unless required
- Prefer direct filtering + sum/mean when possible

EXAMPLES (VERY IMPORTANT):

Q: What is total sales in May?
→ df[df["date"].dt.month == 5]["sales"].sum()

Q: What is total sales for electronics in west region?
→ df[
    (df["category"].astype(str).str.strip().str.lower() == "electronics") &
    (df["region"].astype(str).str.strip().str.lower() == "west")
  ]["sales"].sum()

Q: Which branch has highest total sales?
→ df.groupby("branch")["sales"].sum().reset_index().sort_values("sales", ascending=False).head(1)

Q: Which branch caused the biggest drop in September?
→ compute month-to-month change and return minimum

Q: Show sales trend over time
→ df.groupby(df["date"].dt.month)["sales"].sum().reset_index()

COMMON SIMPLE PATTERNS:

- Total sales in a month:
  df[df["date"].dt.month == X]["sales"].sum()

- Total sales with filters:
  df[(condition1) & (condition2)]["sales"].sum()

- Total profit:
  df["profit"].sum()

- Total quantity:
  df["quantity"].sum()

IMPORTANT:
- ALWAYS compute from df
- NEVER reuse numbers
- NEVER hallucinate missing values
"""

    user_msg = f"""
Data preview:
{dataframe_preview}

Question:
{question}

Write the Pandas code now.
"""

    return [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": user_msg.strip()},
    ]


def build_explanation_prompt(question: str, code: str, result_text: str) -> list:
    system_msg = """
You explain analytical results clearly.

RULES:
- Use ONLY the given result
- Do NOT add new facts
- Do NOT guess
- Be concise and business-focused
- If result is empty, say so clearly
"""

    user_msg = f"""
Question:
{question}

Code:
{code}

Result:
{result_text}

Explain in 2-3 sentences.
"""

    return [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": user_msg.strip()},
    ]


def build_recommendation_prompt(question: str, answer_text: str, dataframe_summary: str) -> list:
    system_msg = """
You are a business analyst.

RULES:
- Base recommendations ONLY on the result
- Do NOT invent data
- Keep it practical and specific
- If data is limited, say so
"""

    user_msg = f"""
Question:
{question}

Answer:
{answer_text}

Data summary:
{dataframe_summary}

Provide a recommendation (2-4 sentences).
"""

    return [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": user_msg.strip()},
    ]