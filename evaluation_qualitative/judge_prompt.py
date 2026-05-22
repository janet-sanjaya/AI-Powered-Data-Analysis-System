def build_judge_prompt(question, answer, analysis, recommendation):
    return [
        {
            "role": "system",
            "content": """
You are an expert evaluator.

Evaluate the quality of analysis and recommendation.

Criteria:
1. Correctness (based on answer)
2. Clarity
3. Business usefulness
4. Logical reasoning

Give:
- score_analysis (1–5)
- score_recommendation (1–5)
- short explanation

Return JSON only.
"""
        },
        {
            "role": "user",
            "content": f"""
Question:
{question}

Answer:
{answer}

Analysis:
{analysis}

Recommendation:
{recommendation}

Evaluate now.
"""
        }
    ]