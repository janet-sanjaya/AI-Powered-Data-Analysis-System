from pathlib import Path
import sys
import pandas as pd
import json
import time
import re

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm import chat_completion
from judge_prompt import build_judge_prompt

# Paths
EVAL_RESULTS_PATH = PROJECT_ROOT / "evaluation_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "judge_results.csv"


# 🎯 Judge logic
# GPT → Claude
# Others → GPT
def pick_judge_model(eval_model: str) -> str:
    model = str(eval_model).lower()

    if "openai" in model or "gpt" in model:
        return "anthropic/claude-sonnet-4.5"

    return "openai/gpt-4o-mini"


# 🧠 Robust JSON extraction (VERY IMPORTANT)
def extract_json(text: str):
    if not text:
        return None

    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # Try extracting JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None

    return None


# Load evaluation results
df = pd.read_csv(EVAL_RESULTS_PATH)

results = []

for i, row in df.iterrows():
    question = row.get("question", "")
    answer = row.get("answer", row.get("answer_preview", ""))
    analysis = row.get("analysis", "")
    recommendation = row.get("recommendation", "")
    eval_model = row.get("model", "")

    judge_model = pick_judge_model(eval_model)

    print(f"Evaluating row {i} | model={eval_model} | judge={judge_model}")

    try:
        messages = build_judge_prompt(
            question,
            answer,
            analysis,
            recommendation
        )

        response = chat_completion(judge_model, messages)

        parsed = extract_json(response)

        if not parsed:
            raise ValueError("Invalid JSON returned by model")

        results.append({
            "model": eval_model,
            "judge_model": judge_model,
            "question": question,
            "score_analysis": parsed.get("score_analysis"),
            "score_recommendation": parsed.get("score_recommendation"),
            "comment": parsed.get("explanation"),
        })

    except Exception as e:
        print("Error:", e)

        results.append({
            "model": eval_model,
            "judge_model": judge_model,
            "question": question,
            "score_analysis": None,
            "score_recommendation": None,
            "comment": "",
        })

    time.sleep(0)


# Save results
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_PATH, index=False)

print(f"Done -> {OUTPUT_PATH}")