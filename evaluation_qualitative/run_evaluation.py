from pathlib import Path
import sys
import pandas as pd
import requests
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_URL = "http://127.0.0.1:8002/analyze"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation_question" / "evaluation_questions.csv"
OUTPUT_PATH = PROJECT_ROOT / "evaluation_results.csv"

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
]

FILE_ID = input("Paste the file_id from your upload: ").strip()

questions_df = pd.read_csv(QUESTIONS_PATH)

results = []

for model in MODELS:
    print(f"\nRunning model: {model}")

    for i, row in questions_df.iterrows():
        question = row["question"]
        q_type = row.get("type", "")
        subtype = row.get("subtype", "")

        print(f"[{i+1}/{len(questions_df)}] {question}")

        try:
            response = requests.post(
                API_URL,
                json={
                    "file_id": FILE_ID,
                    "question": question,
                    "model": model,
                },
                timeout=120,
            )

            data = response.json()

            results.append({
                "model": model,
                "question": question,
                "type": q_type,
                "subtype": subtype,
                "success": data.get("success", False),
                "answer_preview": data.get("answer_preview", ""),
                "analysis": data.get("analysis", ""),
                "recommendation": data.get("recommendation", ""),
                "generated_code": data.get("generated_code", ""),
                "error": data.get("error", ""),
            })

        except Exception as e:
            results.append({
                "model": model,
                "question": question,
                "type": q_type,
                "subtype": subtype,
                "success": False,
                "answer_preview": "",
                "analysis": "",
                "recommendation": "",
                "generated_code": "",
                "error": str(e),
            })

        time.sleep(1)

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Saved to {OUTPUT_PATH}")