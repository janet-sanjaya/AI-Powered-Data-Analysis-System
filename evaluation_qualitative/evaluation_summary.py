import pandas as pd

df = pd.read_csv("judge_results.csv")

# --- Per model average ---
model_avg = df.groupby("model")[["score_analysis", "score_recommendation"]].mean()

print("=== Average Scores by Model ===")
for model, row in model_avg.iterrows():
    print(f"\nModel: {model}")
    print(f"  Avg Analysis Score: {row['score_analysis']:.2f}")
    print(f"  Avg Recommendation Score: {row['score_recommendation']:.2f}")

# --- Overall average ---
overall_analysis = df["score_analysis"].mean()
overall_recommendation = df["score_recommendation"].mean()

print("\n=== Overall Average ===")
print(f"Avg Analysis Score: {overall_analysis:.2f}")
print(f"Avg Recommendation Score: {overall_recommendation:.2f}")