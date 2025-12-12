# AB_test_runner.py
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

def run_ab(hits_a_path="hits_ncf.csv", hits_b_path="hits_xgb.csv"):
    a = pd.read_csv(hits_a_path)  # user_id, hit
    b = pd.read_csv(hits_b_path)

    df = a.merge(b, on="user_id", suffixes=("_A", "_B"))
    if df.empty:
        raise RuntimeError("No common users between A and B.")

    A = df["hit_A"].to_numpy()
    B = df["hit_B"].to_numpy()
    diff = A - B

    print("========================================")
    print(f"n_users = {len(df)}")
    print(f"A HR = {A.mean():.4f}")
    print(f"B HR = {B.mean():.4f}")
    print(f"Mean(A-B) = {diff.mean():.4f}")
    print("----------------------------------------")

    t_stat, p_t = ttest_rel(A, B)
    print(f"Paired t-test: t={t_stat:.4f}, p={p_t}")

    try:
        w_stat, p_w = wilcoxon(diff)
        print(f"Wilcoxon    : w={w_stat}, p={p_w}")
    except Exception as e:
        print("Wilcoxon failed:", e)

    print("========================================")
    df.to_csv("ab_merged_hits.csv", index=False)
    print("Saved: ab_merged_hits.csv")

if __name__ == "__main__":
    run_ab()
