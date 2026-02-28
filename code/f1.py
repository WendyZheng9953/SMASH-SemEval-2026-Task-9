import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report


LABELS_T2 = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
LABELS_T3 = [
    "stereotype", "vilification", "dehumanization",
    "extreme_language", "lack_of_empathy", "invalidation"
]


def load_pair(gold_dir: Path, pred_dir: Path, lang: str):
    gold_path = gold_dir / f"{lang}.csv"
    pred_path = pred_dir / f"pred_{lang}.csv"

    gold_df = pd.read_csv(gold_path)
    pred_df = pd.read_csv(pred_path)

    merged = gold_df.merge(pred_df, on="id", how="inner", suffixes=("_gold", "_pred"))
    return merged, gold_path, pred_path


def task_macro_f1(task: int, merged: pd.DataFrame) -> float:
    if task == 1:
        col = "polarization"
        y_true = merged[f"{col}_gold"].astype(int).to_numpy()
        y_pred = merged[f"{col}_pred"].astype(int).to_numpy()
        return float(f1_score(y_true, y_pred, average="macro"))

    if task == 2:
        cols = LABELS_T2
        y_true = merged[[f"{c}_gold" for c in cols]].astype(int).to_numpy()
        y_pred = merged[[f"{c}_pred" for c in cols]].astype(int).to_numpy()
        return float(f1_score(y_true, y_pred, average="macro"))

    cols = LABELS_T3
    y_true = merged[[f"{c}_gold" for c in cols]].astype(int).to_numpy()
    y_pred = merged[[f"{c}_pred" for c in cols]].astype(int).to_numpy()
    return float(f1_score(y_true, y_pred, average="macro"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--langs", type=str, nargs="+", required=True)
    p.add_argument("--gold", type=str, required=True)
    p.add_argument("--pred", type=str, required=True)
    args = p.parse_args()

    gold_dir = Path(args.gold)
    pred_dir = Path(args.pred)
    scores = []

    for lang in args.langs:
        merged, gold_path, pred_path = load_pair(gold_dir, pred_dir, lang)
        f1m = task_macro_f1(args.task, merged)
        scores.append(f1m)

        print(f"\n=== {lang} ===")
        print(f"F1 macro: {f1m:.6f}")

        # if args.task == 1:
        #     col = "polarization"
        #     y_true = merged[f"{col}_gold"].astype(int).to_numpy()
        #     y_pred = merged[f"{col}_pred"].astype(int).to_numpy()
        #     print(classification_report(y_true, y_pred, digits=4))
        
        if args.task == 2:
            cols = LABELS_T2
            y_true = merged[[f"{c}_gold" for c in cols]].astype(int).to_numpy()
            y_pred = merged[[f"{c}_pred" for c in cols]].astype(int).to_numpy()
            for i, c in enumerate(cols):
                f1c = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
                print(f"{c:15s} F1: {f1c:.4f}")
        elif args.task == 3:
            cols = LABELS_T3
            y_true = merged[[f"{c}_gold" for c in cols]].astype(int).to_numpy()
            y_pred = merged[[f"{c}_pred" for c in cols]].astype(int).to_numpy()
            
            for i, c in enumerate(cols):
                f1c = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
                print(f"{c:15s} F1: {f1c:.4f}")



if __name__ == "__main__":
    main()
