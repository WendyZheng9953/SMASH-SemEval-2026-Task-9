import argparse
import os
import numpy as np
import pandas as pd


LABELS_T2 = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
LABELS_T3 = ["stereotype", "vilification", "dehumanization",
             "extreme_language", "lack_of_empathy", "invalidation"]


def parse_logits_cell(x: str) -> np.ndarray:
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)
    s = str(x).strip()
    if len(s) == 0:
        return np.array([], dtype=float)
    s = s.strip("[]")
    return np.fromstring(s, sep=",", dtype=float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax_prob_class1(logits2: np.ndarray) -> np.ndarray:
    a = logits2[:, 0].astype(float)
    b = logits2[:, 1].astype(float)
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    return eb / (ea + eb)


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def macro_f1_binary(y: np.ndarray, y_pred: np.ndarray) -> float:
    y = y.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)

    tp = int(np.sum((y == 1) & (y_pred == 1)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))

    f1_pos = f1_from_counts(tp, fp, fn)
    f1_neg = f1_from_counts(tn, fn, fp)
    return 0.5 * (f1_pos + f1_neg)


def infer_label_columns(gold_df: pd.DataFrame) -> list[str]:
    exclude = {"id", "text", "language", "input_text", "logits"}
    cols = [c for c in gold_df.columns if c not in exclude]
    label_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(gold_df[c]):
            label_cols.append(c)
        else:
            coerced = pd.to_numeric(gold_df[c], errors="coerce")
            if coerced.notna().any():
                label_cols.append(c)
    return label_cols


def label_cols_for_task(task: int, gold_df: pd.DataFrame) -> list[str]:
    if task == 2:
        return [c for c in LABELS_T2 if c in gold_df.columns]
    if task == 3:
        return [c for c in LABELS_T3 if c in gold_df.columns]
    return infer_label_columns(gold_df)


def read_pred_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["id", "logits"]].copy()
    df["logits"] = df["logits"].map(parse_logits_cell)
    return df


def avg_seed_logits_mono_fold(mono_base_dir: str, task: int, lang: str, fold: int, seeds: list[int]) -> pd.DataFrame:
    dfs = []
    for s in seeds:
        p = os.path.join(mono_base_dir, f"task{task}_kfold_{s}", "mono", lang, f"fold_{fold}_val.csv")
        dfs.append(read_pred_csv(p))

    merged = dfs[0].rename(columns={"logits": "logits_0"})
    for i, d in enumerate(dfs[1:], start=1):
        d = d.rename(columns={"logits": f"logits_{i}"})
        merged = merged.merge(d, on="id", how="inner")

    log_cols = [c for c in merged.columns if c.startswith("logits_")]
    stack = np.stack([np.vstack(merged[c].to_numpy()) for c in log_cols], axis=0)
    avg = np.mean(stack, axis=0)

    out = merged[["id"]].copy()
    out["logits_mono"] = list(avg)
    return out


def avg_seed_logits_multi_fold(multi_base_dir: str, task: int, multi_model: str, lang: str, fold: int, seeds: list[int]) -> pd.DataFrame:
    dfs = []
    for s in seeds:
        p = os.path.join(
            multi_base_dir,
            f"t{task}_seed{s}",
            f"fold_{fold}_{multi_model}_seed{s}",
            f"subtask{task}_kfold",
            lang,
            f"fold_{fold}_val.csv",
        )
        dfs.append(read_pred_csv(p))

    merged = dfs[0].rename(columns={"logits": "logits_0"})
    for i, d in enumerate(dfs[1:], start=1):
        d = d.rename(columns={"logits": f"logits_{i}"})
        merged = merged.merge(d, on="id", how="inner")

    log_cols = [c for c in merged.columns if c.startswith("logits_")]
    stack = np.stack([np.vstack(merged[c].to_numpy()) for c in log_cols], axis=0)
    avg = np.mean(stack, axis=0)

    out = merged[["id"]].copy()
    out["logits_multi"] = list(avg)
    return out


def grid_search_task1(logits: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> dict:
    if logits.ndim == 2 and logits.shape[1] == 2:
        p = softmax_prob_class1(logits)
    else:
        p = sigmoid(logits.reshape(-1))

    best = {"t": None, "macro_f1": -1.0}
    for t in t_grid:
        m = macro_f1_binary(y, (p >= t).astype(int))
        if m > best["macro_f1"]:
            best = {"t": float(t), "macro_f1": float(m)}
    return best


def grid_search_multilabel(logits: np.ndarray, y_true: np.ndarray, t_grid: np.ndarray) -> dict:
    p = sigmoid(logits)
    L = y_true.shape[1]
    best_t = np.zeros(L, dtype=float)
    best_f1 = np.zeros(L, dtype=float)

    for j in range(L):
        yj = y_true[:, j].astype(int)
        pj = p[:, j]
        bj_t = None
        bj_f1 = -1.0

        for t in t_grid:
            ypj = (pj >= t).astype(int)
            tp = int(np.sum((yj == 1) & (ypj == 1)))
            fp = int(np.sum((yj == 0) & (ypj == 1)))
            fn = int(np.sum((yj == 1) & (ypj == 0)))
            f1j = f1_from_counts(tp, fp, fn)
            if f1j > bj_f1:
                bj_f1 = f1j
                bj_t = float(t)

        best_t[j] = bj_t
        best_f1[j] = bj_f1

    return {"t_per_label": best_t, "f1_per_label": best_f1, "macro_f1": float(np.mean(best_f1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--langs", type=str, nargs="+", required=True)
    ap.add_argument("--mono_base_dir", type=str, required=True)
    ap.add_argument("--multi_base_dir", type=str, required=True)
    ap.add_argument("--multi_model", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 111])

    ap.add_argument("--w_step", type=float, default=0.05)

    ap.add_argument("--t_min", type=float, default=0.05)
    ap.add_argument("--t_max", type=float, default=0.95)
    ap.add_argument("--t_step", type=float, default=0.01)

    ap.add_argument("--save_path", type=str, required=True)
    args = ap.parse_args()

    t_grid = np.arange(args.t_min, args.t_max + 1e-12, args.t_step, dtype=float)
    w_grid = np.arange(0.0, 1.0 + 1e-12, args.w_step, dtype=float)

    rows = []

    for lang in args.langs:
        mono_all = []
        multi_all = []
        y_all = []
        label_cols_final = None

        for k in range(args.folds):
            gold_path = os.path.join(args.data_dir, lang, f"fold_{k}", "val.csv")
            gold_df = pd.read_csv(gold_path)
            label_cols = label_cols_for_task(args.task, gold_df)
            label_cols_final = label_cols_final or label_cols

            merged = gold_df[["id"] + label_cols].copy()

            mono_df = avg_seed_logits_mono_fold(args.mono_base_dir, args.task, lang, k, args.seeds)
            multi_df = avg_seed_logits_multi_fold(args.multi_base_dir, args.task, args.multi_model, lang, k, args.seeds)

            merged = merged.merge(mono_df, on="id", how="inner")
            merged = merged.merge(multi_df, on="id", how="inner")

            mono_all.append(np.vstack(merged["logits_mono"].to_numpy()))
            multi_all.append(np.vstack(merged["logits_multi"].to_numpy()))
            y_all.append(merged[label_cols].to_numpy(dtype=int))

        mono_logits = np.vstack(mono_all)
        multi_logits = np.vstack(multi_all)
        y_true = np.vstack(y_all)

        best_pack = None

        for w_mono in w_grid:
            w_multi = 1.0 - float(w_mono)
            ens_logits = w_mono * mono_logits + w_multi * multi_logits

            if args.task == 1:
                yb = y_true.reshape(-1).astype(int)
                bt = grid_search_task1(ens_logits, yb, t_grid)
                score = bt["macro_f1"]
                pack = {
                    "language": lang,
                    "macro_f1": score,
                    "w_mono": float(w_mono),
                    "w_multi": float(w_multi),
                    "t": float(bt["t"]),
                }
            else:
                bt = grid_search_multilabel(ens_logits, y_true, t_grid)
                score = bt["macro_f1"]
                pack = {
                    "language": lang,
                    "macro_f1": score,
                    "w_mono": float(w_mono),
                    "w_multi": float(w_multi),
                }
                for name, t in zip(label_cols_final, bt["t_per_label"]):
                    pack[f"t_{name}"] = float(t)

            if best_pack is None or score > best_pack["macro_f1"]:
                best_pack = pack

        rows.append(best_pack)
        print(f"{lang} | best_macro_f1={best_pack['macro_f1']:.4f} | w_mono={best_pack['w_mono']:.2f} w_multi={best_pack['w_multi']:.2f}")

    out_df = pd.DataFrame(rows)
    cols = ["language", "w_mono", "w_multi"] + [c for c in out_df.columns if c == "t" or c.startswith("t_")] + ["macro_f1"]
    cols = [c for c in cols if c in out_df.columns] + [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols]

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    out_df.to_csv(args.save_path, index=False)
    print(f"Saved: {args.save_path}")


if __name__ == "__main__":
    main()
