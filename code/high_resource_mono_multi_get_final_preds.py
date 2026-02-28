import argparse
import os
import numpy as np
import pandas as pd


LABELS_T2 = ["political", "racial/ethnic", "religious", "gender/sexual", "other"]
LABELS_T3 = ["stereotype", "vilification", "dehumanization",
             "extreme_language", "lack_of_empathy", "invalidation"]


def parse_logits_cell(x):
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)
    s = str(x).strip().strip("[]")
    if s == "":
        return np.array([], dtype=float)
    return np.fromstring(s, sep=",", dtype=float)


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax_prob_class1(logits2):
    a = logits2[:, 0].astype(float)
    b = logits2[:, 1].astype(float)
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    return eb / (ea + eb)


def load_logits_csv(path):
    df = pd.read_csv(path)
    arr = np.vstack(df["logits"].map(parse_logits_cell).to_numpy())
    return df[["id"]].copy(), arr


def average_seed_logits(paths):
    base_ids, base_logits = load_logits_csv(paths[0])
    merged = base_ids.copy()
    merged["__idx__"] = np.arange(len(merged))
    logits_list = [base_logits]

    for p in paths[1:]:
        ids, logits = load_logits_csv(p)
        tmp = ids.copy()
        tmp["__idx2__"] = np.arange(len(tmp))
        m = merged.merge(tmp, on="id", how="inner")
        idx1 = m["__idx__"].to_numpy()
        idx2 = m["__idx2__"].to_numpy()
        merged = m[["id"]].copy()
        merged["__idx__"] = np.arange(len(merged))
        logits_list = [L[idx1] for L in logits_list] + [logits[idx2]]

    avg_logits = np.mean(np.stack(logits_list, axis=0), axis=0)
    return merged[["id"]].copy(), avg_logits


def resolve_output_path(out_path, lang):
    if out_path.endswith(os.sep) or (os.path.exists(out_path) and os.path.isdir(out_path)) or (not out_path.lower().endswith(".csv")):
        os.makedirs(out_path, exist_ok=True)
        return os.path.join(out_path, f"pred_{lang}.csv")
    if "{lang}" in out_path:
        p = out_path.format(lang=lang)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        return p
    base, ext = os.path.splitext(out_path)
    p = f"{base}_{lang}{ext}"
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


def map_multi_short(name: str) -> str:
    s = str(name).lower()
    if "mdeberta" in s:
        return "mdeberta"
    if "mmbert" in s or "jhu" in s:
        return "mmbert"
    if "xlm" in s or "roberta" in s:
        return "xlm"
    return "xlm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=["1", "42", "111"])

    ap.add_argument("--mono_logits_path", type=str, required=True)
    ap.add_argument("--multi_logits_path", type=str, default="./fine-tuned_BERT/final_multi_logits/dev/task{task}_{model}_{seed}/logits_{lang}.csv")
    ap.add_argument("--multi_short", type=str, default=None, choices=["mdeberta", "mmbert", "xlm"])

    ap.add_argument("--weights_thresholds_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    args = ap.parse_args()

    WT = pd.read_csv(args.weights_thresholds_path)

    for lang in args.langs:
        row = WT[WT["language"] == lang].iloc[0].to_dict()
        w_mono = float(row["w_mono"])
        w_multi = float(row["w_multi"])

        multi_short = args.multi_short
        if multi_short is None:
            if "multi_model" in row and str(row["multi_model"]) != "nan":
                multi_short = map_multi_short(row["multi_model"])
            else:
                multi_short = "xlm"

        mono_paths = [args.mono_logits_path.format(task=args.task, lang=lang, language=lang, seed=s) for s in args.seeds]
        multi_paths = [args.multi_logits_path.format(task=args.task, model=multi_short, lang=lang, seed=s) for s in args.seeds]

        ids_m, mono_logits = average_seed_logits(mono_paths)
        ids_u, multi_logits = average_seed_logits(multi_paths)

        merged_ids = ids_m.merge(ids_u, on="id", how="inner")
        m1 = ids_m.merge(merged_ids, on="id", how="inner")
        m2 = ids_u.merge(merged_ids, on="id", how="inner")
        mono_logits = mono_logits[m1.index.to_numpy()]
        multi_logits = multi_logits[m2.index.to_numpy()]

        ens_logits = w_mono * mono_logits + w_multi * multi_logits

        if args.task == 1:
            t = float(row["t"])
            if ens_logits.ndim == 2 and ens_logits.shape[1] == 2:
                p1 = softmax_prob_class1(ens_logits)
            else:
                p1 = sigmoid(ens_logits.reshape(-1))
            pred = (p1 >= t).astype(int)
            out = pd.DataFrame({"id": merged_ids["id"].values, "polarization": pred})
        else:
            labels = LABELS_T2 if args.task == 2 else LABELS_T3
            labels = [lab for lab in labels if f"t_{lab}" in row]
            probs = sigmoid(ens_logits)
            out = pd.DataFrame({"id": merged_ids["id"].values})
            for j, lab in enumerate(labels):
                t = float(row[f"t_{lab}"])
                out[lab] = (probs[:, j] >= t).astype(int)

        out_file = resolve_output_path(args.output_path, lang)
        out.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
