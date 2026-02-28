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


def softmax_prob_index(logits2, idx):
    a = logits2[:, 0].astype(float)
    b = logits2[:, 1].astype(float)
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    denom = ea + eb
    p0 = ea / denom
    p1 = eb / denom
    return p1 if idx == 1 else p0


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


def wt_to_short_weights(row, short_models):
    w_short = {m: 0.0 for m in short_models}
    w_cols = [k for k in row.keys() if k.startswith("w_")]

    for c in w_cols:
        name = c[2:].lower()
        v = float(row.get(c, 0.0))

        if "mdeberta" in name:
            w_short["mdeberta"] += v
        elif "mmbert" in name:
            w_short["mmbert"] += v
        elif ("xlm" in name) or ("roberta" in name):
            w_short["xlm"] += v

    s = sum(w_short.values())
    if s > 0:
        for m in w_short:
            w_short[m] /= s
    return w_short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["mdeberta", "mmbert", "xlm"])
    ap.add_argument("--seeds", nargs="+", default=["1", "42", "111"])
    ap.add_argument("--logits_path", type=str, default="./fine-tuned_BERT/final_multi_logits/dev/task{task}_{model}_{seed}/logits_{lang}.csv")
    ap.add_argument("--weights_thresholds_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    args = ap.parse_args()

    WT = pd.read_csv(args.weights_thresholds_path)

    for lang in args.langs:
        row = WT[WT["language"] == lang].iloc[0].to_dict()
        w_short = wt_to_short_weights(row, args.models)

        avg_by_model = {}
        ids_df = None

        for m in args.models:
            seed_paths = []
            for s in args.seeds:
                p = args.logits_path.format(task=args.task, model=m, seed=s, lang=lang)
                seed_paths.append(p)
            ids_m, avg_logits_m = average_seed_logits(seed_paths)
            ids_df = ids_m if ids_df is None else ids_df.merge(ids_m, on="id", how="inner")
            avg_by_model[m] = avg_logits_m

        ens_logits = None
        for m in args.models:
            part = float(w_short.get(m, 0.0)) * avg_by_model[m]
            ens_logits = part if ens_logits is None else (ens_logits + part)

        if args.task == 1:
            t = float(row["t"])
            pos_index = int(row.get("pos_index", 1))
            if ens_logits.ndim == 2 and ens_logits.shape[1] == 2:
                p1 = softmax_prob_index(ens_logits, 1 if pos_index < 0 else pos_index)
            else:
                p1 = sigmoid(ens_logits.reshape(-1))
            pred = (p1 >= t).astype(int)
            out = pd.DataFrame({"id": ids_df["id"].values, "polarization": pred})
        else:
            labels = LABELS_T2 if args.task == 2 else LABELS_T3
            labels = [lab for lab in labels if f"t_{lab}" in row]
            probs = sigmoid(ens_logits)
            out = pd.DataFrame({"id": ids_df["id"].values})
            for j, lab in enumerate(labels):
                t = float(row[f"t_{lab}"])
                out[lab] = (probs[:, j] >= t).astype(int)

        out_file = resolve_output_path(args.output_path, lang)
        out.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
