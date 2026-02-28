import os
import argparse
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def add_language_prefix(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    df = df.copy()
    df["input_text"] = "[LANG={}] ".format(lang_code) + df["text"].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Task number: 1 (polarization), 2 (types), 3 (manifestations).",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help=(
            "Name of the label column to color by. "
            "For task1: usually 'polarization'. "
            "For task2: e.g. 'political'. "
            "For task3: e.g. 'stereotype'."
        ),
    )
    parser.add_argument("--languages",type=str,nargs="+",required=True,help="List of language codes, e.g. eng spa deu",)
    parser.add_argument("--train_dir",type=str,default="../dev_phase/subtask3/train/")
    parser.add_argument("--model_name",type=str,default="xlm-roberta-large")
    parser.add_argument("--max_length",type=int,default=128)
    parser.add_argument("--max_per_lang",type=int,default=500,help="Max number of examples per language to embed.")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--seed",type=int,default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    if args.label_column is None:
        if args.task == 1:
            label_col = "polarization"
        elif args.task == 2:
            label_col = "political"
        else:  # task 3
            label_col = "stereotype"
        print(f"No --label_column given, defaulting to '{label_col}' for task {args.task}.")
    else:
        label_col = args.label_column

    print(f"Using label column: {label_col}")

    print(f"Loading model from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer)

    all_embeddings = []
    all_lang_labels = []
    all_binary_labels = []

    for lang in args.languages:
        train_path = os.path.join(args.train_dir, f"{lang}.csv")
        df = pd.read_csv(train_path)

        labels = df[label_col].astype(int).tolist()

        df = add_language_prefix(df, lang)
        texts = df["input_text"].tolist()

        if len(texts) > args.max_per_lang:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(texts), size=args.max_per_lang, replace=False)
            texts = [texts[i] for i in idx]
            labels = [labels[i] for i in idx]

        print(f"[{lang}] using {len(texts)} examples")

        dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        lang_embs = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                last_hidden = outputs.last_hidden_state 
                cls_embeddings = last_hidden[:, 0, :]   
                lang_embs.append(cls_embeddings.cpu().numpy())

        lang_embs = np.concatenate(lang_embs, axis=0) 

        all_embeddings.append(lang_embs)
        all_lang_labels.extend([lang] * lang_embs.shape[0])
        all_binary_labels.extend(labels)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_lang_labels = np.array(all_lang_labels)
    all_binary_labels = np.array(all_binary_labels)

    print("Total embeddings shape:", all_embeddings.shape)

    print("Running PCA to 2D...")
    pca = PCA(n_components=2, random_state=args.seed)
    emb_2d = pca.fit_transform(all_embeddings)

    explained = pca.explained_variance_ratio_
    print(f"Explained variance by PC1: {explained[0]:.4f}")
    print(f"Explained variance by PC2: {explained[1]:.4f}")
    print(f"Total explained variance (PC1 + PC2): {explained.sum():.4f}")

    unique_langs = sorted(set(all_lang_labels))
    n_langs = len(unique_langs)
    n_cols = 4
    n_rows = int(np.ceil(n_langs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax_idx, lang in enumerate(unique_langs):
        ax = axes[ax_idx]
        mask_lang = all_lang_labels == lang

        xs = emb_2d[mask_lang, 0]
        ys = emb_2d[mask_lang, 1]
        lbls = all_binary_labels[mask_lang]

        # negatives = 0 (grey), positives = 1 (red)
        mask_neg = lbls == 0
        mask_pos = lbls == 1

        if np.any(mask_neg):
            ax.scatter(
                xs[mask_neg],
                ys[mask_neg],
                s=8,
                alpha=0.3,
                color="lightgrey",
                label=f"{label_col}=0",
            )

        if np.any(mask_pos):
            ax.scatter(
                xs[mask_pos],
                ys[mask_pos],
                s=15,
                alpha=0.9,
                color="red",
                label=f"{label_col}=1",
            )

        ax.set_title(lang)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(unique_langs):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.suptitle(
        f"Task {args.task} – embeddings by language\n"
        f"Color = {label_col} (0 grey, 1 red), PCA total explained variance = {explained.sum():.4f}",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
