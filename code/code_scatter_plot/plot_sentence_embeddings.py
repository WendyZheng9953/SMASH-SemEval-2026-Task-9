import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


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
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        return item

def add_language_prefix(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    df = df.copy()
    df["input_text"] = "[LANG={}] ".format(lang_code) + df["text"].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages",type=str,nargs="+",required=True,help="List of language codes, e.g. eng spa deu",)
    parser.add_argument("--train_dir",type=str,default="../../../dev_phase/subtask1/train/")
    parser.add_argument("--model_name",type=str,default="xlm-roberta-large")
    parser.add_argument("--checkpoint_path",type=str,default=None,help="Path to fine-tuned checkpoint.")
    parser.add_argument("--max_length",type=int,default=128)
    parser.add_argument("--max_per_lang",type=int,default=500,help="Max number of examples per language to embed.")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--dim_reduction",type=str,choices=["pca", "tsne"],default="pca",help="Dimensionality reduction method for plotting: 'pca' or 'tsne'.")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    model_path = args.checkpoint_path if args.checkpoint_path is not None else args.model_name
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer)

    all_embeddings = []
    all_lang_labels = []

    for lang in args.languages:
        train_path = os.path.join(args.train_dir, f"{lang}.csv")

        df = pd.read_csv(train_path)

        df = add_language_prefix(df, lang)

        texts = df["input_text"].tolist()

        # subsample if large
        if len(texts) > args.max_per_lang:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(texts), size=args.max_per_lang, replace=False)
            texts = [texts[i] for i in idx]

        print(f"[{lang}] using {len(texts)} examples for embeddings")

        dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=data_collator,
        )

        lang_embs = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                # output_hidden_states=True to get full stack
                outputs = model(**batch, output_hidden_states=True)
                # last hidden layer, CLS token is index 0
                last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
                cls_embeddings = last_hidden[:, 0, :]    # (batch, hidden)
                lang_embs.append(cls_embeddings.cpu().numpy())

        lang_embs = np.concatenate(lang_embs, axis=0)  # (N_lang, hidden_dim)
        all_embeddings.append(lang_embs)
        all_lang_labels.extend([lang] * lang_embs.shape[0])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_lang_labels = np.array(all_lang_labels)
    print("Total embeddings shape:", all_embeddings.shape)

    if args.dim_reduction == "pca":
        print("Running PCA to 2D...")
        pca = PCA(n_components=2, random_state=args.seed)
        emb_2d = pca.fit_transform(all_embeddings)

        explained = pca.explained_variance_ratio_
        print(f"Explained variance by PC1: {explained[0]:.4f}")
        print(f"Explained variance by PC2: {explained[1]:.4f}")
        print(f"Total explained variance (PC1 + PC2): {explained.sum():.4f}")

        title_suffix = f"PCA (2D, {explained.sum():.2%} variance)"

    elif args.dim_reduction == "tsne":
        print("Running PCA to 50D as a preprocessing step for t-SNE...")
        pca_50 = PCA(n_components=50, random_state=args.seed)
        emb_50 = pca_50.fit_transform(all_embeddings)
        print("Running t-SNE to 2D (this can take a while)...")

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            init="pca",
            random_state=args.seed,
            verbose=1,
        )
        emb_2d = tsne.fit_transform(emb_50)
        title_suffix = "t-SNE (2D on PCA-50)"

    plt.figure(figsize=(8, 6))
    unique_langs = sorted(set(all_lang_labels))
    num_langs = len(unique_langs)
    cmap = get_cmap("tab20", num_langs)
    lang_to_color = {lang: cmap(i) for i, lang in enumerate(unique_langs)}

    for lang in unique_langs:
        mask = (all_lang_labels == lang)
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            alpha=0.7,
            label=lang,
            s=10,
            color=lang_to_color[lang],
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{args.model_name} Sentence Embeddings – {title_suffix}")
    plt.legend(markerscale=2, fontsize="small")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
