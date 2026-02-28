import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)


def set_all_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    set_seed(seed_value)


LABEL_COLUMNS = [
    "stereotype", "vilification", "dehumanization",
    "extreme_language", "lack_of_empathy", "invalidation"
]
NUM_LABELS = len(LABEL_COLUMNS)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = FocalLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


class PolarizationTypeDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_vec = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label_vec, dtype=torch.float)
        return item


def add_language_prefix(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    df = df.copy()
    df["language"] = lang_code
    df["input_text"] = f"[LANG={lang_code}] " + df["text"].astype(str)
    return df


def list_kfold_languages(kfold_root: Path):
    return sorted([p.name for p in kfold_root.iterdir() if p.is_dir()])


def load_all_languages_fold_train(base_dir: Path, fold: int) -> pd.DataFrame:

    kfold_root = base_dir / "subtask3_kfold"
    if not kfold_root.exists():
        raise FileNotFoundError(f"Missing kfold folder: {kfold_root}")

    dfs = []
    for lang in list_kfold_languages(kfold_root):
        train_path = kfold_root / lang / f"fold_{fold}" / "train.csv"
        if not train_path.exists():
            print(f"[WARN] Missing: {train_path} (skipping)")
            continue

        df = pd.read_csv(train_path)
        missing = [c for c in (["text"] + LABEL_COLUMNS) if c not in df.columns]
        if missing:
            raise ValueError(f"{train_path} missing columns: {missing}")

        dfs.append(add_language_prefix(df, lang))

    if not dfs:
        raise RuntimeError(f"No training files found for fold {fold} under {kfold_root}")

    return pd.concat(dfs, ignore_index=True)


def load_all_languages_fold_val(base_dir: Path, fold: int) -> dict[str, pd.DataFrame]:

    kfold_root = base_dir / "subtask3_kfold"
    if not kfold_root.exists():
        raise FileNotFoundError(f"Missing kfold folder: {kfold_root}")

    out = {}
    for lang in list_kfold_languages(kfold_root):
        val_path = kfold_root / lang / f"fold_{fold}" / "val.csv"
        if not val_path.exists():
            print(f"[WARN] Missing: {val_path} (skipping)")
            continue

        df = pd.read_csv(val_path)
        if "text" not in df.columns:
            raise ValueError(f"{val_path} missing column: text")

        out[lang] = add_language_prefix(df, lang)

    if not out:
        raise RuntimeError(f"No val files found for fold {fold} under {kfold_root}")

    return out


def save_val_with_logits(base_dir: Path, language: str, fold: int, val_df: pd.DataFrame, logits: np.ndarray):

    out_path = base_dir / "subtask3_kfold" / language / f"fold_{fold}_val.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    val_df = val_df.copy()
    val_df["logits"] = [json.dumps(row.tolist(), ensure_ascii=False) for row in logits]
    val_df.to_csv(out_path, index=False)
    print(f"Saved logits to: {out_path}")


def run_one_fold(args, fold: int):
    base_dir = Path(args.base_dir).resolve()
    fold_seed = args.seed
    set_all_seeds(fold_seed)

    train_df = load_all_languages_fold_train(base_dir, fold)

    train_df = train_df.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)

    train_labels = train_df[LABEL_COLUMNS].to_numpy().astype(float)
    train_texts = train_df["input_text"].tolist()

    load_source = args.from_checkpoint if args.from_checkpoint else args.model_name
    print(f"\n=== Task3 | Fold={fold} | Train on ALL languages ===")
    print(f"Loading model/tokenizer from: {load_source}")
    tokenizer = AutoTokenizer.from_pretrained(load_source)

    train_dataset = PolarizationTypeDataset(
        train_texts,
        train_labels.tolist(),
        tokenizer,
        max_length=args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        load_source,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    model_name_safe = args.model_name.replace("/", "_")
    fold_out_dir = Path(args.output_dir).resolve() / f"fold_{fold}_{model_name_safe}_seed{args.seed}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(fold_out_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        seed=fold_seed,
        data_seed=fold_seed,
        disable_tqdm=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("Train size:", len(train_df))
    print("Training label sums:")
    print(train_df[LABEL_COLUMNS].sum())

    trainer.train()

    val_by_lang = load_all_languages_fold_val(base_dir, fold)

    for lang, val_df in val_by_lang.items():
        val_texts = val_df["input_text"].tolist()
        dummy_labels = np.zeros((len(val_texts), NUM_LABELS), dtype=float)

        val_dataset = PolarizationTypeDataset(
            val_texts,
            dummy_labels.tolist(),
            tokenizer,
            max_length=args.max_length,
        )

        pred_out = trainer.predict(val_dataset)
        logits = pred_out.predictions  
        save_val_with_logits(Path(args.output_dir), lang, fold, val_df, logits)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=-1,
                        help="Fold number 0-4. Use -1 to run all 5 folds.")

    parser.add_argument("--base_dir", type=str, default="../dev_phase",
                        help="Base folder that contains subtask3_kfold/")
    parser.add_argument("--output_dir", type=str, default="fine-tuned_BERT/task3_kfold_all_lang/",
                        help="Where to store trainer outputs/checkpoints")

    parser.add_argument("--model_name", type=str, default="xlm-roberta-large")
    parser.add_argument("--from_checkpoint", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    if args.fold == -1:
        for fold in range(5):
            run_one_fold(args, fold)
    else:
        if args.fold < 0 or args.fold > 4:
            raise ValueError("--fold must be in {0,1,2,3,4} or -1 for all folds")
        run_one_fold(args, args.fold)


if __name__ == "__main__":
    main()
