import os
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
from torch.utils.data import DataLoader

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


class PolarizationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


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
    def __init__(self, *args, sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels.dtype == torch.float:
            loss_fct = FocalLoss()
            loss = loss_fct(logits, labels)
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        if self.sampler is None:
            return super().get_train_dataloader()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )


def add_language_prefix(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    df = df.copy()
    df["language"] = lang_code
    df["input_text"] = f"[LANG={lang_code}] " + df["text"].astype(str)
    return df


def load_fold_train(base_dir: Path, language: str, fold: int) -> pd.DataFrame:
    p = base_dir / "subtask1_kfold" / language / f"fold_{fold}" / "train.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing train file: {p}")
    df = pd.read_csv(p)
    return add_language_prefix(df, language)


def load_fold_val(base_dir: Path, language: str, fold: int) -> pd.DataFrame:
    p = base_dir / "subtask1_kfold" / language / f"fold_{fold}" / "val.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing val file: {p}")
    df = pd.read_csv(p)
    return add_language_prefix(df, language)


def load_other_languages_train(base_dir: Path, exclude_language: str, fold: int) -> pd.DataFrame:
    train_dir = base_dir / "subtask1" / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train folder: {train_dir}")

    dfs = []
    for csv_path in sorted(train_dir.glob("*.csv")):
        lang = csv_path.stem  # e.g. eng.csv -> "eng"
        if lang == exclude_language:
            print(f"excluding {lang}")
            continue
        df = pd.read_csv(csv_path)
        dfs.append(add_language_prefix(df, lang))

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
    


def save_val_with_logits(base_dir: Path, language: str, fold: int, val_df: pd.DataFrame, logits: np.ndarray):
    out_path = base_dir / f"mono" / language / f"fold_{fold}_val.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    val_df = val_df.copy()
    val_df["logits"] = [json.dumps(row.tolist(), ensure_ascii=False) for row in logits]

    val_df.to_csv(out_path, index=False)
    print(f"Saved logits to: {out_path}")


def run_one_fold(args, fold: int):
    base_dir = Path(args.base_dir).resolve()

    fold_seed = args.seed
    set_all_seeds(fold_seed)

    train_df = load_fold_train(base_dir, args.language, fold)
    if args.include_other_languages:
        other_df = load_other_languages_train(base_dir, args.language, fold)
        if len(other_df) > 0:
            train_df = pd.concat([train_df, other_df], ignore_index=True)

    train_df = train_df.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)

    val_df = load_fold_val(base_dir, args.language, fold)

    load_source = args.from_checkpoint if args.from_checkpoint else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(load_source)

    train_texts = train_df["input_text"].tolist()
    train_labels = train_df["polarization"].tolist()

    val_texts = val_df["input_text"].tolist()
    dummy_labels = [0] * len(val_texts)

    train_dataset = PolarizationDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = PolarizationDataset(val_texts, dummy_labels, tokenizer, max_length=args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(load_source, num_labels=2)

    model_name = args.model_name.replace("/", "_")
    fold_out_dir = Path(args.output_dir).resolve() / args.language / f"fold_{fold}_{model_name}_{args.seed}"
    # fold_out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(fold_out_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_strategy="no",
        # save_total_limit=1,
        logging_steps=200,
        report_to="none",
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        seed=fold_seed,
        data_seed=fold_seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print(f"\n=== Language={args.language} | Fold={fold} | include_other_languages={args.include_other_languages} ===")
    print("Train size:", len(train_df), "| Val size:", len(val_df))
    print("Training label distribution:\n", train_df["polarization"].value_counts())

    trainer.train()

    pred_out = trainer.predict(val_dataset)
    logits = pred_out.predictions

    save_val_with_logits(Path(args.output_dir), args.language, fold, val_df, logits)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--language", type=str, required=True, help="Training language code, e.g. eng")
    parser.add_argument("--include_other_languages", action="store_true",
                        help="If set, merge ../dev_phase/subtask1/train/*.csv excluding <language>.csv into training data")
    parser.add_argument("--fold", type=int, default=-1,
                        help="Fold number 0-4. Use -1 to run all 5 folds.")

    parser.add_argument("--base_dir", type=str, default="../dev_phase", help="Base folder that contains subtask1/ and subtask1_kfold/")
    parser.add_argument("--output_dir", type=str, default="fine-tuned_BERT/task1_kfold/", help="Where to store trainer outputs/checkpoints")

    parser.add_argument("--model_name", type=str, default="none")
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
