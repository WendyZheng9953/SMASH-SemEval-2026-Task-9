import os
import argparse
import pandas as pd
import numpy as np
import random
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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Task1 is single-label classification, keep CrossEntropy
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def add_language_prefix(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    df = df.copy()
    df["language"] = lang_code
    df["input_text"] = "[LANG={}] ".format(lang_code) + df["text"].astype(str)
    return df


def load_and_merge_train(train_dir: str, languages, seed: int) -> pd.DataFrame:
    dfs = []
    for lang in languages:
        path = os.path.join(train_dir, f"{lang}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing train file: {path}")
        df = pd.read_csv(path)

        required = {"text", "polarization"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df = add_language_prefix(df, lang)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return merged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--languages", type=str, nargs="+", required=True,
                        help="List of language codes, e.g. eng spa deu")
    parser.add_argument("--train_dir", type=str, default="../dev_phase/subtask1/train/")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large")
    parser.add_argument("--from_checkpoint", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="fine-tuned_BERT/task1_multilingual/")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    set_all_seeds(args.seed)

    save_dir = os.path.join(args.output_dir, str(args.seed))
    os.makedirs(save_dir, exist_ok=True)

    print("Languages:", args.languages)
    print("Loading and merging training data...")
    train_df = load_and_merge_train(args.train_dir, args.languages, seed=args.seed)

    print("Training label distribution:\n", train_df["polarization"].value_counts())

    load_source = args.from_checkpoint if args.from_checkpoint else args.model_name
    print(f"Loading model/tokenizer from: {load_source}")
    tokenizer = AutoTokenizer.from_pretrained(load_source)

    train_dataset = PolarizationDataset(
        train_df["input_text"].tolist(),
        train_df["polarization"].tolist(),
        tokenizer,
        max_length=args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        load_source,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        save_strategy="no",             
        logging_steps=200,
        report_to="none",
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        disable_tqdm=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("Starting fine-tuning...")
    trainer.train()

    trainer.save_model(save_dir)
    print(f"Saved fine-tuned model to: {save_dir}")


if __name__ == "__main__":
    main()
