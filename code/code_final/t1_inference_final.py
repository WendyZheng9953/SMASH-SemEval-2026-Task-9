import os
import argparse
import json
import pandas as pd
import numpy as np
import random
import torch
torch._dynamo.config.suppress_errors = True

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


def add_language_prefix(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    df = df.copy()
    df["language"] = lang_code
    df["input_text"] = "[LANG={}] ".format(lang_code) + df["text"].astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--languages", type=str, nargs="+", required=True,
                        help="List of language codes, e.g. eng spa deu")
    parser.add_argument("--input_dir", type=str, default="../dev_phase/subtask1/dev/",
                        help="Folder containing <lang>.csv files with columns: id, text")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to saved fine-tuned model directory (output_dir/<seed>/)")
    parser.add_argument("--tokenizer_name", type=str, default="xlm-roberta-large",
                        help="Tokenizer name/path to load (since tokenizer wasn't saved)")
    parser.add_argument("--output_dir", type=str, default="inference_logits/subtask1/",
                        help="Where to save <lang>_val.csv with logits column")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
    set_all_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    infer_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_tmp"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=infer_args,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    model.eval()

    for lang in args.languages:
        in_path = os.path.join(args.input_dir, f"{lang}.csv")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Missing input file: {in_path}")

        df = pd.read_csv(in_path)
        if "text" not in df.columns:
            raise ValueError(f"{in_path} must contain 'text' column")

        df = add_language_prefix(df, lang)
        texts = df["input_text"].tolist()

        dummy_labels = [0] * len(texts)

        dataset = PolarizationDataset(texts, dummy_labels, tokenizer, max_length=args.max_length)

        print(f"Predicting logits for {lang} (N={len(dataset)})")
        pred_out = trainer.predict(dataset)
        logits = pred_out.predictions

        out_df = df.copy()
        out_df["logits"] = [json.dumps(row.tolist(), ensure_ascii=False) for row in logits]

        out_df = pd.DataFrame({"id": df["id"].tolist(),
                               "logits": [json.dumps(r.tolist(), ensure_ascii=False) for r in logits]})

        out_path = os.path.join(args.output_dir, f"logits_{lang}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved logits to: {out_path}")


if __name__ == "__main__":
    main()
