# Overview

This repo is to document our work on [Task 9: Detecting Multilingual, Multicultural and Multievent Online Polarization](https://polar-semeval.github.io/).

Code for final submission are under `code/`.

Folders:

- `code_kfold`: the scripts for k-fold cross validation training, supporting multi and mono lingual models
- `code_final`: the scripts for fine-tuning final multi/mono models and inferencing on dev/test set
- `code_scatter_plot`: the scripts to draw scatter plots
- `kfold_multi_logits`: the generated logits in 5-fold cross validation of multi-lingual models
- `kfold_mono_logits`: the generated logits in 5-fold cross validation of mono-lingual models
- `final_multi_logits`: final multi-lingual models (trained on 3 random seeds) generated logits on the dev/test set
- `final_mono_logits`: final mono-lingual models (trained on 3 random seeds) generated logits on the dev/test set
- `final_preds`: final predictions for submission

Code (run them undert `code/`): 

- `low_resource_multi_fold_searchWT.py`: this script does grid search on the ensemble weights and class thresholds of **multiple** multi-lingual models, saves file `task<no.>_kfold_multi_WT.csv`
- `low_resource_multi_fold_get_final_preds.py`: this script reads saved ensemble weights and class thresholds and generates final predictions for submission, saves under folder `final_preds`
- `high_resource_mono_multi_fold_searchWT.py`:  this script does grid search on the ensemble weights and class thresholds of **one** mono-lingual model and **one** multi-lingual model, saves file `task<no.>_kfold_mono_multi_WT.csv`
- `high_resource_mono_multi_get_final_preds.py`: this script reads saved ensemble weights and class thresholds and generates final predictions for submission, saves under folder `final_preds`

# Final Submission - Low resource languages

We split all data into 5-fold.

In conceptualisation phase, we fine-tuned mDeberta-v3-base, XLM-RoBERTa-Large and mmBERT-base with 5-fold cross validation on three random seeds (1, 42, 111).

For the final submission, we fine-tuned mDeberta-v3-base, XLM-RoBERTa-Large and mmBERT-base separately on three random seeds (1, 42, 111).

## Method

Languages: amh ben deu fas hau hin ita khm mya nep ori pan swa tel tur urd

We picked 3 multi-lingual models as backbone (mDeBERTa, XLM-R, mmBERT). We tune ensemble weights and class thresholds.

**Process**

- For each fold K and each backbone model:
  - Train backbone with 3 seeds
  - Average logits across seeds
- Merge folds to get OOF logits per model
- Grid-search weights and threshold(s) to maximize OOF macro-F1
- Retrain each backbone models on all language data  with three seeds
- At inference, average seed logits per model, combine with ensemble weights, apply thresholds

```
# k-fold fine-tuning, multi-lingual models
code/fine-tuned_BERT/code_k_fold/t1_kfold_all_lang.py
```

## Results

| Task | t_min | t_max | t_step | w_step |
| ---- | ----- | ----- | ------ | ------ |
| 1    | 0.01  | 0.99  | 0.01   | 0.1    |
| 2    | 0.01  | 0.99  | 0.01   | 0.1    |
| 3    | 0.01  | 0.99  | 0.01   | 0.05   |

# Final Submission - High resource languages

Languages: arb, eng, pol, rus, spa, zho

These are the languages that mono-lingual models can outperform most multi-lingual models. We will ensemble them with multi-lingual models. We're also using weighting the logits when ensemble. The logits are averaged together. 

### Method

- train mono on all [language] data, seeds 1 42 111
- train multi on all data, seeds 1 42 111
- At inference:
  - average logits across seeds within each model family
  - then apply your frozen weight `w*`
  - then apply threshold `t*`

### Results

| Task | Language | Multi model | t_min | t_max | t_step | w_step | f1-dev   | f1-OOF |
| ---- | -------- | ----------- | ----- | ----- | ------ | ------ | -------- | ------ |
| 1    | arb      | mdeberta    | 0.01  | 0.99  | 0.01   | 0.05   | 0.826433 | 0.8422 |
|      | eng      | mmbert      |       |       |        |        | 0.801633 | 0.8204 |
|      | pol      | xlm         |       |       |        |        | 0.801105 | 0.8368 |
|      | rus      | xlm         |       |       |        |        | 0.842991 | 0.8009 |
|      | spa      | xlm         |       |       |        |        | 0.714984 | 0.7836 |
|      | zho      | xlm         |       |       |        |        | 0.915881 | 0.9001 |
| 2    | arb      | xlm         |       |       |        |        | 0.633725 | 0.6752 |
|      | eng      | mmbert      |       |       |        |        | 0.437691 | 0.5235 |
|      | pol      | xlm         |       |       |        |        | 0.672809 | 0.6306 |
|      | rus      | xlm         |       |       |        |        | 0.655310 | 0.6204 |
|      | spa      | xlm         |       |       |        |        | 0.659300 | 0.6785 |
|      | zho      | mdeberta    |       |       |        |        | 0.804962 | 0.8061 |
| 3    | arb      | mdeberta    |       |       |        |        | 0.640169 | 0.6416 |
|      | eng      | xlm         |       |       |        |        | 0.529423 | 0.5345 |
|      | spa      | xlm         |       |       |        |        | 0.519094 | 0.5471 |
|      | zho      | xlm         |       |       |        |        | 0.705929 | 0.6591 |

