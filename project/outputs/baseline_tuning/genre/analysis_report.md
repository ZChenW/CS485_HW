# Genre TF-IDF Logistic Regression Tuning Analysis

## Setup

- Split: stratified 70/15/15 train/dev/test with fixed seed.
- Selection: best dev macro-F1, then dev accuracy, then earlier simpler candidate.
- Model family: TF-IDF features with sklearn LogisticRegression using lbfgs.
- Multiclass formulation: sklearn 1.8 lbfgs uses multinomial loss for multiclass problems.
- Stop words: not used because the corpus is multilingual and function words may carry style or genre signal.

## Best Hyperparameters

```json
{
  "candidate": "bigrams_sublinear_C2",
  "ngram_range": [
    1,
    2
  ],
  "min_df": 2,
  "max_df": 1.0,
  "sublinear_tf": true,
  "lowercase": true,
  "stop_words": null,
  "C": 2.0,
  "class_weight": "balanced",
  "solver": "lbfgs",
  "task": "genre"
}
```

## Model Comparison

| dataset | model | dev_accuracy | test_accuracy | dev_macro_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| genre | majority | 0.1908 | 0.1908 | 0.0267 | 0.0267 |
| genre | tfidf_logreg_original | 0.4334 | 0.4420 | 0.4144 | 0.4172 |
| genre | tfidf_logreg_tuned | 0.4947 | 0.5062 | 0.4640 | 0.4689 |

## Label Distribution

| split | label | count | proportion |
| --- | --- | --- | --- |
| all | adventure | 461 | 0.0663 |
| all | children_young_adult | 388 | 0.0558 |
| all | comedy_satire | 233 | 0.0335 |
| all | drama | 1126 | 0.1620 |
| all | general_fiction | 903 | 0.1299 |
| all | historical | 1325 | 0.1906 |
| all | mystery_crime | 348 | 0.0501 |
| all | nonfiction_essay | 796 | 0.1145 |
| all | poetry | 384 | 0.0552 |
| all | religious_philosophical | 289 | 0.0416 |
| all | romance | 421 | 0.0606 |
| all | speculative | 277 | 0.0399 |
| train | adventure | 323 | 0.0664 |
| train | children_young_adult | 271 | 0.0557 |
| train | comedy_satire | 163 | 0.0335 |
| train | drama | 788 | 0.1620 |
| train | general_fiction | 632 | 0.1299 |
| train | historical | 927 | 0.1905 |
| train | mystery_crime | 244 | 0.0502 |
| train | nonfiction_essay | 557 | 0.1145 |
| train | poetry | 269 | 0.0553 |
| train | religious_philosophical | 202 | 0.0415 |
| train | romance | 295 | 0.0606 |
| train | speculative | 194 | 0.0399 |
| dev | adventure | 69 | 0.0662 |
| dev | children_young_adult | 59 | 0.0566 |
| dev | comedy_satire | 35 | 0.0336 |
| dev | drama | 169 | 0.1620 |
| dev | general_fiction | 136 | 0.1304 |
| dev | historical | 199 | 0.1908 |
| dev | mystery_crime | 52 | 0.0499 |
| dev | nonfiction_essay | 119 | 0.1141 |
| dev | poetry | 57 | 0.0547 |
| dev | religious_philosophical | 43 | 0.0412 |
| dev | romance | 63 | 0.0604 |
| dev | speculative | 42 | 0.0403 |
| test | adventure | 69 | 0.0662 |
| test | children_young_adult | 58 | 0.0556 |
| test | comedy_satire | 35 | 0.0336 |
| test | drama | 169 | 0.1620 |
| test | general_fiction | 135 | 0.1294 |
| test | historical | 199 | 0.1908 |
| test | mystery_crime | 52 | 0.0499 |
| test | nonfiction_essay | 120 | 0.1151 |
| test | poetry | 58 | 0.0556 |
| test | religious_philosophical | 44 | 0.0422 |
| test | romance | 63 | 0.0604 |
| test | speculative | 41 | 0.0393 |

## Tuned Test Per-Class Results

| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| adventure | 0.5517 | 0.6957 | 0.6154 | 69.0000 |
| children_young_adult | 0.4242 | 0.4828 | 0.4516 | 58.0000 |
| comedy_satire | 0.1333 | 0.0571 | 0.0800 | 35.0000 |
| drama | 0.4726 | 0.4083 | 0.4381 | 169.0000 |
| general_fiction | 0.4427 | 0.4296 | 0.4361 | 135.0000 |
| historical | 0.7007 | 0.4824 | 0.5714 | 199.0000 |
| mystery_crime | 0.4588 | 0.7500 | 0.5693 | 52.0000 |
| nonfiction_essay | 0.6288 | 0.6917 | 0.6587 | 120.0000 |
| poetry | 0.5676 | 0.7241 | 0.6364 | 58.0000 |
| religious_philosophical | 0.4043 | 0.4318 | 0.4176 | 44.0000 |
| romance | 0.3523 | 0.4921 | 0.4106 | 63.0000 |
| speculative | 0.3714 | 0.3171 | 0.3421 | 41.0000 |
| accuracy | 0.5062 | 0.5062 | 0.5062 | 0.5062 |
| macro avg | 0.4590 | 0.4969 | 0.4689 | 1043.0000 |
| weighted avg | 0.5119 | 0.5062 | 0.5004 | 1043.0000 |

## Top Tuned Test Misclassification Patterns

| gold | prediction | count | gold_total | share_of_gold |
| --- | --- | --- | --- | --- |
| drama | romance | 25 | 169 | 0.1479 |
| drama | general_fiction | 24 | 169 | 0.1420 |
| historical | nonfiction_essay | 23 | 199 | 0.1156 |
| general_fiction | drama | 18 | 135 | 0.1333 |
| historical | adventure | 18 | 199 | 0.0905 |
| general_fiction | romance | 13 | 135 | 0.0963 |
| comedy_satire | general_fiction | 12 | 35 | 0.3429 |
| drama | mystery_crime | 12 | 169 | 0.0710 |
| historical | drama | 12 | 199 | 0.0603 |
| drama | poetry | 11 | 169 | 0.0651 |
| children_young_adult | general_fiction | 10 | 58 | 0.1724 |
| general_fiction | mystery_crime | 10 | 135 | 0.0741 |
| historical | romance | 10 | 199 | 0.0503 |
| poetry | drama | 9 | 58 | 0.1552 |
| romance | drama | 9 | 63 | 0.1429 |

## Targeted Error Notes

- `historical` -> `general_fiction` was not a top tuned test pattern.
- `general_fiction` -> `historical` occurred 9 times (6.7% of `general_fiction` test examples).
- `drama` -> `comedy_satire` was not a top tuned test pattern.
- `comedy_satire` -> `drama` was not a top tuned test pattern.
- `mystery_crime` -> `adventure` was not a top tuned test pattern.
- `adventure` -> `mystery_crime` was not a top tuned test pattern.
- `speculative` -> `general_fiction` was not a top tuned test pattern.
- `general_fiction` -> `speculative` was not a top tuned test pattern.
- `general_fiction` absorbed 54 top-pattern errors from: drama (24), comedy_satire (12), children_young_adult (10), romance (8).

## Tuning Effect

Tuning meaningfully improved test macro-F1 by 0.0517.

## Label Redesign Recommendation

Audit the low-F1 labels `comedy_satire`, `speculative`. A redesign or merge should only be considered if annotation review confirms that frequent confusions such as `drama` -> `romance`, `drama` -> `general_fiction`, `historical` -> `nonfiction_essay`, `general_fiction` -> `drama`, `historical` -> `adventure` reflect unclear category boundaries rather than sparse data.
