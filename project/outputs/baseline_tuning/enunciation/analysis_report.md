# Enunciation TF-IDF Logistic Regression Tuning Analysis

## Setup

- Split: stratified 70/15/15 train/dev/test with fixed seed.
- Selection: best dev macro-F1, then dev accuracy, then earlier simpler candidate.
- Model family: TF-IDF features with sklearn LogisticRegression using lbfgs.
- Multiclass formulation: sklearn 1.8 lbfgs uses multinomial loss for multiclass problems.
- Stop words: not used because the corpus is multilingual and function words may carry style or genre signal.

## Best Hyperparameters

```json
{
  "candidate": "original_settings",
  "ngram_range": [
    1,
    2
  ],
  "min_df": 2,
  "max_df": 1.0,
  "sublinear_tf": false,
  "lowercase": true,
  "stop_words": null,
  "C": 1.0,
  "class_weight": "balanced",
  "solver": "lbfgs",
  "task": "enunciation"
}
```

## Model Comparison

| dataset | model | dev_accuracy | test_accuracy | dev_macro_f1 | test_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| enunciation | majority | 0.5417 | 0.5408 | 0.1405 | 0.1404 |
| enunciation | tfidf_logreg_original | 0.7575 | 0.7217 | 0.4411 | 0.4159 |
| enunciation | tfidf_logreg_tuned | 0.7575 | 0.7217 | 0.4411 | 0.4159 |

## Label Distribution

| split | label | count | proportion |
| --- | --- | --- | --- |
| all | dialogue | 1660 | 0.2076 |
| all | epistolary | 18 | 0.0023 |
| all | first_person | 1927 | 0.2410 |
| all | mixed | 63 | 0.0079 |
| all | third_person | 4329 | 0.5413 |
| train | dialogue | 1162 | 0.2076 |
| train | epistolary | 12 | 0.0021 |
| train | first_person | 1349 | 0.2410 |
| train | mixed | 44 | 0.0079 |
| train | third_person | 3030 | 0.5414 |
| dev | dialogue | 249 | 0.2075 |
| dev | epistolary | 3 | 0.0025 |
| dev | first_person | 289 | 0.2408 |
| dev | mixed | 9 | 0.0075 |
| dev | third_person | 650 | 0.5417 |
| test | dialogue | 249 | 0.2075 |
| test | epistolary | 3 | 0.0025 |
| test | first_person | 289 | 0.2408 |
| test | mixed | 10 | 0.0083 |
| test | third_person | 649 | 0.5408 |

## Tuned Test Per-Class Results

| label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| dialogue | 0.5627 | 0.6667 | 0.6103 | 249.0000 |
| epistolary | 0.0000 | 0.0000 | 0.0000 | 3.0000 |
| first_person | 0.6886 | 0.6505 | 0.6690 | 289.0000 |
| mixed | 0.0000 | 0.0000 | 0.0000 | 10.0000 |
| third_person | 0.8114 | 0.7889 | 0.8000 | 649.0000 |
| accuracy | 0.7217 | 0.7217 | 0.7217 | 0.7217 |
| macro avg | 0.4126 | 0.4212 | 0.4159 | 1200.0000 |
| weighted avg | 0.7214 | 0.7217 | 0.7204 | 1200.0000 |

## Top Tuned Test Misclassification Patterns

| gold | prediction | count | gold_total | share_of_gold |
| --- | --- | --- | --- | --- |
| third_person | dialogue | 81 | 649 | 0.1248 |
| first_person | third_person | 60 | 289 | 0.2076 |
| dialogue | third_person | 56 | 249 | 0.2249 |
| third_person | first_person | 55 | 649 | 0.0847 |
| first_person | dialogue | 41 | 289 | 0.1419 |
| dialogue | first_person | 27 | 249 | 0.1084 |
| mixed | dialogue | 7 | 10 | 0.7000 |
| epistolary | first_person | 2 | 3 | 0.6667 |
| mixed | third_person | 2 | 10 | 0.2000 |
| epistolary | third_person | 1 | 3 | 0.3333 |
| mixed | first_person | 1 | 10 | 0.1000 |
| third_person | mixed | 1 | 649 | 0.0015 |

## Targeted Error Notes

- `dialogue` -> `mixed` was not a top tuned test pattern.
- `mixed` -> `dialogue` occurred 7 times (70.0% of `mixed` test examples).
- `first_person` -> `epistolary` was not a top tuned test pattern.
- `epistolary` -> `first_person` occurred 2 times (66.7% of `epistolary` test examples).
- `third_person` absorbed 119 top-pattern errors from: first_person (60), dialogue (56), mixed (2), epistolary (1).

## Tuning Effect

Tuning did not change test macro-F1.

## Label Redesign Recommendation

Audit the low-F1 labels `epistolary`, `mixed`. A redesign or merge should only be considered if annotation review confirms that frequent confusions such as `third_person` -> `dialogue`, `first_person` -> `third_person`, `dialogue` -> `third_person`, `third_person` -> `first_person`, `first_person` -> `dialogue` reflect unclear category boundaries rather than sparse data.
