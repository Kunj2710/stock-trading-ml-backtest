<div align="center">

# Can ML Beat Buy-and-Hold?

**Backtesting 10 ML trading strategies on Coca-Cola (KO) stock, 2021–2025**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

</div>

> **TL;DR:** Each week is labelled *green* (good week to hold) or *red* (stay in cash). Models train on 2021–23 and trade 2024–25 starting with $100. **Only 3 of 10 strategies beat buy-and-hold, and the two most profitable models were the two least accurate.** In trading, accuracy is the wrong metric.

<br>

## Results: $100 invested over the 2-year test period

| Strategy | Test accuracy | Final balance | vs buy-and-hold |
|:--|--:|--:|:-:|
| **Decision tree (entropy)** | 51.9% | **$133.90** | ✅ +$7.76 |
| **Decision tree (Gini)** | 57.6% | **$131.72** | ✅ +$5.58 |
| **LDA** | 67.9% | **$126.96** | ✅ +$0.82 |
| *Buy-and-hold* | – | *$126.14* | – |
| QDA | 70.8% | $118.18 | ❌ |
| Logistic regression | 69.8% | $114.58 | ❌ |
| Random forest (N=10, d=5) | – | $113.49 | ❌ |
| Gaussian Naive Bayes | 69.8% | $107.88 | ❌ |
| kNN (k=5) | 60.4% | $106.40 | ❌ |
| Student-t Naive Bayes | 68.9% | $103.22 | ❌ |
| Linear SVM | 68.9% | $100.00 | ❌ never took a position |

**Why accuracy misleads:** a model can be "right" on many small weeks and still miss the few big ones. The SVM reached 68.9% accuracy while never trading at all.

## What's inside

| Notebook | Technique |
|:--|:--|
| `01`–`03` | Label engineering and the label-following strategy (the "perfect foresight" upper bound) |
| `04_knn` · `05_logistic_regression` | kNN with k tuning; logistic regression equation and interpretation |
| `06_polynomial_window_models` | Rolling-window polynomial regression (window W = 5–12, degree d = 1–3) |
| `07_linear_separability` | Hand-drawn linear decision boundary on reduced data |
| `08`–`10` | Gaussian and Student-t Naive Bayes, LDA vs QDA |
| `11`–`12` | Decision tree (entropy vs Gini) and random forest grid search (N × depth) |
| `13_strategy_hamming_distance` | How far each strategy's week-by-week decisions sit from the true labels |
| `14_svm` · `15_kmeans_regimes` | Linear, RBF and polynomial SVM; k-means market regimes (k = 3 by knee method) |

Features: mean and volatility of daily returns for each week. Data: `KO_weekly_return_*.csv`, built with `utils/` from Yahoo Finance.

## Run it

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn yfinance kneed
jupyter lab            # notebooks read the KO_*.csv files in the repo root
```

## What I'd do next

- Add **transaction costs and slippage**, since weekly switching could erase the small edges
- **Walk-forward validation** instead of one train/test split
- Optimise for **profit or Sharpe ratio directly** rather than classification accuracy

<br>

<sub>Built during Boston University MET CS 577 (Data Science with Python), Spring 2026. Data helper scripts in <code>utils/</code> were provided by the course.</sub>
