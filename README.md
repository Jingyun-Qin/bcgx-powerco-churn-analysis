# PowerCo SME Churn Analysis — BCG X Data Science Job Simulation

End-to-end churn analysis for **PowerCo**, a gas and electricity utility, completed as part of the
[BCG X Data Science Job Simulation](https://www.theforage.com/) on Forage (September 2026).

**Business question:** PowerCo believed its SME customers were leaving because they are price-sensitive.
Is price really the driver of churn, and how can PowerCo identify at-risk clients early?

## Key results

| | Result |
|---|---|
| Data | 14,606 SME clients, 193,002 monthly price records (2015) |
| Baseline churn rate | 9.7% |
| Price movement in 2015 | Off-peak energy prices **fell** for ~95% of clients |
| Price-only model | ROC-AUC 0.58 (barely better than a coin flip) |
| Full Random Forest | ROC-AUC 0.72 |
| Threshold tuning | Recall on churners **5% → 36%** (threshold 0.5 → 0.2, tuned on training folds only) |
| Top 10% risk decile | Actual churn **32% (3.3x average)**, containing ~1/3 of all churners |

**Conclusion:** price is one driver among many, not the main one. Churn is higher among newer clients,
certain sales channels and acquisition campaigns, and the model leans on margin and consumption features.
Rather than across-the-board discounts, PowerCo should rank clients by predicted risk and pilot a
targeted retention offer on the top decile against a control group.

## Repository structure

```
notebooks/
  01_EDA_and_Data_Cleaning.ipynb   data quality, churn by segment, did prices move?
  02_Feature_Engineering.ipynb     price-sensitivity + relationship features, encoding, log transforms
  03_Modeling_and_Evaluation.ipynb Random Forest, threshold tuning, price-only vs full model, risk deciles
reports/
  PowerCo_Executive_Summary.pdf    one-page summary for client leadership
data/                              not included (see below)
```

## How to run

1. Place `client_data.csv` and `price_data.csv` (provided in the Forage simulation) in `data/`.
2. `pip install -r requirements.txt`
3. Run the notebooks in order; each one writes the input file for the next
   (`clean_data_after_eda.csv` → `data_for_predictions.csv` → `test_predictions.csv`).

## Tools

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## Data note

The datasets are synthetic data provided by BCG X / Forage for the job simulation and are not redistributed here.
