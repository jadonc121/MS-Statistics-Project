# A Comparison of Time Series Forecasting Methods: Classical to Deep Learning

_**Early work (2021-2022).** This was my first large end-to-end project, written as a single exploratory notebook before I moved into industry. The report was the main deliverable. I've kept it as submitted to show my starting point. I now build projects as modular, reusable Python packages. See [my current project](https://github.com/jadonc121/causal-inference) for what that looks like._

Master's project for the M.S. in Statistics, University of Colorado Denver (2021-2022).
**Full write-up:** [MS Project Report.pdf](MS%20Project%20Report.pdf) | **Code:** [MS Project Workbook FINAL.ipynb](MS%20Project%20Workbook%20FINAL.ipynb)

## Question
Do machine learning and deep learning models actually beat classical statistical models at time series forecasting? The literature disagrees, so this project tests it directly on the same data, measuring both accuracy and computational cost.

## Scope
- **13 models from 3 method classes**, compared against a naive benchmark:
  - Classical statistics: exponential smoothing (ETS, 4 variants) and ARIMA / seasonal ARIMA
  - Machine learning: support vector regression, CART, k-nearest neighbors, multilayer perceptron
  - Deep learning: RNN, LSTM, GRU
- **2 datasets with different behavior** (~40,000 hourly observations each):
  - NOAA hourly air temperature, Boulder CO, 2017-2021: strongly seasonal
  - Bitcoin/USD hourly price, 2017-2021: no seasonality, high volatility
- **Metrics:** RMSE, MAE, sMAPE (accuracy) and run time (efficiency)

## Methods
- Cleaned and prepared both series (duplicate and missing-hour handling, datetime indexing).
- Split each series into train and test periods; truncated the Bitcoin series to its first 75% because the test window otherwise fell entirely inside an extreme volatility spike.
- Tuned and fit each model, forecast the test window, and scored every model with the same metrics.
- Compared models within each class, across classes, and across the two datasets.

## Results
- **ARIMA and a shallow MLP** beat the naive benchmark on both datasets, making them the most robust choices across seasonal and volatile data.
- **All machine learning models** performed at or above the benchmark on both datasets.
- **Deep learning models performed worst.** Their forecasts flattened into plateaus, which I suspect comes from exploding gradients and limited hyperparameter tuning (see below).
- **Most efficient models** (lowest run time): non-seasonal ETS, CART, and KNN.
- Takeaway: simple, cheap models were hard to beat, consistent with the part of the literature this project set out to test.

## Limitations
- Deep learning models were under-tuned because of limited local compute (no full hyperparameter search).
- sMAPE occasionally disagreed with RMSE/MAE, so a different percent-error metric would be better.
- Only two series were tested, so results describe these datasets and not forecasting in general.

## Skills demonstrated
Time series forecasting (ETS, ARIMA/SARIMA), classical vs. ML vs. deep learning model comparison, experimental design, hyperparameter selection, data cleaning, model evaluation (RMSE/MAE/sMAPE), technical writing.

## Repository contents
| File | What it is |
|---|---|
| `MS Project Report.pdf` | Full paper: literature review, methods, results, conclusion |
| `MS Project Workbook FINAL.ipynb` | Code for all models and experiments |
| `BTCdf.csv`, `NOAAdf.csv` | Cleaned datasets used in the notebook |
| `ModelTree.PNG` | Diagram of the models by class |

## How to run
Python 3 with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `statsmodels`, `pmdarima`, `tensorflow`/`keras`. Open the notebook and run top to bottom (the data files are already in the repo).

## Next steps
A planned rewrite as modular code, with a proper train/validation/test design, tuned deep learning models, and a combination (ensemble) model, which the literature suggests performs especially well. Cross-validated time series approaches (such as rolling-origin evaluation) will be considered, and newer model architectures, such as transformers, will be included.
