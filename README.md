# LAPD Crime Data – Machine Learning & Forecasting Project

> **Bar-Ilan University – Introduction to Artificial Intelligence (2025)**  
> *Team project submission by Yael Avni and Tal Cohen*  

---

## Project Overview

This project investigates **Los Angeles Police Department (LAPD) crime data (2020–Present)** using a **full AI/ML pipeline**:
1. **Exploratory Data Analysis (EDA)** – cleaning, feature engineering, visualization.  
2. **Supervised Learning** – predicting crime-related outcomes with classification models, specifically the gender of the victim. 
3. **Unsupervised Learning** – discovering hidden patterns and clusters in crime incidents.  
4. **Time Series Forecasting** – predicting daily crime counts with classical and deep models, neural networks.  

Our aim was not only to apply algorithms, but to **understand their suitability, limitations, and insights they yield**.  
The report is structured to reflect both **practical implementation** and **conceptual reasoning**: *why these methods, how they were tuned, and what we learned*, all to better understand and get insights about the LAPD crimes.

---

## Repository Structure

```
├── eda.py              # EDA function definitions
├── supervised.py       # Supervised learning function definitions 
├── unsupervised.py     # Unsupervised clustering function definitions
├── timeseries.py       # Forecasting function definitions 
├── LAPD_EDA.ipynb      # Data cleaning, preprocessing, EDA visualizations
├── LAPD_supervised.ipynb   # Supervised ML framework (cross-validation, tuning, metrics)
├── LAPD_unsupervised.ipynb # Dimensionality reduction + clustering + anomaly detection
├── LAPD_timeseries.ipynb   # Forecasting (ARIMA, SARIMAX, Prophet, LSTM, ensembles)
└── data/Crime_Data_from_2020_to_Present.csv.zip
```
---

## Exploratory Data Analysis (EDA)

Implemented in **`eda.py`** and demonstrated in **`LAPD_EDA.ipynb`**.

- **Data source:** [LAPD Crime Data 2020–Present](https://www.kaggle.com/datasets/kushsheth/los-angeles-police-department-lapd-crime-data).  
- **Cleaning:** handled missing values, divided into categorical and numerical features, normalized numeric ranges, detected possible outliers.  
- **Feature engineering:**  
  - Temporal features (weekday, weekend, holidays, seasonality).  
  - Victim features (age, sex).  
  - Spatial aggregation (precincts, beats).  
- **Visualizations:**  
  - Heatmaps (crime × time-of-day, crime × day-of-week).  
  - Calendar-style daily crime plots.  
  - Correlation maps for numeric/categorical features.  

**Key insight:** Crime is not evenly distributed. Strong temporal and demographic patterns emerged (e.g., weekend/night spikes, gender/age victim disparities).  

---

## Supervised Learning

Implemented in **`supervised.py`** and run in **`LAPD_supervised.ipynb`**.

### Pipeline
- Unified framework (`SupervisedMLTrainer`) with:
  - Preprocessing (imputation, scaling, one-hot encoding).  
  - Automated stratified sampling.  
  - Model factory supporting: **Logistic Regression, Random Forest, HistGB, SVM, MLP, XGBoost**.  
  - Hyperparameter search (RandomizedSearchCV / GridSearchCV).  
  - Evaluation: ROC-AUC, Precision-Recall, F1, confusion matrices.  
  - Visualizations: ROC, PR curves, confusion matrices, model comparisons.  

### Insights
- **Best models:** Gradient Boosting and XGBoost consistently achieved the strongest ROC-AUC.  
- **Imbalanced data:** Class weighting & stratified sampling were essential.  
- **Feature importance:** Location, victim demographics, and time-of-day proved most predictive.  

---

## Unsupervised Learning

Implemented in **`unsupervised.py`** and explored in **`LAPD_unsupervised.ipynb`**.

### Methods
- Dimensionality Reduction: **PCA, SVD**, **t-SNE**, **UMAP** (2–4 components).  
- Clustering: **KMeans (k=2–10), DBSCAN (eps sweep)**.  
- Validation: **Silhouette, Davies-Bouldin, Calinski-Harabasz**.  
- Anomaly detection: **Isolation Forest, Local Outlier Factor (LOF)**.  
- Visualization: 2D\3D scatterplots of embeddings with cluster labels.  

### Findings
- Best clusterings werer accurate but most statistically meaningful were later shown in the report.
- Clear **incident-type clusters** emerged (e.g., property vs violent crimes).  

---

## Time Series Forecasting

Implemented in **`timeseries.py`** and tested in **`LAPD_timeseries.ipynb`**.

### Models
- **Naïve baselines** (last value, weekly seasonal, adaptive mean).  
- **Classical**: ARIMA, SARIMAX (grid search for best orders).  
- **Prophet**: flexible trend/seasonality with holidays.  
- **LSTM**: sequence-to-sequence neural forecaster with engineered lags & seasonal features.  
- **Ensembles**: weighted blends of bias-corrected models.  

### Evaluation
- Metrics: **MAE, RMSE, MAPE, sMAPE**.  
- Cross-validation: **expanding window** & **rolling horizon**.  
- Diagnostics: stationarity checks (ADF, KPSS), residual analysis, bias detection.  

### Results
- **SARIMAX** captured weekly structure best (lowest RMSE).  
- **Prophet** adapted well to holiday/weekend seasonality.  
- **LSTM** showed promise but required more data for stability.  
- Ensemble of corrected models yielded the most **robust forecasts**.  

 No single model dominates. Classical + deep + ensemble forecasting gave complementary strengths.  

---

## Reflections & Contributions

- **Integration across paradigms:** Instead of treating EDA, supervised, unsupervised, and forecasting separately, we built **a modular ecosystem** where insights in one stage (e.g., anomalies in clustering) informed others (e.g., supervised features).  
- **Bias awareness:** Every stage included checks against overfitting and bias (regularization, dropout, cross-validation, anomaly detection, bias correction in time series).  
- **Practicality:** Each script can run **standalone** in Colab with minimal dependencies, making the project reproducible and educational.  
- **Meaning:** Beyond metrics, we aimed to interpret results in terms of **real-world crime insights** – seasonality, demographic vulnerabilities, clustering of crime types.  

---

## How to Run

1. Clone repository & install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run EDA:
   ```bash
   python eda.py --data data/Crime_Data_from_2020_to_Present.csv.zip
   ```
3. Supervised learning:
   ```bash
   python supervised.py --data lapd_clean_final.parquet --target victim_sex
   ```
4. Unsupervised analysis:
   ```bash
   python unsupervised.py --data lapd_clean_final_nolabel.parquet --output results_unsup
   ```
5. Time series forecasting:
   ```bash
   python timeseries.py --data data/Crime_Data_from_2020_to_Present.csv.zip
   ```

---

## References

- [LAPD Crime Data 2020–Present](https://www.kaggle.com/datasets/kushsheth/los-angeles-police-department-lapd-crime-data).
- Course textbook & lecture notes  
- Project assignment instructions  

---

## Conclusion

Through this project, we demonstrated:
- **EDA** uncovers structure and informs feature design.  
- **Supervised learning** can reliably predict per crime the vistim sex with explainability.  
- **Unsupervised learning** reveals latent structure and outliers.  
- **Time series models** highlight temporal dynamics and enable forecasting.  

## Related links
[Git](https://github.com/yaeliavni/AI)
r\AI community about ai 

---
