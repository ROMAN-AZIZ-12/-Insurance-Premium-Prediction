# Insurance Premium Prediction

A machine learning project that predicts annual health insurance charges based on a customer's age, BMI, smoking status, and number of dependents helping insurance companies price their policies fairly and accurately.

## The Business Problem

Insurance companies need to set premiums that reflect each customer's health risk. Charge too little and the company loses money. Charge too much and customers leave.

This project builds a regression model that predicts what a customer's annual insurance charges should be, based on their personal health profile.

> A model that accurately predicts premiums means fairer pricing for customers and sustainable profits for the company.

---

## Dataset

- **Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Rows:** 1,338 customers
- **Features:** Age, sex, BMI, children, smoker status, region, charges
- **Target:** Annual insurance charges (USD)
- **Problem Type:** Regression

---

## What I Found in the Data (EDA)

**Smoking is the dominant factor**
Smokers pay on average **$32,050/year** vs **$8,441/year** for non-smokers — that is 3.8x more. This single feature has a 0.832 correlation with charges, far stronger than any other variable.

**Age is the second strongest driver**
Insurance charges increase consistently with age, reflecting higher medical risk as people get older.

**BMI alone is weak — but combined with smoking it is powerful**
BMI has only a 0.119 direct correlation with charges. However, when combined with smoking status, the interaction becomes the second most important feature in the model.

**Sex and region carry almost no signal**
Sex had a 0.063 correlation with charges and region had 0.065. Both were dropped to reduce noise and keep the model clean.

**Charges are right-skewed**
A small number of very high-cost customers (old, obese smokers) pull the distribution rightward. Log transformation was applied to normalize the target before training.

---

## Feature Engineering

The key insight in this project was creating interaction features that the model cannot discover on its own.

| Feature | Formula | Why |
|---------|---------|-----|
| `bmi_smoker` | bmi × smoker | Obese smoker = extreme risk |
| `obese` | 1 if bmi > 30 | Obesity flag |
| `age_smoker` | age × smoker | Older smoker = decades of damage |
| `age_bmi` | age × bmi | Older + heavier = higher chronic risk |

**Impact of feature engineering:**
Linear Regression R² jumped from **0.635 → 0.843** simply by adding these interaction features.

---



## Model Comparison

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | 0.843 | $2,844 | $5,365 |
| **Random Forest** ✅ | **0.890** | **$2,235** | **$4,504** |
| Gradient Boosting | 0.888 | $2,091 | $4,544 |

Random Forest delivered the best overall performance with the highest R² and lowest RMSE.

## Final Results

| Metric | Score |
|--------|-------|
| R² Score | 0.890 |
| MAE | $2,235 |
| RMSE | $4,504 |

The model explains **89% of the variation** in insurance charges. On average, predictions are within **$2,235** of the actual premium.

---

## Feature Importance

| Rank | Feature | Importance | Meaning |
|------|---------|------------|---------|
| 1 | age | 0.362 | Older = higher risk = higher premium |
| 2 | bmi_smoker | 0.308 | Obese + smoker = extreme charges |
| 3 | smoker | 0.074 | Smoking alone also drives charges up |
| 4 | age_bmi | 0.072 | Aging + weight compound each other |
| 5 | bmi | 0.065 | BMI contributes independently too |
| 6 | age_smoker | 0.055 | Older smoker = decades of damage |
| 7 | children | 0.051 | More dependents = slightly higher cost |
| 8 | obese | 0.012 | Captured mainly by bmi_smoker already |

**Key finding:** `age` and `bmi_smoker` together account for **67% of the model's prediction power**.

---

## Business Insights

**1. Smoking is the biggest pricing factor**
Smokers pay 3.8x more than non-smokers. Any customer who smokes should be flagged immediately for a higher risk tier.

**2. The obesity-smoking combination is disproportionately expensive**
An obese smoker's charges are not just additive — the risks multiply. The `bmi_smoker` interaction feature captured this and became the second most important predictor in the model.

**3. Age drives steady, predictable cost increases**
Unlike smoking (which causes a sharp jump), age increases charges gradually and linearly. Insurers can plan for this with standard age-based pricing tiers.

**4. Region and gender do not meaningfully affect premiums**
The data shows no significant difference in charges across US regions or between males and females. Pricing based on these factors would be both inaccurate and potentially discriminatory.

---

## Sample Predictions

| Customer Profile | Predicted Annual Premium |
|-----------------|--------------------------|
| Age 25, BMI 22, Non-Smoker, 0 children | ~$4,200/year |
| Age 45, BMI 35, Non-Smoker, 2 children | ~$9,800/year |
| Age 60, BMI 40, Smoker, 3 children | ~$38,000/year |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/insurance-premium-prediction.git
cd insurance-premium-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# 3. Add dataset
# Download from Kaggle and place here: insurance.csv

# 4. Run the notebook
jupyter notebook insurance_prediction.ipynb

