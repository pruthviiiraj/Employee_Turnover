# Employee Turnover Prediction

Predicting whether an employee will leave a company, using logistic regression with L1 and L2 regularization.

## Problem

A company is losing too many employees. Every resignation raises hiring costs, delays projects and loses skilled people. The HR team wants a model that flags employees who are likely to leave, based on factors like job satisfaction, salary, work-life balance and training.

**Tasks:**

1. Build a baseline logistic regression model
2. Improve it with L1 (Lasso) and L2 (Ridge) regularization
3. Compare the models and recommend the best one

## Dataset

- **900 employees, 15 features, 1 target**
- **Target:** `Employee_Turnover` (1 = left, 0 = stayed)
- **Features include:** Job_Satisfaction, Performance_Rating, Years_At_Company, Work_Life_Balance, Distance_From_Home, Monthly_Income, Age, Annual_Bonus, Training_Hours, plus two engineered features (Annual_Bonus², and Annual_Bonus × Training_Hours)
- All features were already normalized to the 0 to 1 range
- Classes are balanced: 51% stayed, 49% left

## Approach

### 1. Data cleaning

The raw file had **1,350 rows**, but **450 were exact duplicates**. Removing them left 900 rows, matching the official dataset description. Keeping the duplicates would have let the same employee appear in both the training and test sets (data leakage), which inflates test scores.

### 2. Exploratory data analysis

- **5 features** showed a clear link with turnover (correlation around 0.30 to 0.35): Distance_From_Home, Performance_Rating, Job_Satisfaction, Work_Life_Balance, Years_At_Company
- The other 10 features were close to zero
- The three bonus-related features were strongly correlated with each other (**multicollinearity**)

### 3. Preprocessing

- 80/20 train/test split (`random_state=42`)
- Features standardized with `StandardScaler` (fit on training data only, to avoid leakage)

### 4. Models

| Model | Setup |
|---|---|
| Baseline | Logistic regression, no penalty |
| L2 (Ridge) | Penalty on squared coefficients. Best C = 1, chosen with 5-fold GridSearchCV |
| L1 (Lasso) | Penalty on absolute coefficients. Best C = 0.1, chosen with 5-fold GridSearchCV |

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Features used |
|---|---|---|---|---|---|---|
| Baseline | 0.850 | 0.911 | 0.783 | 0.842 | 0.950 | 15 |
| **L1** | **0.850** | **0.911** | **0.783** | **0.842** | **0.958** | **8** |
| L2 | 0.844 | 0.910 | 0.772 | 0.835 | 0.951 | 15 |

### Key findings

- **All three models perform almost the same.** The baseline was not overfitting, so regularization did not raise accuracy.
- **L1 removed 7 of 15 features** (income, age, bonus, training hours, number of previous companies, and both engineered bonus features) with no loss in performance.
- **The features L1 kept match the EDA.** The model and the correlation analysis agree on what drives turnover.
- **L2 shrank every coefficient but removed none.**
- **The baseline gave Annual_Bonus and Annual_Bonus² large opposite coefficients** that cancel each other out, a typical sign of multicollinearity. L2 reduced this, and L1 removed both.

## Recommendation

**L1 (Lasso) logistic regression** is the recommended model:

- Same accuracy as the baseline, and the highest AUC (0.958)
- Uses only 8 features instead of 15, so it is simpler and less likely to overfit
- Easier for HR to understand and act on

**Limitation:** recall is 0.78, so the model misses about 1 in 5 employees who leave. Lowering the decision threshold below 0.5 would catch more of them, at the cost of more false alarms.

## Business insights

**Main drivers of turnover** (strongest first):

1. Performance_Rating
2. Work_Life_Balance
3. Job_Satisfaction
4. Distance_From_Home
5. Years_At_Company

**Suggested actions for HR:**

- Offer remote or hybrid options to employees who live far away
- Review workloads and offer flexible hours
- Focus retention efforts on high performers, who are the most likely to leave
- Hold check-ins at key work anniversaries

**Salary, bonus and training hours showed no measurable effect.** Raising pay alone may not reduce turnover.

## Limitations

- Higher Job_Satisfaction and Work_Life_Balance scores were linked to **more** turnover, which is counterintuitive. The dataset may be synthetic, or these columns may be encoded in reverse.
- Employee_Role and Department are labelled as "encoded" but stored as continuous values, so they were treated as numeric.
- The dataset is small (900 rows). Results should be confirmed on real company data.
- Correlation is not causation.

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/employee-turnover-prediction.git
cd employee-turnover-prediction

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the libraries
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Open the notebook
jupyter notebook assignment2.ipynb
```

## Project structure

```
├── assignment2.ipynb       # Full analysis: cleaning, EDA, models, comparison
├── employee_turnover.csv   # Dataset
└── README.md               # This file
```

## Tools

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter