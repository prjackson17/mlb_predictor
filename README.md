# MLB Game Prediction Model

Video URL: https://www.loom.com/share/3918571438644e149fb047c1da675311  
Kaggle Dataset (mined by me): https://www.kaggle.com/datasets/parkerjackson17/mlb-2015-thru-2025

## Project Overview
A machine learning project that predicts MLB game outcomes using historical data from 2015-2025. The model uses differential features between home and away teams to predict game winners.

## Dataset
- **Time Period**: 2015-2025 MLB seasons
- **Features**: Team statistics (batting, pitching, recent performance), weather conditions, park factors
- **Target**: Binary classification (home win vs away win)
- **Split**: Training (2015-2023), Testing (2024-2025)

## Key Features
- `diff_run_diff`: Difference in run differentials between teams
- `diff_ops`: Difference in On-Base Plus Slugging
- `diff_whip`: Difference in Walks + Hits per Inning Pitched
- `diff_wins_last_10`: Recent performance differential
- `park_factor`: Venue-specific scoring environment
- Weather features: temperature, wind speed

## Models Implemented

| Model | Test Accuracy | Test ROC-AUC | Notes                                                                                  |
|-------|---------------|--------------|----------------------------------------------------------------------------------------|
| Logistic Regression | ~0.55         | ~0.57         | Baseline model with L1/L2 regularization                                               |
| Decision Tree | ~0.55         | ~0.56        | GridSearchCV for hyperparameter tuning                                                 |
| Random Forest | ~0.56         | ~0.57        | RandomizedSearchCV with 20 iterations                                                  |
| XGBoost | ~0.55         | ~0.57        | Gradient boosting with advanced regularization; RandomizedSearchCV with 30 iterations. |

## Key Findings
- **Most Important Feature**: `diff_run_diff` (run differential) consistently dominates across all models
- **Baseline Comparison**: All models outperform naive "always predict home win" baseline (~53%)
- **Home Team Bias**: Models predict home wins more accurately than away wins
- **Feature Insights**: WHIP differential and OPS differential show significant predictive power

## Technologies Used
- **Python Libraries**: pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib
- **ML Techniques**: GridSearchCV, RandomizedSearchCV, cross-validation
- **Feature Engineering**: Differential features, standardization, park factors
