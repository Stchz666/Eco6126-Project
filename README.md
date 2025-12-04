# ✈️ Airline Passenger Satisfaction Analysis

![Project Banner](results/figures/all_rounds_comparison.png)

## 📌 Overview
- **Goal**: Identify key drivers of passenger satisfaction and provide data-backed recommendations
- **Dataset**: 129,880 passenger feedback records with 25 features
- **Best Model**: XGBoost (AUC=0.995) after feature selection and hyperparameter tuning
- **Key Insight**: Online boarding and cabin class are the strongest predictors of satisfaction

## 🚀 How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw dataset in data/airline.csv
#    (Download from: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

# 3. Run full pipeline
python main.py