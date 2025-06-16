# House Prices Prediction

A machine learning solution for the Kaggle House Prices competition using XGBoost and hyperparameter optimisation.

## 🏆 Competition Overview

This project tackles the [House Prices Kaggle Competition](https://www.kaggle.com/competitions/home-data-for-ml-course/overview). The goal is to predict house sale prices in Ames, Iowa based on 79 explanatory variables describing various aspects of residential homes.

**Evaluation Metric:** Mean Absolute Error (MAE)

## 📊 Dataset

- **Training Data:** 1,460 houses with sale prices
- **Test Data:** 1,459 houses (predictions needed for submission)
- **Features:** 79 variables including lot size, quality ratings, year built, neighborhood, etc.
- **Target:** SalePrice (continuous variable)

## 🔧 Methodology

### Data Preprocessing
- **Train/Validation Split:** 80/20 split with random_state=42
- **Feature Selection:** 
  - Categorical features: < 10 unique values
  - Numerical features: int64/float64 types (excluding Id)
- **Missing Value Handling:** SimpleImputer with median/most frequent strategies
- **Scaling:** StandardScaler for numerical features
- **Encoding:** OneHotEncoder for categorical features

### Model Architecture
```python
Pipeline:
├── ColumnTransformer
│   ├── Numerical Pipeline: SimpleImputer → StandardScaler
│   └── Categorical Pipeline: SimpleImputer → OneHotEncoder
└── XGBRegressor
```

### Hyperparameter Optimization
Used RandomizedSearchCV with 100 iterations and 5-fold cross-validation:

```python
param_dist = {
    'model__n_estimators': randint(100, 1000),
    'model__learning_rate': uniform(0.01, 0.19),
    'model__max_depth': randint(3, 9),
    'model__reg_alpha': uniform(0, 10),
    'model__reg_lambda': uniform(1, 100),
    'model__subsample': uniform(0.7, 0.3),
    'model__colsample_bytree': uniform(0.7, 0.3)
}
```

## 📈 Results

- **Leaderboard Position:** 472 (top 3% of all submissions)
- **Kaggle Public Score:** 15391.55342
- **Validation MAE:** 16943.706897474316



## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn xgboost scipy
```

### Running the Code
1. Clone this repository
2. Download the dataset from Kaggle
3. Run the main script:
```bash
python house_prices_prediction.py
```

### Project Structure
```
House-Prices-Kaggle-Competition/
├── README.md
├── house-prices-predictor.ipynb    # Main script
├── submission.csv               # Kaggle submission file
├── data/
│   ├── train.csv               # Training data
│   └── test.csv                # Test data
```


## 📚 Dependencies

- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- scipy >= 1.7.0

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


*This project was developed as part of the Kaggle House Prices competition to demonstrate machine learning techniques for regression problems.*
