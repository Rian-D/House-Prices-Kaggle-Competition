# House Prices Prediction

My machine learning solution for the Kaggle House Prices competition, earning me a leaderboard position of **#231** and a score of 14292.55428 (**top 4%** of all submissions).

<img width="1236" alt="image" src="https://github.com/user-attachments/assets/5b9eee29-3f0a-4e70-995c-fd95ef2266e3" />


## Competition Overview

This project tackles the [House Prices Kaggle Competition](https://www.kaggle.com/competitions/home-data-for-ml-course/overview). The goal is to predict house sale prices in Ames, Iowa based on 79 explanatory variables describing various aspects of residential homes.

**Evaluation Metric:** Mean Absolute Error (MAE)


## Dataset Summary

- **Training Data:** 1,460 houses with sale prices
- **Test Data:** 1,459 houses (predictions needed for submission)
- **Features:** 79 variables including lot size, quality ratings, year built, neighborhood, etc.
- **Target:** SalePrice (continuous variable)


## Methodology

1. Data Preparation

- Immediate train/test split to prevent data leakage
- Feature engineering applied to both datasets consistently
- Train-test split to simulate real-world performance & prevent data leakage

2. Preprocessing Pipeline & Model Pipeline Creation

- Handles missing values with appropriate strategies per feature type 
- Scales numerical features for optimal model performance
- One-hot encodes categorical variables
- Preprocessing pipeline combined with model in model pipeline

3. Hyperparameter Tuning

- RandomizedSearchCV for efficient parameter exploration (base model used grid search but randomised search is better for high-dimensional parameter spaces and significantly reduced MAE)
- Chose balanced parameter ranges to prevent underfitting & overfitting whilst ensuring runtime isn't drastically increased
- Regularisation focus e.g reg_alpha, reg_lambda, subsample etc helps model generalise to unseen data (another feature added onto base model to reduce MAE further)

4. Model Training & Validation

- Chose to use XGBoost as it is often the best for regression problems using tabular data
- Cross-validation for robust performance estimation


## Feature Engineering Details
**Created Features:**

- TotalSF: Combined square footage (basement + 1st floor + 2nd floor)
- TotalBaths: Total bathroom count including half-baths
- HouseAge: Age of house at time of sale
- YearsSinceRemodel: Time since last renovation
- TotalPorchSF: Combined porch area
- HasPool/HasGarage/HasBsmt/HasFireplace: Binary presence indicators
- OverallGrade: Interaction between overall quality and condition

**Why These Features Matter:**

- Total square footage is often the strongest predictor of house value
- Age-related features capture depreciation and modernisation effects
- Binary indicators help the model understand property amenities
- Interaction terms capture relationships between multiple factors


## Potential Improvements

- Ensemble Methods: Combine XGBoost with other models like Random Forest and Ridge regression
- Log Transformation: Apply log transformation to target variable for better distribution
- Outlier Handling: Use Interquartile range to identify extreme prices and remove them from training data


## License

This project is licensed under the MIT License.


*This project was developed as part of the Kaggle House Prices competition to demonstrate machine learning techniques for regression problems.*
