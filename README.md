# Diabetes Prediction Challenge - Kaggle Playground Series S5E12
Kaggle Comnpetition: https://www.kaggle.com/competitions/playground-series-s5e12/overview

A comprehensive machine learning pipeline for predicting diabetes diagnosis using ensemble methods and advanced imbalance handling techniques.

## Competition Overview

- **Competition**: [Kaggle Playground Series S5E12](https://www.kaggle.com/competitions/playground-series-s5e12)
- **Objective**: Predict diabetes diagnosis (binary classification)
- **Evaluation Metric**: ROC-AUC Score
- **Dataset Size**: 700,000 training samples, 300,000 test samples
- **Class Imbalance**: 62.3% diabetes (positive class) vs 37.7% healthy (negative class)

## Results

| Metric | Score |
|--------|-------|
| **Best Single Model** | 0.7288 AUC (ADASYN + LightGBM) |
| **Final Ensemble** | 0.7301 AUC (Elastic Net Meta-Learner) |
| **Improvement** | +0.13% over best single model |

## Pipeline Architecture

### 1. **Data Preprocessing**
- **Feature Engineering**:
  - Dropped low-information features: `ethnicity`, `gender`, `employment_status`
  - Ordinal encoding for risk-based features:
    - `income_level`: Low (0) → High (100)
    - `education_level`: No formal (0) → Postgraduate (100)
    - `smoking_status`: Current (0) → Never (100)
- **No missing values** in the dataset

### 2. **Imbalance Handling Strategies**

Tested 5 resampling techniques:
- **None**: Baseline with optional class weights
- **ADASYN**: Adaptive synthetic sampling (focuses on harder examples)
- **BorderlineSMOTE**: Synthetic oversampling near decision boundaries
- **RandomUnderSampler**: Reduces majority class
- **SMOTEENN**: Combined oversampling + Edited Nearest Neighbors cleaning

**Key Design Decision**: Class weights (`balanced`) only tested with "None" sampling to avoid double-correction.

### 3. **Model Selection**

Tested 3 gradient boosting algorithms:
- **LightGBM**: Fast, efficient, supports class weights
- **XGBoost**: Robust, uses `scale_pos_weight` for imbalance
- **CatBoost**: Handles categorical features, uses `auto_class_weights`

**Total Configurations Tested**: 18 (5 sampling strategies × 3 models, with class weights for baseline)

### 4. **Ensemble Methods (8 Strategies)**

#### Basic Methods:
1. **Simple Average**: Equal weight to all top models
2. **Weighted Average**: Weight by validation AUC
3. **Rank Average**: Robust to prediction outliers

#### Meta-Learning (Stacking):
4. **Ridge Regression**: L2 regularization, learns optimal linear combination
5. **Lasso Regression**: L1 regularization, automatic feature selection
6. **Elastic Net**: Combined L1+L2, best generalization (WINNER ✓)
7. **Logistic Regression**: Probability-calibrated predictions
8. **Constrained Optimization**: Direct AUC maximization via scipy.optimize

**Winner**: Elastic Net Meta-Learner achieved the best validation AUC (0.7301)

## Project Structure

```
DiabetesPredictionChallenge/
├── playground-series-s5e12/
│   ├── train.csv                    # 700k training samples
│   ├── test.csv                     # 300k test samples
│   └── sample_submission.csv
├── diabetes_prediction.ipynb        # Main notebook
├── submission_imbalance_best_model.csv  # Final submission
├── model_comparison_results.csv     # All 18 configurations
└── README.md                        # This file
```

## Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn
pip install lightgbm xgboost catboost
pip install imbalanced-learn pycaret
pip install scipy matplotlib seaborn
```

### Running the Pipeline

```python
# 1. Load and preprocess data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Apply feature engineering
train = clean_dataframe(train)
test = clean_dataframe(test)

# 3. Run the full pipeline (takes ~12 hours on 7 CPUs)
# See notebook for complete code

# 4. Generate predictions
submission.to_csv('submission.csv', index=False)
```

## Key Findings

### Best Performing Configurations

| Rank | Model | Sampling | Class Weight | Validation AUC |
|------|-------|----------|--------------|----------------|
| 1 | LightGBM | ADASYN | None | 0.7288 |
| 2 | CatBoost | None | None | 0.7288 |
| 3 | CatBoost | None | Balanced | 0.7287 |
| 4 | CatBoost | BorderlineSMOTE | None | 0.7284 |
| 5 | CatBoost | ADASYN | None | 0.7282 |

### Ensemble Weights (Elastic Net)

```
lightgbm_ADASYN    : 28.5%
catboost_None      : 27.6%
catboost_Bord      : 19.3%
catboost_ADASYN    : 15.0%
catboost_None      :  9.7%
```

### Insights

1. **ADASYN performed best** for oversampling (focuses on harder examples)
2. **CatBoost dominated top results** (6 of top 7 configs)
3. **Class weights alone** (without resampling) achieved competitive results
4. **SMOTEENN underperformed** badly (0.6998 AUC) - too aggressive cleaning
5. **Ensemble improved by 0.13%** - small but consistent gain

## Validation Strategy

- **Stratified split**: 90% train, 10% validation
- **5-fold cross-validation** within training set
- **Hyperparameter tuning**: 10 iterations per model using PyCaret
- **Early stopping** to prevent overfitting

## Technical Details

### Imbalance Handling Logic

```python
if sampling_strategy == "None":
    # Test both weighted and unweighted
    class_weights = ["None", "balanced"]
else:
    # Only test unweighted (avoid double-correction)
    class_weights = ["None"]
```

### Meta-Learning Pipeline

```python
# 1. Collect predictions from top N models
X_meta = [model.predict_proba(X_val)[:, 1] for model in top_models]

# 2. Train meta-learner
meta_model = ElasticNet(alpha=0.001, l1_ratio=0.5)
meta_model.fit(X_meta, y_val)

# 3. Generate final predictions
test_proba = meta_model.predict(X_test_meta)
```

### Performance Optimization

- **Multiprocessing**: Used 7/8 CPUs for parallel training
- **CatBoost fix**: Set `thread_count=1` to avoid Windows file locking
- **Total runtime**: ~12 hours for 18 configurations

## Visualization

Key plots included in notebook:
- Class distribution (pie chart)
- Feature distributions (train vs test)
- Correlation heatmap
- Missing value analysis
- Feature importance (planned)

## Future Improvements

1. **Feature Engineering**:
   - Interaction features (e.g., BMI × age)
   - Polynomial features for continuous variables
   - Domain-specific ratios (e.g., cholesterol ratios)

2. **Model Selection**:
   - Neural networks (TabNet, FT-Transformer)
   - Stacking with level-2 non-linear models
   - Hyperparameter optimization with Optuna

3. **Ensemble**:
   - Test more diverse base models
   - Cross-validation for meta-learner training
   - Weighted blending based on model diversity

4. **Validation**:
   - Out-of-fold predictions for meta-learning
   - Time-based split if temporal patterns exist
   - Adversarial validation

## References

- **PyCaret**: [https://pycaret.org/](https://pycaret.org/)
- **imbalanced-learn**: [https://imbalanced-learn.org/](https://imbalanced-learn.org/)
- **Kaggle Competition**: [Playground S5E12](https://www.kaggle.com/competitions/playground-series-s5e12)

## Author

Your Name - [GitHub Profile](https://github.com/yourusername)

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Kaggle for hosting the competition
- PyCaret team for the amazing AutoML library
- scikit-learn and imbalanced-learn communities

---

**Note**: This solution achieved a validation AUC of 0.7301. The actual leaderboard score may vary due to:
- Distribution shift between validation and test sets
- Overfitting to validation data (ensemble optimization)
- Random seed variation in sampling techniques

For reproducibility, ensure you use the same random seeds and library versions specified in `requirements.txt`.
