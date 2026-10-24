# Credit Card Customer Churn Analysis and Prediction

This project predicts which bank credit card customers are likely to leave (churn) and explores why. It covers exploratory data analysis (EDA), compares different ways of preparing the data, trains five machine learning models, and evaluates them both on standard metrics and on **business value**, using gain and lift charts.

I completed this project as part of the **KaggleX BIPOC Mentorship Program**. I wrote about the process in two Medium articles:

- 📖 [KaggleX BIPOC Program Experience: Comprehensive Guide to Customer Churn Analysis and Prediction](https://medium.com/@sidneyjoyclyn/kagglex-bipoc-program-experience-comprehensive-guide-to-customer-churn-analysis-and-prediction-eeb05f55bb48)
- 📖 [KaggleX BIPOC Mentorship Program Experience: Comprehensive Guide to Customer Churn Analysis and Prediction](https://medium.com/@sidneyjoyclyn/kagglex-bipoc-mentorship-program-experience-comprehensive-guide-to-customer-churn-analysis-and-d4cd5fb3a8a3)

## Notebooks

| Notebook | What it covers | Kaggle |
|----------|----------------|--------|
| [credit-card-churn-EDA.ipynb](credit-card-churn-EDA.ipynb) | Data cleaning, analysis of each variable, outliers, correlations and chi-square tests | [Open in Kaggle](https://www.kaggle.com/code/ujunwajoyclyn/credit-card-churn) |
| [credit-card-churn-datapreprocessing-model.ipynb](credit-card-churn-datapreprocessing-model.ipynb) | Encoding and scaling experiments, five models, threshold tuning, cross-validation, gain and lift | [Open in Kaggle](https://www.kaggle.com/code/ujunwajoyclyn/credit-card-churn-datapreprocessing-model) |

## Dataset

- **10,127 customers** of a bank's credit card service, with 20 features after cleaning
- The features cover demographics (age, gender, dependents, education, marital status, income), card category, how long the customer has been with the bank, number of products, inactivity, how often they contacted the bank, credit limit, revolving balance, and transaction amounts and counts
- The target is `Attrition_Flag`: an existing customer or one who has churned. **Only 16.07% of customers churned**, so the classes are imbalanced.

## Exploratory data analysis: key findings

- **More contact with the bank means more churn.** Customers who contacted the bank 5–6 times in a year churned more, which suggests these contacts may be complaints.
- **Higher earners churn less.** They also tend to have higher credit limits and more open-to-buy credit.
- **Most customers barely use their cards.** Many have a revolving balance of 0.
- **Low activity is a warning sign.** Churned customers tend to have lower transaction counts, lower transaction amounts, and a bigger drop in activity from Q4 to Q1.
- **Hidden customer segments:** total transaction amount has several peaks (it's multimodal), which suggests distinct groups of customers.
- **No single strong driver:** no feature is strongly correlated with churn. All correlations are between −0.4 and 0.2.

**Recommendation:** reduce the number of times customers need to contact the bank, for example by resolving their issues quickly, and use the model to reach at-risk customers before they leave.

## Modelling approach

1. **Encoding:** one-hot encoding compared with label encoding
2. **Scaling:** no scaling compared with Robust, Min-Max and Standard scaling, which gives **8 versions of the data**
3. **Models:** Logistic Regression, Random Forest, CatBoost, XGBoost and LightGBM, each trained on all 8 versions
4. **Threshold tuning:** precision-recall curves, the ROC curve and Youden's J statistic, to trade off precision against recall depending on the bank's retention strategy
5. **Overfitting checks:** training scores compared with test scores, plus 10-fold cross-validation
6. **Business evaluation:** cumulative gain and lift charts

## Results

Best data version for each model, on the test set:

| Model | Best data version | Accuracy | Precision | Recall | F1 |
|-------|-------------------|:--------:|:---------:|:------:|:--:|
| **LightGBM** | Label encoded + standard scaling | **0.968** | **0.912** | **0.893** | **0.902** |
| XGBoost | Label encoded | 0.964 | 0.897 | 0.879 | 0.888 |
| CatBoost | One-hot encoded | 0.957 | 0.918 | 0.810 | 0.861 |
| Random Forest | Label encoded | 0.912 | 0.917 | 0.510 | 0.655 |
| Logistic Regression | Label encoded + standard scaling | 0.901 | 0.775 | 0.556 | 0.648 |

- **LightGBM performed best overall.** 10-fold cross-validation gave an average accuracy of about 97%, which confirms it isn't overfitting.
- **Business value:** using LightGBM, the bank could reach **98% of churners by contacting just 20% of its customers**, with a lift of 4.9 at the second decile. Logistic Regression reaches 72% of churners for the same effort.
- **Threshold tuning on Logistic Regression:** the optimal cut-off from the ROC curve raised recall from 0.56 to 0.82, while precision fell from 0.78 to 0.56.

## Tech stack

Python · pandas · NumPy · SciPy · Matplotlib · Seaborn · scikit-learn · CatBoost · XGBoost · LightGBM · kds
