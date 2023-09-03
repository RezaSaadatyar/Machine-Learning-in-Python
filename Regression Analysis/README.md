**Regression Analysis**<br/>
Regression is a technique used to analyze the connection between independent variables or features and a dependent variable or outcome. Once the relationship between the independent and dependent variables has been determined, it becomes possible to make predictions about the outcomes.

**Types of Regression Analysis:**
 - `Simple Linear Regression:` This form of regression is utilized to depict the correlation between one independent variable and a reliant variable. For example, studying the relationship between the number of hours of exercise and weight loss.
 - `Multiple Linear Regression:` This regression technique is utilized to depict the correlation between a reliant variable and multiple independent variables. For example, analyzing the influence of multiple lifestyle variables (e.g., diet, exercise, smoking) on a person's risk of developing a particular health condition.
 - `Polynomial Regression:` Polynomial regression is a regression method that involves fitting a polynomial equation to the data in order to model the relationship between a dependent variable and an independent variable. For example, This approach is used in predicting the rate of spread of diseases like COVID-19 and other infectious diseases or modeling the growth of tumors based on factors like time and treatment dosage, as tumor growth may not follow a linear pattern.
 - `Logistic Regression:` This regression technique is utilized to represent the correlation between a binary dependent variable and one or more independent variables. For example, predicting whether a patient has a particular disease (e.g., diabetes, cancer) based on medical test results and patient characteristics.
 - `Ridge Regression:` It is a method of multiple linear regression that includes a penalty term in the cost function to avoid overfitting. For example, analyzing medical images (e.g., MRI, CT scans) to detect and diagnose diseases. Ridge Regression can be used to build models that incorporate multiple image features and patient data to improve diagnostic accuracy.
 - `Lasso Regression:` It is a type of regression used for feature selection in multiple linear regression. It adds a penalty term to the cost function, which encourages some independent variable coefficients to be exactly zero. For example, analyzing medical images (e.g., radiographs, MRIs, CT scans) for disease detection or progression. Lasso Regression can be used to select image features that are most indicative of specific pathologies or conditions.
- `ElasticNet Regression:` This method of regression analysis merges the L1 and L2 regularization approaches of Lasso and Ridge regression, respectively. It is applied in situations where there are more independent variables than observations or when the independent variables have a high level of correlation with one another. Its primary use is in linear regression problems. For example, analyzing medical images, such as MRI or CT scans, to discover imaging features that are associated with specific disease outcomes or treatment responses. ElasticNet can select relevant imaging features and control for their interdependencies.
- `Bayesian Regression:` Bayesian Regression uses a prior distribution to represent the researcher's belief about the parameters before data is collected. After data is observed, the prior distribution is updated to become the posterior distribution, representing the researcher's belief about the parameters after observing the data. For example, modeling the progression of chronic diseases over time, such as Alzheimer's disease, using Bayesian state-space models. These models can capture disease dynamics, incorporate prior knowledge about disease mechanisms, and make probabilistic predictions about disease stages

**Challenges:**
- `Overfitting:` Overfitting happens when the model is excessively complex and fits the training data too closely, leading to poor performance on new or unseen data. To address this issue, regularization techniques such as L1 and L2 regularization or early stopping can be used.
- `Underfitting:` Underfitting happens when the model is overly simplistic and is unable to accurately capture the underlying patterns in the data. This issue can be resolved by either increasing the model's complexity or introducing additional pertinent features.
- `Multicollinearity:` Multicollinearity happens when two or more independent variables are strongly correlated. This can cause problems in determining the individual impact of each variable and may result in unstable parameter estimates.
- `Non-linearity:` Regression models assume a linear relationship between independent and dependent variables, but non-linear relationships can result in inaccurate predictions.
- `Outliers:` Outliers are data points that differ significantly from the majority of the data and can have a significant impact on regression models, leading to inaccurate parameter estimates.

**Evaluation metrics:**
- `Mean Absolute Error (MAE):` MAE is a commonly used metric in statistics and machine learning to measure the average absolute difference between the actual values and the predicted values in a dataset.
- `Mean Absolute Percentage Error (MAPE):` MAPE measures the average percentage difference between the actual values and the predicted values. It is expressed as a percentage and is useful for understanding the magnitude of errors in relation to the actual values. However, MAPE has some limitations, such as being sensitive to zero or very small actual values (which can result in division by zero) and not penalizing large errors proportionally. 
- `Mean Squared Error (MSE):` It measures the average of the squared differences between the actual values and the predicted values in a dataset. MSE is particularly useful when you want to penalize larger errors more heavily than smaller errors.
- `Root Mean Squared Error (RMSE):` RMSE measures the square root of the average of the squared differences between the actual values and the predicted values in a dataset. It is particularly useful when you want to express the error metric in the same unit as the target variable and penalize larger errors more heavily.
 - `Coefficient of Determination (R²):` R-squared is a measure of the proportion of variance in the target variable explained by independent variables. It indicates the model's fit to the data.

| Metric                         | Description                                          | Formula                                                                  | Range                 | Interpretation                                         |
|--------------------------------|------------------------------------------------------|--------------------------------------------------------------------------|-----------------------|--------------------------------------------------------|
| MAE      | Average absolute difference between actual and predicted values. |$$MAE = \frac{1}{n} \sum_{i=1}^{n}abs(y_i - \hat{y}_i)$$| 0 to ∞               | Lower values indicate better accuracy.                |
| MAPE | Average percentage difference between actual and predicted values. |$$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{abs(y_i - \hat{y}_i)}{abs(y_i)}\right) \times 100$$| 0% to ∞ (no upper bound) | Represents errors as a percentage of actual values. |
| MSE       | Average of squared differences between actual and predicted values. |$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$      | 0 to ∞               | Larger errors are more heavily penalized.             |
| RMSE | Square root of MSE, providing an error metric in the same units as the target variable. |$$RMSE = \sqrt{\text{MSE}}$$| 0 to ∞               | Penalizes larger errors while keeping units interpretable. |
|$R^2$ | Measures the proportion of variance in the dependent variable explained by independent variables. |$$R^2 = 1 - \frac{\text{SSR}}{\text{SST}}$$ $$SST = \sum_{i=1}^{n} (y_i - \bar{y})^2$$ $$SSR = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ | 0 to 1 (or 0% to 100%) | Higher values indicate better fit and explanatory power. |

Where:<br/>
n is the number of data points in the dataset; $y_{i}$ represents the actual or observed value for the i-th data point; $\hat{y}_i$ represents the predicted value for the i-th data point.<br/>
1. **MAE vs. MSE/RMSE:**
   - **MAE** treats all errors equally and provides a simple, interpretable measure.
   - **MSE/RMSE** squares errors, giving more weight to larger errors, making them sensitive to outliers.
2. **MSE vs. RMSE:**
   - **RMSE** is the square root of MSE, providing an error metric in the same units as the target variable.
   - **MSE** gives errors in squared units, which may not be as interpretable.
3. **MAE/MSE/RMSE vs. $R^2$:**
   - **MAE/MSE/RMSE** focus on error magnitude and precision.
   - $R^2$ assesses the overall model fit and explanatory power, quantifying the proportion of variance explained by the model.
4. **$R^2$ vs. MAE/MSE/RMSE:**
   - **$R^2$** ranges from 0 to 1 (or 0% to 100%) and represents the proportion of variance explained.
   - **MAE/MSE/RMSE** provide absolute error measures but do not consider variance explained.

