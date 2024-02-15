# Section 4: Solution and Data Monitoring

## Drift Types

### Compare and contrast label drift and feature drift

Label drift and feature drift are two types of data drift that can occur in machine learning models.

Label drift refers to a situation where the distribution of the target variable (or label) changes over time. This means that the true values of the label are different in the training data compared to the data that the model is being applied to. Label drift can occur in situations where the underlying phenomenon being modeled is changing over time, or when there are changes in the data collection process. Label drift can significantly impact the performance of a model, as the model may not be able to accurately predict the new values of the target variable.

Feature drift, on the other hand, refers to a situation where the distribution of the input features changes over time. This means that the patterns and relationships between the features and the target variable are different in the training data compared to the data that the model is being applied to. Feature drift can occur due to changes in the data collection process, changes in the underlying phenomenon being modeled, or changes in the relationship between the features and the target variable. Feature drift can also impact the performance of a model, as the model may not be able to capture the new patterns and relationships between the features and the target variable.

In summary, label drift and feature drift are both types of data drift that can impact the performance of machine learning models. Label drift refers to changes in the distribution of the target variable, while feature drift refers to changes in the distribution of the input features. Both types of drift can occur due to changes in the underlying phenomenon being modeled or changes in the data collection process. It is important to monitor and address both label drift and feature drift to ensure the accuracy and reliability of machine learning models.

### Identify scenarios in which feature drift and/or label drift are likely to occur

Scenarios in which feature drift and label drift are likely to occur include:

Concept Drift: When the statistical properties of the target variable change, causing the concept being predicted to change as well. For example, the definition of what is considered a fraudulent transaction may change over time.
Data Drift: When the statistical properties of the input data used to train the model change, affecting the model's quality. This could be due to changes in seasonality, trends, or personal preferences.
Upstream Data Changes: When there are operational changes in the data pipeline that impact the model's quality. This could include changes in feature encoding or missing values due to changes in data generation.
To detect and address model drift, it is important to monitor for changes and take action when they occur. This can be done by establishing a feedback loop from a monitoring system and refreshing models over time. Delta Lake can help detect data drift, while MLflow can be used to monitor model performance metrics and compare predicted and actual labels to detect drift in the predictive quality of the model. By monitoring for drift and taking appropriate actions, you can ensure that your models remain accurate and effective over time.

### Describe concept drift and its impact on model efficacy
Concept drift refers to the phenomenon where the statistical properties of the target variable or the input features change over time. In other words, the underlying data distribution on which a model was trained may no longer accurately represent the data distribution in the real world.

Concept drift can have a significant impact on the efficacy of machine learning models. When a model is trained on historical data that does not accurately reflect the current data distribution, the model's predictions may become less accurate over time. This is because the model's assumptions about the relationships between the input features and the target variable may no longer hold true.

The impact of concept drift on model efficacy can manifest in several ways:

Decreased accuracy: As the data distribution changes, the model's predictions on new, unseen data may become less accurate. This can lead to a decrease in overall model performance.

Increased false positives/negatives: Concept drift can cause the model to make more false positive or false negative predictions. For example, a fraud detection model trained on historical data may fail to identify new types of fraudulent activities that were not present in the training data.

Degraded performance over time: Models that are not regularly updated to account for concept drift may experience a gradual decline in performance as the data distribution shifts further away from the training data.

To mitigate the impact of concept drift, it is important to continuously monitor the performance of machine learning models and regularly retrain them with new data. Techniques such as online learning, active learning, and ensemble methods can also be employed to adapt the model to changing data distributions. Additionally, domain knowledge and expert input can help identify and address concept drift by updating the model's features, algorithms, or training data.

## Drift Tests and Monitoring

(https://www.databricks.com/blog/2019/09/18/productionizing-machine-learning-from-deployment-to-drift-detection.html)

### Describe summary statistic monitoring as a simple solution for numeric feature 

Summary statistic monitoring is a simple solution for monitoring numeric feature drift. It involves routine statistical testing of feature distributions and logging the results with MLflow. By comparing runs, you can easily detect changes in the shape of feature and target distributions. If a distribution breaches a threshold value, an alert can trigger a training job to automatically redeploy a new version. This approach helps maintain the validity and efficacy of a model in production.

### Describe mode, unique values, and missing values as simple solutions for categorical feature drift

### Describe tests as more robust monitoring solutions for numeric feature drift than simple summary statistics

### Describe tests as more robust monitoring solutions for categorical feature drift than simple summary statistics

### Compare and contrast Jenson-Shannon divergence and Kolmogorov-Smirnov tests for numerical drift detection

Jensen-Shannon divergence (JSD) and Kolmogorov-Smirnov (KS) test are both statistical methods used for numerical drift detection, but they have different approaches and assumptions.

Jensen-Shannon divergence is a measure of similarity between two probability distributions. It quantifies the difference between two probability distributions by computing the average of their Kullback-Leibler divergences. In the context of drift detection, JSD can be used to compare the distributions of two datasets collected at different time points or from different sources. A low JSD value indicates a high similarity between the distributions, while a high JSD value suggests a significant drift.

Kolmogorov-Smirnov test, on the other hand, is a non-parametric statistical test that assesses the difference between two cumulative distribution functions (CDFs). It compares the maximum vertical distance between the two CDFs, also known as the Kolmogorov-Smirnov statistic. The test assumes that the two datasets being compared are independent and identically distributed. In drift detection, the KS test can be used to determine if there is a significant difference between the two datasets. If the p-value of the test is below a certain threshold (e.g., 0.05), it suggests a drift in the data.

In summary, Jenson-Shannon divergence measures the difference between probability distributions, while the Kolmogorov-Smirnov test compares the CDFs of two datasets. JSD is more flexible and can handle any type of probability distribution, while the KS test assumes independent and identically distributed data. Both methods are useful for numerical drift detection, but the choice between them depends on the specific requirements and characteristics of the data.

### Identify a scenario in which a chi-square test would be useful


## Comprehensive Drift Solutions

### Describe a common workflow for measuring concept drift and feature drift

### Identify when retraining and deploying an updated model is a probable solution to drift

### Test whether the updated model performs better on the more recent data
