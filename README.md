### Approach taken to solve the problem

 Steps Involed:

1. Identify how our features are related to the dependent variable and get some insights about the data using scatter plot.
2. Check for missing values and replace them with appropriate values.
3. Find the features which are highly correlated and select the most significant features amongst them.
4. If the feature’s variance is orders of magnitude more than the variance of other features, that particular feature might dominate other features in the dataset and our model will not train well which gives us bad model. Hence scale the features.
5. Train our data on a Logistic regression algorithm and try to correctly predict the class whether somebody will face financial distress in the next two years, and evaluate the model.
6. Since this is an imbalanced dataset, use the SMOTE technique and check for different evaluation metrics.
7. The performance of our Logistic regression model has signifiacntly improved after balancing the data, furthur try to improve it by using a Random Forrest model.


