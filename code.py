# --------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Path variable


# Load the data
df = pd.read_csv(path)

# First 5 columns
print((df.iloc[:, :5]).head())
df.drop(columns = "Unnamed: 0", axis = 1, inplace = True)

# Independent variables
X = df.drop(columns = "SeriousDlqin2yrs", axis = 1).copy()
X.head()

# Dependent variable
y = df.SeriousDlqin2yrs
y.head()

# Check the value counts
count = y.value_counts()

# Split the data set into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6) 


# --------------
# save list of all the columns of X in cols
cols = (X_train.columns).tolist()

# create subplots
fig, axes = plt.subplots(figsize=(10,20), nrows = 5, ncols = 2)

# nested for loops to iterate over all the features and plot the same
for i in range(5):
    for j in range (2):
        col = cols[i * 2 + j]
        axes[i, j].scatter(X_train[col], y_train)
        axes[i, j].set_title(col)
        
fig.tight_layout()
plt.show()




# --------------
# Check for null values
X_train.isnull().sum()
print("===================================================")

# Filling the missing values for columns in training data set
X_train["MonthlyIncome"].fillna(X_train["MonthlyIncome"].median(), inplace=True)
X_train["NumberOfDependents"].fillna(X_train["NumberOfDependents"].median(), inplace=True)

# Filling the missing values for columns in testing data set
X_test["MonthlyIncome"].fillna(X_test["MonthlyIncome"].median(), inplace=True)
X_test["NumberOfDependents"].fillna(X_test["NumberOfDependents"].median(), inplace=True)

# Checking for null values
X_train.isnull().sum()
print("===================================================")
X_test.isnull().sum()
print("===================================================")


# --------------
# Correlation matrix for training set
corr = X_train.corr()

# Plot the heatmap of the correlation matrix
sns.heatmap(corr)
fig.show()

# drop the columns which are correlated amongst each other except one
X_train.drop(columns = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse"], axis = 1, inplace = True)
X_test.drop(columns = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse"], axis = 1, inplace = True)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)


# --------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --------------
# Import Logistic regression model and accuracy score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Instantiate the model in a variable in log_reg
log_reg = LogisticRegression()

# Fit the model on training data
log_reg.fit(X_train, y_train)

# Predictions of the training dataset
y_pred = log_reg.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy: ", round(accuracy,4))



# --------------
# Import all the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score

# Plot the auc-roc curve
score = roc_auc_score(y_pred, y_test)
print("ROC-AUC Score: ", round(score, 4))

y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
print("FPR, TPR :", fpr, ",", tpr)

# Evaluation parameters for the model
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC Score: ", round(auc, 4))

plt.plot(fpr, tpr, label="Logistic model, auc="+str(auc))
plt.show()

f1 = f1_score(y_test, y_pred)
print("F1 Score: \n", round(f1, 4))
precision = precision_score(y_test, y_pred)
print("Precision Score: \n", round(precision, 4))
recall = recall_score(y_test, y_pred)
print("Recall Score: \n", round(recall, 4))
rocscore = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score: \n", round(rocscore, 4))

cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cf)

print("Classification Report:")
print(classification_report(y_test, y_pred))



# --------------
# Import SMOTE from imblearn library
from imblearn.over_sampling import SMOTE

# Check value counts of target variable for data imbalance
print(y_train.value_counts())

# Instantiate smote
smote = SMOTE(random_state = 9)

# Fit Smote on training set
X_sample, y_sample = smote.fit_sample(X_train , y_train)

# Check for count of class
print("Count Plot")
sns.countplot(x = y_sample)


# --------------
# Fit logistic regresion model on X_sample and y_sample
log_reg.fit(X_sample, y_sample)

# Store the result predicted in y_pred
y_pred = log_reg.predict(X_test)

# Store the auc_roc score
score = roc_auc_score(y_pred, y_test)
print("ROC-AUC Score: ", round(score, 4))

# Store the probablity of any class
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
print(fpr, tpr)

# Plot the auc_roc_graph
auc = roc_auc_score(y_test, y_pred_proba)
print(round(auc, 4))
plt.plot(fpr, tpr, label="Logistic model, auc="+str(auc))
plt.show()

# Print f1_score,Precision_score,recall_score,roc_auc_score and confusion matrix
f1 = f1_score(y_test, y_pred)
print("F1 Score: \n", round(f1, 4))
precision = precision_score(y_test, y_pred)
print("Precision Score: \n", round(precision, 4))
recall = recall_score(y_test, y_pred)
print("Recall Score: \n", round(recall, 4))
rocscore = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score: \n", round(rocscore, 4))

cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cf)

print("Classification Report:")
print(classification_report(y_test, y_pred))


# --------------
# Import RandomForestClassifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Instantiate RandomForrestClassifier to a variable rf.
rf = RandomForestClassifier(random_state = 9)

# Fit the model on training data.
rf.fit(X_sample, y_sample)

# store the predicted values of testing data in variable y_pred.
y_pred = rf.predict(X_test)

# Store the different evaluation values.
f1 = f1_score(y_test, y_pred)
print("F1 Score: \n", round(f1, 4))
precision = precision_score(y_test, y_pred)
print("Precision Score: \n", round(precision, 4))
recall = recall_score(y_test, y_pred)
print("Recall Score: \n", round(recall, 4))
rocscore = roc_auc_score(y_test, y_pred)
print("ROC-AUC Score: \n", round(rocscore, 4))

cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cf)

print("Classification Report:")
print(classification_report(y_test, y_pred))


y_pred_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
print(fpr, tpr)

auc = roc_auc_score(y_test, y_pred_proba)
print(round(auc, 4))

# Plot the auc_roc graph
plt.plot(fpr, tpr, label="Logistic model, auc="+str(auc))
plt.show()


