#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# Importing the dataset
dataset = pd.read_csv("adult.csv")

def summerize_data(df):
    for column in df.columns:
        print(column)
        if df.dtypes[column] == np.object: # Categorical data
            print(df[column].value_counts()) 
            print("Number of unique values: ", df[column].nunique())  
        else:
            print(df[column].describe())
        print("\n")

dataset.info()
summerize_data(dataset)

#putting the missing values into a new category called "Others"
#Grouping some categories

sns.countplot(hue='income', y='race', data=dataset,)
dataset["race"]=dataset["race"].replace(["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other" ], "Others")
sns.countplot(y='race', hue='income', data=dataset,)

sns.countplot(y='native-country', data=dataset,)
#distribution is highly skewed.
dataset["native-country"]=dataset["native-country"].replace(["Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?" ], "Others")
sns.countplot(y='native-country', data=dataset,)

sns.countplot(y='workclass', hue='income', data=dataset,)
dataset["workclass"]=dataset["workclass"].replace(["Without-pay", "Never-worked", "?" ], "Others")
sns.countplot(y='workclass', hue='income', data=dataset,)

sns.countplot(y='marital-status', hue='income', data=dataset,)
dataset["marital-status"]=dataset["marital-status"].replace(["Married-spouse-absent", "Married-AF-spouse", "Separated", "Widowed", "Divorced" ], "Others")
sns.countplot(y='marital-status', hue='income', data=dataset,)

sns.countplot(y='occupation', hue='income', data=dataset,)
dataset["occupation"]=dataset["occupation"].replace(["?", "Tech-support", "Protective-serv", "Armed-Forces", "Other-service", "Handlers-cleaners", "Machine-op-inspct", "Priv-house-serv"], "Others")
sns.countplot(y='occupation', hue='income', data=dataset,)

sns.countplot(y='income', hue='income', data=dataset,)

dataset["age"].plot.kde()
dataset["educational-num"].plot.kde()
dataset["hours-per-week"].plot.kde()
dataset["fnlwgt"].plot.kde()
dataset["capital-gain"].plot.kde()
dataset["capital-loss"].plot.kde()


# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders
    
    
#Calculate the correlation
encode_for_corr, _ = number_encode_features(dataset)
sns.heatmap(encode_for_corr.corr(), square=True)

#they are the same data
sns.countplot(y='education', hue='income', data=dataset,)
sns.countplot(y='educational-num', hue='income', data=dataset,)
dataset.drop("education", axis=1, inplace=True)

#negative correlation between "gender" and "relationship"
#"Female" and "wife" and anti-correlated, so are "Male" and "Husband"
#Females are likely to be wives and males are likely to be husbands
sns.countplot(y='gender', hue='income', data=dataset,)
sns.countplot(y='relationship', hue='income', data=dataset,)
dataset.drop("relationship", axis=1, inplace=True)

binary_data = pd.get_dummies(dataset)
binary_data["income"] = binary_data["income_>50K"]
binary_data.drop("income_<=50K", axis=1, inplace=True)
binary_data.drop("income_>50K", axis=1, inplace=True)


plt.subplots(figsize=(20,20))
sns.heatmap(binary_data.corr(), square=True)
plt.show()



# Avoiding the Dummy Variable Trap
binary_data.drop("workclass_Others", axis=1, inplace=True)
binary_data.drop("marital-status_Others", axis=1, inplace=True)
binary_data.drop("occupation_Others", axis=1, inplace=True)
binary_data.drop("race_Others", axis=1, inplace=True)
binary_data.drop("gender_Female", axis=1, inplace=True)
binary_data.drop("native-country_Others", axis=1, inplace=True)



X = binary_data.iloc[:, :-1].values #dropping the last column
# int output istersek:  X = X.iloc[:, :-1].values
y = binary_data.iloc[:, -1] #creating y



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scale the data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#confusion matrix as fractions
def confusion_matrix_fr(confusion_matrix, as_fractions=False):
    if as_fractions:
        x = np.array(confusion_matrix)
        x = np.apply_along_axis(
            lambda row: [
                row[0] / (row[0] + row[1]),
                row[1] / (row[0] + row[1])
            ],
            1,
            x
        )
    else:
        x = confusion_matrix
    df = pd.DataFrame(
        x,
        index=["<= 50K", "> 50K"],
        columns=["<= 50K", "> 50K"]
    )
    df.index.names = ["Actual"]
    df.columns.names = ["Predicted"]
    return df

#Logistic Regression
#Fitting Logistic Regression to the Training set with solver = "newton-cg"
classifier = LogisticRegression(random_state = 0, solver = "liblinear")
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_logreg1 = classifier.predict(X_test)
y_train_logreg1 =classifier.predict(X_train)
# Making the Confusion Matrix
cm_test_logreg1 = confusion_matrix(y_test, y_pred_logreg1)
cm_train_logreg1 = confusion_matrix(y_train, y_train_logreg1)
cm_test_logreg1_fr = confusion_matrix_fr(cm_test_logreg1, as_fractions=True)
cm_train_logreg1_fr = confusion_matrix_fr(cm_train_logreg1, as_fractions=True)
acc_test_logreg1 = accuracy_score(y_test, y_pred_logreg1)
acc_train_logreg1 = accuracy_score(y_train, y_train_logreg1)
rec_test_logreg1 = recall_score(y_test, y_pred_logreg1)
rec_train_logreg1 = recall_score(y_train, y_train_logreg1)

print(classification_report(y_test, y_pred_logreg1))
print(classification_report(y_train, y_train_logreg1))    


# Building the optimal model using Backward Elimination
import statsmodels.api as sm
import statsmodels.formula.api as smf

X1 = np.append(arr = np.ones((48842, 1)).astype(int), values = X, axis = 1)
#adds a coln of 1's to the front, then we'll remove one by one the indep values that are not statistically significant
X_opt = X1[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
#select a significance level SL to stay in the model 0.05
#if p-value of a indep var is below then it will stay
#if above then will be removed
SL = 0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# for x16 we have 0.633, so we'll remove 16
X_opt = X1[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# for x8 we have 0.487, so we'll remove 8
X_opt = X1[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# for x13 we have 0.368, so we'll remove 14
X_opt = X1[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# for x11 we have 0.057, so we'll remove 12
X_opt = X1[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#all below 0.05, stop the elimination process


# Splitting the dataset into the Training set and Test set
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_opt, y, test_size = 0.2, random_state = 60)

#Scale the data
sc_X = StandardScaler()
X_train2 = sc_X.fit_transform(X_train2)
X_test2 = sc_X.transform(X_test2)

#Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train2, y_train2)
# Predicting the Test set results
y_pred_logreg2 = classifier.predict(X_test2)
y_train_logreg2 =classifier.predict(X_train2)
# Making the Confusion Matrix
cm_test_logreg2 = confusion_matrix(y_test2, y_pred_logreg2)
cm_train_logreg2= confusion_matrix(y_train2, y_train_logreg2)
cm_test_logreg2_fr = confusion_matrix_fr(cm_test_logreg2, as_fractions=True)
cm_train_logreg2_fr = confusion_matrix_fr(cm_train_logreg2, as_fractions=True)
acc_test_logreg2 = accuracy_score(y_test2, y_pred_logreg2)
acc_train_logreg2 = accuracy_score(y_train2, y_train_logreg2)
rec_test_logreg2 = recall_score(y_test2, y_pred_logreg2)
rec_train_logreg2 = recall_score(y_train2, y_train_logreg2)

print(classification_report(y_test2, y_pred_logreg2))
print(classification_report(y_train2, y_train_logreg2))



# Random Forest
#Fitting Random Forest to the Training set
classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy', random_state = 0, max_depth=7)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_ranfor_ent = classifier.predict(X_test)
y_train_ranfor_ent = classifier.predict(X_train)
# Making the Confusion Matrix
cm_test_ranfor_ent = confusion_matrix(y_test, y_pred_ranfor_ent)
cm_train_ranfor_ent = confusion_matrix(y_train, y_train_ranfor_ent)
cm_test_ranfor_ent_fr = confusion_matrix_fr(cm_test_ranfor_ent, as_fractions=True)
cm_train_ranfor_ent_fr = confusion_matrix_fr(cm_train_ranfor_ent, as_fractions=True)
acc_test_ranfor_ent = accuracy_score(y_test, y_pred_ranfor_ent)
acc_train_ranfor_ent = accuracy_score(y_train, y_train_ranfor_ent)
rec_test_ranfor_ent = recall_score(y_test, y_pred_ranfor_ent)
rec_train_ranfor_ent = recall_score(y_train, y_train_ranfor_ent)

print(classification_report(y_test, y_pred_ranfor_ent))
print(classification_report(y_train, y_train_ranfor_ent))

# Compute the Random Forest model for 25, 31, 37, 43 estimators
accuracy_ranfor_ent = []
recall_ranfor_ent = []
estimators=[25, 31, 37, 43] #tek al ve 25 ten basla
for i in estimators:
    classifier = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 0, max_depth=7)
    classifier.fit(X_train, y_train)
    y_pred_ranfor_ent_i = classifier.predict(X_test)
    accuracy_ranfor_ent.append(accuracy_score(y_test, y_pred_ranfor_ent_i))
    recall_ranfor_ent.append(recall_score(y_test, y_pred_ranfor_ent_i))

fig1=plt.plot(estimators,accuracy_ranfor_ent)
fig1=plt.title("Random Forest - Accuracy vs Number of estimators")
fig1=plt.xlabel("Estimators")
fig1=plt.ylabel("Accuracy")
fig1.figure.savefig('foo1.png', bbox_inches='tight')

fig1=plt.plot(estimators,recall_ranfor_ent)
fig1=plt.title("Random Forest - Recall vs Number of estimators")
fig1=plt.xlabel("Estimators")
fig1=plt.ylabel("Recall")
fig1.figure.savefig('foo1.png', bbox_inches='tight')



# Compute the Random Forest model for 25, 31, 37, 43 estimators
accuracy_ranfor_ent_dept = []
recall_ranfor_ent_dept = []
max_dept=[3, 5, 7, 11, 17, 21] #tek al ve 25 ten basla
for i in max_dept:
    classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy', random_state = 0, max_depth=i)
    classifier.fit(X_train, y_train)
    y_pred_ranfor_ent_dept_i = classifier.predict(X_test)
    accuracy_ranfor_ent_dept.append(accuracy_score(y_test, y_pred_ranfor_ent_dept_i))
    recall_ranfor_ent_dept.append(recall_score(y_test, y_pred_ranfor_ent_dept_i))

fig1=plt.plot(max_dept,accuracy_ranfor_ent_dept)
fig1=plt.title("Random Forest - Accuracy vs Number of Max Depth")
fig1=plt.xlabel("Max_depth")
fig1=plt.ylabel("Accuracy")
fig1.figure.savefig('foo1.png', bbox_inches='tight')








#Fitting Random Forest to the Training set
classifier = RandomForestClassifier(n_estimators = 25, criterion = 'gini', random_state = 0, max_depth=7)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_ranfor_gini = classifier.predict(X_test)
y_train_ranfor_gini = classifier.predict(X_train)
# Making the Confusion Matrix
cm_test_ranfor_gini = confusion_matrix(y_test, y_pred_ranfor_gini)
cm_train_ranfor_gini = confusion_matrix(y_train, y_train_ranfor_gini)
cm_test_ranfor_gini_fr = confusion_matrix_fr(cm_test_ranfor_gini, as_fractions=True)
cm_train_ranfor_gini_fr = confusion_matrix_fr(cm_train_ranfor_gini, as_fractions=True)
acc_test_ranfor_gini = accuracy_score(y_test, y_pred_ranfor_gini)
acc_train_ranfor_gini = accuracy_score(y_train, y_train_ranfor_gini)
rec_test_ranfor_gini = recall_score(y_test, y_pred_ranfor_gini)
rec_train_ranfor_gini = recall_score(y_train, y_train_ranfor_gini)

print(classification_report(y_test, y_pred_ranfor_gini))
print(classification_report(y_train, y_train_ranfor_gini))


#KNN
# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn_5 = classifier.predict(X_test)
y_train_knn_5 = classifier.predict(X_train)
# Making the Confusion Matrix
cm_test_knn_5 = confusion_matrix(y_test, y_pred_knn_5)
cm_train_knn_5 = confusion_matrix(y_train, y_train_knn_5)
cm_test_knn_5_fr = confusion_matrix_fr(cm_test_knn_5, as_fractions=True)
cm_train_knn_5_fr = confusion_matrix_fr(cm_train_knn_5, as_fractions=True)
acc_test_knn_5 = accuracy_score(y_test, y_pred_knn_5)
acc_train_knn_5 = accuracy_score(y_train, y_train_knn_5)
rec_test_knn_5 = recall_score(y_test, y_pred_knn_5)
rec_train_knn_5 = recall_score(y_train, y_train_knn_5)

print(classification_report(y_test, y_pred_knn_5))
print(classification_report(y_train, y_train_knn_5))

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn_15 = classifier.predict(X_test)
y_train_knn_15 = classifier.predict(X_train)
# Making the Confusion Matrix
cm_test_knn_15 = confusion_matrix(y_test, y_pred_knn_15)
cm_train_knn_15 = confusion_matrix(y_train, y_train_knn_15)
cm_test_knn_15_fr = confusion_matrix_fr(cm_test_knn_15, as_fractions=True)
cm_train_knn_15_fr = confusion_matrix_fr(cm_train_knn_15, as_fractions=True)
acc_test_knn_15 = accuracy_score(y_test, y_pred_knn_15)
acc_train_knn_15 = accuracy_score(y_train, y_train_knn_15)
rec_test_knn_15 = recall_score(y_test, y_pred_knn_15)
rec_train_knn_15 = recall_score(y_train, y_train_knn_15)

print(classification_report(y_test, y_pred_knn_15))
print(classification_report(y_train, y_train_knn_15))

# Compute the KNN model for 3,5,15 & 25 neighbors
accuracy_knn_test = []
accuracy_knn_train = []
recall_knn_test = []
recall_knn_train = []
neighbors = [3,5,15,25] #tek al 1 kullanma
for i in neighbors:
    classifier = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    y_pred_knn_i = classifier.predict(X_test)
    y_train_knn_i = classifier.predict(X_train)
    accuracy_knn_test.append(accuracy_score(y_test, y_pred_knn_i))
    accuracy_knn_train.append(accuracy_score(y_train, y_train_knn_i))
    recall_knn_test.append(recall_score(y_test, y_pred_knn_i))
    recall_knn_train.append(recall_score(y_train, y_train_knn_i))

    

fig1=plt.plot(neighbors,accuracy_knn_test)
fig1=plt.title("KNN - Accuracy(Test) vs Number of Neighbors")
fig1=plt.xlabel("Neighbors")
fig1=plt.ylabel("Accuracy")
fig1.figure.savefig('foo1.png', bbox_inches='tight')

fig1=plt.plot(neighbors,recall_knn_test)
fig1=plt.title("KNN - Recall(Test) vs Number of Neighbors")
fig1=plt.xlabel("Neighbors")
fig1=plt.ylabel("Recall")
fig1.figure.savefig('foo1.png', bbox_inches='tight')



fig1=plt.plot(neighbors,accuracy_knn_train)
fig1=plt.title("KNN - Accuracy(Train) vs Number of Neighbors")
fig1=plt.xlabel("Neighbors")
fig1=plt.ylabel("Accuracy")
fig1.figure.savefig('foo1.png', bbox_inches='tight')

fig1=plt.plot(neighbors,recall_knn_train)
fig1=plt.title("KNN - Recall(Train) vs Number of Neighbors")
fig1=plt.xlabel("Neighbors")
fig1=plt.ylabel("Recall")
fig1.figure.savefig('foo1.png', bbox_inches='tight')


