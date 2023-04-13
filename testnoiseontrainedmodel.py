import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# read from the data from a CSV file
data = pd.read_csv('heartuci.csv')
X = data.drop(columns=['target'])
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train five different classifiers on the raw dataset
classifiers = [
    DecisionTreeClassifier(random_state=42),
    LogisticRegression(random_state=42),
    RandomForestClassifier(random_state=42),
    GaussianNB()
]
clf_names = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Naive Bayes']

print("Classifier\tAccuracy (Raw)\tPrecision (Raw)\tRecall (Raw)\tAccuracy (Noisy)\tPrecision (Noisy)\tRecall (Noisy)")

for clf, clf_name in zip(classifiers, clf_names):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_raw = accuracy_score(y_test, y_pred)
    precision_raw = precision_score(y_test, y_pred)
    recall_raw = recall_score(y_test, y_pred)

    # Adding noise Laplace noise to specific columns of the dataset
    col_names = ['age', 'time']
    sensitivities = [1.0, 1.0]
    epsilons = [0.1, 0.1]              
    X_train_noisy = X_train.copy()
    X_test_noisy = X_test.copy()
    for col_name, sensitivity, epsilon in zip(col_names, sensitivities, epsilons):
        scale = sensitivity / epsilon
        noise = np.random.laplace(scale=scale, size=X_train.shape[0])
        X_train_noisy[col_name] += noise
        X_test_noisy[col_name] += noise[:X_test.shape[0]]

    y_pred_noisy = clf.predict(X_test_noisy)
    accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
    precision_noisy = precision_score(y_test, y_pred_noisy)
    recall_noisy = recall_score(y_test, y_pred_noisy)

    # Print results as table
    print(f"{clf_name}\t{accuracy_raw:.4f}\t{precision_raw:.4f}\t{recall_raw:.4f}\t{accuracy_noisy:.4f}\t{precision_noisy:.4f}\t{recall_noisy:.4f}")

