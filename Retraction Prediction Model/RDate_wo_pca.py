import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('retractions35215.csv').sample(n=2000, random_state=42)

# Convert RetractionDate to datetime with a specified format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], format='%m/%d/%Y', errors='coerce')

# Extract year from RetractionDate
data['RetractionYear'] = data['RetractionDate'].dt.year

# Drop rows with missing RetractionYear
data = data.dropna(subset=['RetractionYear'])

# Select features
features = ['Title', 'Subject', 'Journal', 'Publisher', 'ArticleType', 'Reason', 'CitationCount']

X = data[features]

# Target variable
y = data['RetractionYear']

# Handle missing values
X = X.fillna('Unknown')

# Encode categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# Predict on the test set with Random Forest
y_pred_rf = clf_rf.predict(X_test)

# Calculate the accuracy score 
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy score of Random Forest Classifier: ", accuracy_rf)

# Train a Logistic Regression model
clf_lr = LogisticRegression(max_iter=100, random_state=42)
clf_lr.fit(X_train, y_train)

# Predict on the test set with Logistic Regression
y_pred_lr = clf_lr.predict(X_test)

# Calculate the accuracy score
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Accuracy score of Logistic Regression: ", accuracy_lr)

# Define the SVM kernels to evaluate
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Evaluate SVM with different kernels
for kernel in kernels:
    svm_clf = SVC(kernel=kernel, random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}")
    print(f"Accuracy without PCA: {accuracy}")
    print("\n" + "="*80 + "\n")