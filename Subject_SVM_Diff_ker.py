import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('retractions35215.csv').sample(n=2000, random_state=42)

# Drop rows with missing Reason
data = data.fillna("NA")

# Convert RetractionDate to datetime with a specified format
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], format='%m/%d/%Y', errors='coerce')

# Extract year from RetractionDate
data['OriginalPaperDate'] = data['OriginalPaperDate'].dt.year

# Select features
features = ['Title', 'Journal', 'Publisher', 'ArticleType', 'OriginalPaperDate', 'Reason', 'Paywalled', 'CitationCount']
# if 'OriginalPaperDate' in data.columns:
#     features.append('OriginalPaperDate')

X = data[features]

# Target variable
y = data['Subject']

# Handle missing values
X = X.fillna('Unknown')

# Encode categorical variables
X = pd.get_dummies(X)

# # Convert 'OriginalPaperDate' to datetime and extract year if it exists
# if 'OriginalPaperDate' in X.columns:
#     X['OriginalPaperDate'] = pd.to_datetime(X['OriginalPaperDate'], errors='coerce').dt.year
#     X['OriginalPaperDate'] = X['OriginalPaperDate'].fillna(-1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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