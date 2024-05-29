import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('retractions35215 (1) 8.51.43 PM.csv').sample(n=2000, random_state=42)

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

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA without standardization
pca = PCA(n_components=2)  # Retain 95% of the variance
X_pca_scaled = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_scaled, y, test_size=0.2, random_state=42)

# Define the SVM kernels to evaluate
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Evaluate SVM with different kernels
for kernel in kernels:
    svm_clf = SVC(kernel=kernel, random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}")
    print(f"Accuracy with PCA (with standardization): {accuracy}")
    print("\n" + "="*80 + "\n")