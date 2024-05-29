import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('retractions35215 (1) 8.51.43 PM.csv').sample(n=2000, random_state=42)

# Convert RetractionDate to datetime with a specified format
data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], format='%m/%d/%Y', errors='coerce')

# Extract year from RetractionDate
data['RetractionYear'] = data['RetractionDate'].dt.year

# Drop rows with missing RetractionYear
data = data.dropna(subset=['RetractionYear'])

# Select a reduced set of features
features = ['Title', 'Subject', 'Journal', 'Publisher', 'ArticleType', 'Reason', 'CitationCount']
X = data[features]

# Target variable
y = data['RetractionYear']

# Handle missing values
X = X.fillna('Unknown')

# Encode categorical variables
X = pd.get_dummies(X)

# Apply PCA without standardization
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score and classification report
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)