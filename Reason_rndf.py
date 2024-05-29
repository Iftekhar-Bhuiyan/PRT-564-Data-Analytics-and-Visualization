import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('retractions35215.csv').sample(n=2000, random_state=42)

# Drop rows with missing Reason
data = data.dropna(subset=['Reason'])

# Convert RetractionDate to datetime with a specified format
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], format='%m/%d/%Y', errors='coerce')

# Extract year from RetractionDate
data['OriginalPaperDate'] = data['OriginalPaperDate'].dt.year

# Select features
# Replace 'OriginalPaperDate' with the correct column name if different
features = ['Title', 'Subject', 'Journal', 'Publisher', 'ArticleType', 'RetractionNature', 'OriginalPaperDate', 'Paywalled', 'CitationCount']
# if 'OriginalPaperDate' in data.columns:
#     features.append('OriginalPaperDate')

X = data[features]

# Target variable
y = data['Reason']

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

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy score and classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of Random Forest Classifier when Reason of Retraction is the target Variable: ", accuracy)