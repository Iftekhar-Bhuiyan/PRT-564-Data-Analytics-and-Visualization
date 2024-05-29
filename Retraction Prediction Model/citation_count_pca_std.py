import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Loading the dataset
data = pd.read_csv('retractions35215.csv').sample(n=2000, random_state=42)

# Converting OriginalDate to datetime with a specified format
data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], format='%m/%d/%Y', errors='coerce')

# Extracting year from OriginalDate
data['OriginalPaperDate'] = data['OriginalPaperDate'].dt.year

# Dropping rows with missing CitationCount
data = data.dropna(subset=['CitationCount'])

# Selecting features
features = ['Title', 'Subject', 'Journal', 'Publisher', 'ArticleType', 'RetractionNature', 'OriginalPaperDate','Reason', 'Paywalled']

X = data[features]

# Target variable
y = data['CitationCount']

# Handling missing values
X = X.fillna('Unknown')

# Encoding categorical variables
X = pd.get_dummies(X)

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Applying PCA with standardization
pca = PCA(n_components=.90)
X_pca_scaled = pca.fit_transform(X_scaled)

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_scaled, y, test_size=0.2, random_state=42)

# Training a Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# Predicting on the test set with Random Forest
y_pred_rf = clf_rf.predict(X_test)

# Calculating the accuracy score
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy score of Random Forest Classifier: ", accuracy_rf)

# Training a Logistic Regression model
clf_lr = LogisticRegression(max_iter=100, random_state=42)
clf_lr.fit(X_train, y_train)

# Predicting on the test set with Logistic Regression
y_pred_lr = clf_rf.predict(X_test)

# Calculating the accuracy score
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Accuracy score of Logistic Regression: ", accuracy_lr)

# Defining the SVM kernels to evaluate
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Evaluating SVM with different kernels
for kernel in kernels:
    svm_clf = SVC(kernel=kernel, random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}")
    print(f"Accuracy with PCA and Standardisation: {accuracy}")
    print("\n" + "="*80 + "\n")