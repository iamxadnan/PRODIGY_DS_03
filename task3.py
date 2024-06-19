import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = 'bank.csv'  # Assuming 'bank.csv' is in the same directory
bank_data = pd.read_csv(file_path, sep=';')

# Display the first few rows of the dataset to understand its structure
print(bank_data.head())

# Convert categorical variables to numerical format
bank_data['default'] = bank_data['default'].map({'no': 0, 'yes': 1})
bank_data['housing'] = bank_data['housing'].map({'no': 0, 'yes': 1})
bank_data['loan'] = bank_data['loan'].map({'no': 0, 'yes': 1})
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})

# Convert categorical variables with more categories into dummy variables
bank_data = pd.get_dummies(bank_data, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'])

# Separate features (X) and target variable (y)
X = bank_data.drop(columns=['y'])  # Features
y = bank_data['y']  # Target variable

# Split data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Classifier Accuracy: {accuracy:.2f}")

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualizing Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=bank_data)
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Visualizing Feature Importances
feature_importances = clf.feature_importances_
sorted_idx = feature_importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Decision Tree Classifier - Feature Importances')
plt.show()
