import numpy as np
import sklearn.model_selection as model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the IDS2017 dataset (assuming CSV format)
with open('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'r') as f:
    data = [line.strip().split(',') for line in f]

# Handle potential header row (optional)
if data[0][0] == 'Destination Port':  # Check if the first element is the header
    data = data[1:]  # Skip the header row if it exists

# Extract features and labels
features = np.array([row[:-1] for row in data], dtype=float)

try:
    # Attempt conversion to int for numerical labels
    labels = np.array([int(row[-1]) for row in data])
except ValueError:
    # Handle non-numerical labels (if necessary)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(np.array([row[-1] for row in data]))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

# Check for infinite values and filter rows accordingly
inf_mask_train = np.isinf(X_train).any(axis=1)
inf_mask_test = np.isinf(X_test).any(axis=1)

X_train = X_train[~inf_mask_train]
y_train = y_train[~inf_mask_train]
X_test = X_test[~inf_mask_test]
y_test = y_test[~inf_mask_test]

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Get predicted probabilities
y_proba = clf.predict_proba(X_test)[:, 1]  # Assuming "DoS" is the second class

# Set a threshold for classification
threshold = 0.7
y_pred_proba_threshold = (y_proba > threshold).astype(int)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_proba_threshold)

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc)
