import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler  # RobustScaler for potential outliers
from sklearn.impute import SimpleImputer  # for missing value imputation

# Load the dataset
data = np.genfromtxt("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", delimiter=",")

# Check for infinite and missing values
inf_rows = np.isinf(data).any(axis=1)  # Find rows with any infinity
missing_rows = np.isnan(data).any(axis=1)  # Find rows with any missing values

# Handle infinite values
if inf_rows.any():
  # Check for all rows with infinity
  if inf_rows.all():
    print("Error: All data points contain infinite values. Data cleaning is required.")
    exit()  # Exit if all data is unusable
  else:
    # Clip remaining infinite values (adjust clip values as needed)
    data = np.clip(data, a_min=-1e10, a_max=1e10)

# Handle missing values (consider alternative strategies if needed)
if missing_rows.any():
  # Impute missing values (adjust strategy if necessary)
  imputer = SimpleImputer(strategy='mean')  # Adjust strategy if necessary
  data = imputer.fit_transform(data)
else:
  print("No missing values found in the data.")

# Check for empty data
if data.shape[0] == 0:
  print("Error: No data remaining after handling missing values and infinite values. Please check your data source.")
  exit()  # Exit the program if no data is available

# Split features and labels
features = data[:, :-1]
labels = data[:, -1]

# Preprocess the data
# Use RobustScaler for potential outliers
scaler = RobustScaler()

# Check if data has any rows before scaling (to avoid the error)
if data.shape[0] > 0:
  features_scaled = scaler.fit_transform(features)
else:
  print("Error: No data to scale. Skipping model training.")
  exit()  # Exit the program if no data is available

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)


# Create the model (example using a CNN)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(features_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=12, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)