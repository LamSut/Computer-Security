import tensorflow as tf
import numpy as np

# Load your dataset (replace with your own data)
X_train, y_train = load_dos_data()

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),  # Add dropout for regularization
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions on new data
X_test = load_new_data()
y_pred = model.predict(X_test)

# Threshold the predictions to classify as DoS or not
threshold = 0.7  # Adjust threshold based on your requirements
y_pred_binary = (y_pred > threshold).astype(int)

# Evaluate the model's performance
accuracy = np.mean(y_pred_binary == y_test)
print("Accuracy:", accuracy)