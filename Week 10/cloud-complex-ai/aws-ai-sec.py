import boto3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize AWS clients
ec2_client = boto3.client('ec2')
s3_client = boto3.client('s3')

# Define instance type mapping (adjust as needed)
instance_type_mapping = {
    't2.micro': 0,
    't2.small': 1,
    'm4.large': 2,
    # ... other instance types
}

# Define launch time mapping (adjust as needed)
launch_time_mapping = {
    '2023-01-01': 0,
    '2023-02-01': 1,
    '2023-03-01': 2,
    # ... other launch times
}

# Define labels (adjust as needed)
labels = [
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    # ... other labels
]

# Function to fetch and process EC2 instance data
def fetch_ec2_data():
    response = ec2_client.describe_instances()
    instances = response['Reservations'][0]['Instances']

    # Extract relevant data (adjust as needed)
    data = []
    for instance in instances:
        instance_id = instance['InstanceId']
        instance_type = instance['InstanceType']
        launch_time = instance['LaunchTime']
        public_ip = instance['PublicIpAddress']
        private_ip = instance['PrivateIpAddress']
        security_groups = instance['SecurityGroups']
        data.append([instance_id, instance_type, launch_time, public_ip, private_ip, security_groups])

    return data

# Function to preprocess data and create features
def preprocess_data(data):
    # Convert data to NumPy array
    data = np.array(data)

    # Encode categorical features (adjust as needed)
    encoded_data = np.zeros((len(data), len(data[0])))
    for i, row in enumerate(data):
        encoded_data[i, 0] = instance_type_mapping[row[1]]  # Example mapping
        encoded_data[i, 1] = launch_time_mapping[row[2]]  # Example mapping
        # ... other encodings

    return encoded_data

# Function to create and train the AI model
def train_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model

# Function to predict anomalies
def predict_anomalies(model, X_test):
    predictions = model.predict(X_test)
    anomalies = np.where(predictions > 0.5)[0]

    return anomalies

# Main execution
if __name__ == '__main__':
    data = fetch_ec2_data()
    preprocessed_data = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, labels, test_size=0.2)

    model = train_model(X_train, y_train)

    # Predict anomalies on the testing set
    anomalies = predict_anomalies(model, X_test)

    # Print or log the anomalies
    print("Detected anomalies:", anomalies)

    # Generate CSV file
    csv_data = pd.DataFrame(data, columns=['Instance ID', 'Instance Type', 'Launch Time', 'Public IP', 'Private IP', 'Security Groups'])
    csv_data.to_csv('ec2_data.csv', index=False)