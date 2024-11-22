import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# Create a copy of the dataset to avoid modifying the original one
heart_df = heart.copy()

# Renaming the target column
heart_df = heart_df.rename(columns={'condition': 'target'})

# Features (X) and target (y)
X = heart_df.drop(columns='target')
y = heart_df['target']

# Split the dataset into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the StandardScaler to scale the data
scaler = StandardScaler()

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# Scale the testing data
X_test_scaled = scaler.transform(X_test)

# Create and train the RandomForest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Save the trained model to a file
with open('heart-disease-prediction-rf-model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Optionally, print the accuracy of the trained model on the test set
accuracy = model.score(X_test_scaled, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')

print("Model and Scaler saved successfully!")
