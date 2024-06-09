import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Load CSV files
blinking_data = pd.read_pickle("BLINK.pkl")
right_wink_data = pd.read_pickle("CLENCH.pkl")
test_data = pd.read_pickle("test.pkl")

# Add 'label' column to test_data with default value 'unknown'
test_data['label'] = 'unknown'

# Step 1: Feature Extraction (You need to implement this based on your EEG data)
# For demonstration, let's assume we have extracted features and stored them in 'features' DataFrame

# Step 2: Labeling
blinking_data['label'] = blinking_data.index.map(lambda _: 'blinking')
right_wink_data['label'] = right_wink_data.index.map(lambda _: 'right_wink')

# Step 3: Combine data and split into features and labels
combined_data = pd.concat([blinking_data, right_wink_data], axis=0, ignore_index=True)
X = combined_data.drop(columns=['label'])  # Features
y = combined_data['label']  # Labels

# Step 4: Split data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, _, y_test, _ = train_test_split(test_data, test_data['label'], test_size=0.5, random_state=42)

# Step 5: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Save the trained model as a .p file
joblib.dump(model, 'trained_model.p')

# Step 7: Model Evaluation
print("Feature names seen at fit time:", model.feature_names_in_)
print("Validation Set:\n", classification_report(y_val, model.predict(X_val)))
print("Test Set:\n", classification_report(y_test, model.predict(X_test)))