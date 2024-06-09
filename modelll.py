import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import __version__ as sklearn_version
import joblib  # Import joblib for model serialization

# Step 1: Load CSV files
blinking_data = pd.read_csv("BLINK.csv")
right_wink_data = pd.read_csv("CLENCH.csv")

# Step 2: Feature Extraction (You need to implement this based on your EEG data)
# For demonstration, let's assume we have extracted features and stored them in 'features' DataFrame

# Step 3: Labeling
blinking_data['label'] = 'blinking'
right_wink_data['label'] = 'right_wink'

# Step 4: Combine data and split into features and labels
data = pd.concat([blinking_data, right_wink_data], ignore_index=True)
X = data.drop(columns=['label'])  # Features
y = data['label']  # Labels

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Save the trained model as a pickle file
joblib.dump(model, 'trained_model.pkl')

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))