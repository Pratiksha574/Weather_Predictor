import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("weather.csv")

# Features and target
X = data[["tempC", "humidity", "windspeedKmph"]]
y = data["weather"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print("Model Accuracy:", acc)

# Save model + encoder
joblib.dump(model, "weather_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model saved successfully!")
