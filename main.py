import pandas as pd

# Load the machine sensor data
data = pd.read_csv("Machine_Failure_Prediction/data/ai4i2020.csv")

# Display the first 5 rows
print(data.head())

# Display the number of rows and columns
print(data.shape)# Display all column names
print(data.columns)

# Display dataset information
print(data.info())

# Check missing values
print(data.isnull().sum())

# Check missing values
print(data.isnull().sum())


print(data["Machine failure"].value_counts())

print(data.dtypes)

# Remove unnecessary columns
data = data.drop(["UDI", "Product ID"], axis=1)

# Display remaining columns
print(data.columns)

# Convert categorical data into numerical data
data = pd.get_dummies(data, columns=["Type"], dtype=int)

# Display the first 5 rows
print(data.head())

# Input features
X = data.drop("Machine failure", axis=1)

# Target variable
y = data["Machine failure"]

# Display shapes
print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display the shapes
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Predict using testing data
y_pred = model.predict(X_test)

# Display first 10 predictions
print("Predicted values:", y_pred[:10])

# Display actual values
print("Actual values:", y_test[:10].values)

from sklearn.metrics import accuracy_score

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# Remove failure-related columns from input features
X = data.drop(
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)

# Target variable
y = data["Machine failure"]

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42,
    stratify=y
)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", model.score(X_test, y_test) * 100, "%")

from sklearn.metrics import confusion_matrix, classification_report

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Detailed performance report
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Create a new model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight={0: 1, 1: 5}
)

# Train again
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight={0: 1, 1: 10}
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("WEIGHT 10 CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))


import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)


import joblib

joblib.dump(model, "machine_failure_model.pkl")
print("Model saved successfully!")


import joblib
import pandas as pd

saved_model = joblib.load("machine_failure_model.pkl")

new_machine = pd.DataFrame([{
    "Air temperature [K]": 300,
    "Process temperature [K]": 310,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40,
    "Tool wear [min]": 100,
    "Type_H": 0,
    "Type_L": 1,
    "Type_M": 0
}])

prediction = saved_model.predict(new_machine)

if prediction[0] == 1:
    print("Prediction: Machine failure")
else:
    print("Prediction: Normal")


def predict_machine_failure(air_temp, process_temp, speed, torque, tool_wear, machine_type):
    new_machine = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": 1 if machine_type == "H" else 0,
        "Type_L": 1 if machine_type == "L" else 0,
        "Type_M": 1 if machine_type == "M" else 0
    }])

    prediction = saved_model.predict(new_machine)

    if prediction[0] == 1:
        return "Machine failure"
    else:
        return "Normal"


result = predict_machine_failure(300, 310, 1500, 40, 100, "L")
print("Prediction:", result)

print("Saved model expects features:", saved_model.n_features_in_)
print("Input features:", list(new_machine.columns))