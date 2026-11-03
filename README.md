# 🏭 Machine Failure Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to predict potential failures in manufacturing machines based on machine sensor data.

The main objective is to identify whether a machine is operating normally or is likely to experience a failure, helping manufacturing industries improve maintenance planning and reduce unexpected machine downtime.

## 🎯 Project Objectives

- Predict machine failures using sensor data.
- Identify patterns in machine operating conditions.
- Support predictive maintenance.
- Reduce unexpected machine downtime.

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset.

The dataset contains manufacturing machine information and sensor measurements.

## ⚙️ Input Features

The model uses 8 input features, including:

- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (rpm)
- Torque (Nm)
- Tool Wear (min)
- Product Type (encoded into numerical features)

## 🧠 Machine Learning Workflow

1. Load the manufacturing dataset.
2. Perform data preprocessing.
3. Prepare input features and target labels.
4. Train the classification model.
5. Evaluate the trained model.
6. Save the trained model.
7. Predict machine operating conditions.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Git and GitHub

## 📁 Project Structure

```text
Machine-Failure-Prediction/
│
├── data/
│   └── ai4i2020.csv
│
├── app.py
├── main.py
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/richardhub094/Machine-Failure-Prediction.git
```

### 2. Open the project folder

```bash
cd Machine-Failure-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Train the machine learning model

```bash
python train_model.py
```

### 5. Run the prediction program

```bash
python predict.py
```

## 🔍 Expected Output

The model predicts machine operating conditions, such as:

- Normal
- Failure

## 💡 Real-World Application

This project demonstrates how Machine Learning can support predictive maintenance in manufacturing industries by identifying potential machine failures from operating data.

## 👨‍💻 Author

**Richard Mathew**

B.Tech – Artificial Intelligence and Data Science

GitHub: https://github.com/richardhub094

---

⭐ If you find this project interesting, feel free to explore the repository.