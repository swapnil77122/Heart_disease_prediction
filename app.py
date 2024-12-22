# Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the Dataset (ensure the correct path)
st.title("Heart Disease Prediction App")
df = pd.read_csv('synthetic_heart_disease_data.csv')  # Update this path if necessary

# Print out the column names to check for any discrepancies
st.write(df.columns)

# Headings
st.sidebar.header("Patient Data Input")
st.subheader("Dataset Overview")
st.write(df.describe())

# Drop 'ChestPainType' and 'Thalassemia' columns
df = df.drop(columns=['ChestPainType', 'Thalassemia'], errors='ignore')  # Ignore errors in case columns are missing

# Preprocess categorical data: Label Encoding for categorical features
label_encoder = LabelEncoder()

# Apply LabelEncoder to categorical columns (remove ChestPainType and Thalassemia here)
categorical_columns = ['RestingECG', 'ExerciseAngina', 'ST_Slope']  # Categorical columns after dropping irrelevant ones
for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# Encode the target variable
if 'Target' in df.columns:
    df['Target'] = label_encoder.fit_transform(df['Target'])

# Splitting Data into Features (X) and Target (y)
X = df.drop(columns=["Target"])  # 'Target' is the outcome variable for heart disease
y = df["Target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function for User Input via Sidebar (handling categorical and numerical columns separately)
def user_input_features():
    user_data = {}
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:  # Check if the column is numeric
            min_val, max_val = int(X[col].min()), int(X[col].max())
            mean_val = float(X[col].mean())
            user_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, int(mean_val))
        else:  # Handle categorical columns
            options = X[col].unique().tolist()
            user_data[col] = st.sidebar.selectbox(f"{col}", options)
    return pd.DataFrame(user_data, index=[0])

# Collect User Data
user_data = user_input_features()
st.subheader("Patient Data")
st.write(user_data)

# Preprocess user data (Label Encoding for categorical columns)
for col in categorical_columns:
    if col in user_data.columns:
        user_data[col] = label_encoder.transform(user_data[col])

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Prediction
prediction = model.predict(user_data)
prediction_proba = model.predict_proba(user_data)

# Accuracy Score
st.subheader("Model Performance")
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

# Visualizations
st.title("Visualized Patient Report")

# Set Color Based on Prediction
color = "red" if prediction[0] == 1 else "blue"

# Visualization Functions
def create_scatter_plot(feature):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="Age", y=feature, data=df, hue="Target", palette="coolwarm")
    sns.scatterplot(x=user_data["Age"], y=user_data[feature], s=200, color=color, label="Your Data")
    plt.title(f"Age vs {feature.capitalize()} (0 = No Heart Disease, 1 = Heart Disease)")
    st.pyplot(plt)

# Features to Plot (Removed 'ST_Slope' from here)
features_to_plot = ["Cholesterol", "RestingBloodPressure", "MaxHeartRateAchieved"]
for feature in features_to_plot:
    st.header(f"Age vs {feature.capitalize()} Comparison")
    create_scatter_plot(feature)

# Confusion Matrix
st.header("Confusion Matrix")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
st.pyplot(plt)

# Display the Prediction
st.subheader("Prediction Result")
if prediction[0] == 0:
    st.write("🟢 **No Heart Disease Detected**")
else:
    st.write("🔴 **Heart Disease Detected**")

st.subheader("Prediction Probability")
st.write(f"Probability of No Heart Disease: {prediction_proba[0][0] * 100:.2f}%")
st.write(f"Probability of Heart Disease: {prediction_proba[0][1] * 100:.2f}%")
