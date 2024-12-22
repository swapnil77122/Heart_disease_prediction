# Heart_disease_prediction
  

This **Streamlit-based web application** predicts the likelihood of heart disease using a **Random Forest Classifier**. The app allows users to input patient data, view predictions, and visualize key trends in the dataset.  

---

## Features  

1. **Interactive Patient Data Input:**  
   - Users can input patient information through sliders (for numerical features) and dropdowns (for categorical features).  
2. **Dataset Overview:**  
   - Displays statistical insights such as mean, min, and max values of the dataset.  
3. **Heart Disease Prediction:**  
   - Predicts whether a patient has heart disease based on input data.  
4. **Model Performance Metrics:**  
   - Shows model accuracy and confusion matrix to evaluate performance.  
5. **Visualizations:**  
   - Scatter plots comparing user-inputted data with trends in the dataset for features like cholesterol and blood pressure.  
6. **Prediction Probabilities:**  
   - Displays probabilities of both positive and negative predictions.  

---

## Installation  

### Prerequisites  
Ensure you have the following installed:  
- Python >= 3.8  
Usage
Clone this repository or download the files.
Ensure the dataset synthetic_heart_disease_data.csv is in the same directory as the script.
Run the application using the command:
bash
Copy code
streamlit run app.py
The app will open in your web browser, where you can interact with the interface.
File Structure
app.py: The main application script.
synthetic_heart_disease_data.csv: The dataset used for predictions (replace this with your own dataset if necessary).
How It Works
Data Loading:

The application loads the dataset and drops unnecessary columns (ChestPainType, Thalassemia).
Categorical features are preprocessed using Label Encoding.
Model Training:

A Random Forest Classifier is trained on the dataset using an 80-20 train-test split.
The model predicts whether a patient has heart disease based on the user input.
Interactive Features:

User Data Input: Sidebar options allow users to input patient data.
Visualizations: Scatter plots show comparisons between user data and dataset trends.
Prediction: The app provides a binary prediction (Heart Disease/No Heart Disease) and associated probabilities.
Key Visualizations
Age vs Cholesterol:

A scatter plot comparing cholesterol levels across ages for both heart disease and non-heart disease cases.
Age vs Resting Blood Pressure:

Highlights the relationship between age and resting blood pressure.
Confusion Matrix:

A heatmap showing the performance of the classifier in terms of true positives, true negatives, false positives, and false negatives.
Model Performance
The model achieves an accuracy of ~80-90% depending on the dataset and random state during splitting. The confusion matrix is provided in the app for detailed performance insights.

Prediction Output
Result:

The app displays whether the patient is likely to have heart disease.
A color-coded output:
🟢 No Heart Disease Detected
🔴 Heart Disease Detected
Prediction Probabilities:

Probability of No Heart Disease: e.g., 72.5%
Probability of Heart Disease: e.g., 27.5%
Dataset
The application uses a synthetic dataset (synthetic_heart_disease_data.csv) containing features such as:

Age
Cholesterol
Resting Blood Pressure
Max Heart Rate Achieved
ST_Slope
Exercise Angina
Resting ECG
Target (Heart Disease: 1, No Heart Disease: 0)
You can replace this dataset with a real one as long as the format matches.

Future Improvements
Add more advanced visualizations (e.g., feature importance plots).
Expand the dataset for better generalization.
Use hyperparameter tuning to improve model accuracy.
License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

Author
Swapnil Dhage
Email: swapnildhage7885@gmail.com
LinkedIn: Swapnil Dhage
- `pip` (Python package manager)  

### Required Libraries  
Install the necessary libraries by running:  
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
