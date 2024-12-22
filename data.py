import pandas as pd
import random

# List of features
ages = [random.randint(30, 80) for _ in range(150)]
sex = [random.choice([0, 1]) for _ in range(150)]  # 0 = female, 1 = male
chest_pain_types = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
resting_blood_pressure = [random.randint(90, 180) for _ in range(150)]
cholesterol = [random.randint(150, 400) for _ in range(150)]
fasting_blood_sugar = [random.choice([0, 1]) for _ in range(150)]
resting_ecg = ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy']
max_heart_rate_achieved = [random.randint(100, 200) for _ in range(150)]
exercise_induced_angina = [random.choice([0, 1]) for _ in range(150)]
st_depression = [random.uniform(0, 6) for _ in range(150)]
slope = [random.choice([1, 2, 3]) for _ in range(150)]
number_of_major_vessels = [random.choice([0, 1, 2, 3]) for _ in range(150)]
thalassemia = ['Normal', 'Fixed defect', 'Reversable defect']
target = [random.choice([0, 1]) if random.random() > 0.15 else 1 for _ in range(150)]  # 15% chance of heart disease

# Creating the dataframe
data = {
    'Age': ages,
    'Sex': sex,
    'ChestPainType': [random.choice(chest_pain_types) for _ in range(150)],
    'RestingBloodPressure': resting_blood_pressure,
    'Cholesterol': cholesterol,
    'FastingBloodSugar': fasting_blood_sugar,
    'RestingECG': [random.choice(resting_ecg) for _ in range(150)],
    'MaxHeartRateAchieved': max_heart_rate_achieved,
    'ExerciseInducedAngina': exercise_induced_angina,
    'STDepression': st_depression,
    'Slope': slope,
    'NumberOfMajorVessels': number_of_major_vessels,
    'Thalassemia': [random.choice(thalassemia) for _ in range(150)],
    'Target': target
}

df = pd.DataFrame(data)

# Save the data to a CSV file
df.to_csv('synthetic_heart_disease_data.csv', index=False)

# Displaying the first 30 rows as a preview
df.head(30)
