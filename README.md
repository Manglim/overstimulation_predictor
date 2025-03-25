# Overstimulation Prediction Using KNN

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier to predict whether an individual is overstimulated based on personal and behavioral features. It includes data exploration, feature engineering, model training, evaluation, and a prediction function. The dataset is assumed to be in `overstimulation_dataset.csv`.

## Functionality
1. **Data Exploration and Visualization**:
   - Loads data and visualizes feature correlations (heatmap) and target distribution (count plot).

2. **Feature Engineering**:
   - Creates derived features:
     - `Sleep_Quality_Ratio`: `Sleep_Hours / Sleep_Quality`
     - `Tech_Stress`: `Screen_Time * Stress_Level`
     - `Work_Life_Balance`: `Work_Hours / (Social_Interaction + 1)`
   - Uses 22 features (19 original + 3 derived).

3. **Data Preprocessing**:
   - Splits data into 80% training and 20% testing sets.
   - Scales features with `StandardScaler`.

4. **KNN Model**:
   - Tests k-values (1-30) to find the optimal number of neighbors.
   - Trains a KNN classifier with the best k-value.

5. **Model Evaluation**:
   - Provides classification report and confusion matrix.

6. **Prediction**:
   - Function (`predict_overstimulation`) predicts overstimulation and probability for new inputs.

## Frameworks and Libraries
- **Python**: Core language.
- **Pandas**: Data manipulation (`pd`).
- **NumPy**: Numerical operations (`np`).
- **Matplotlib**: Plotting (`plt`).
- **Seaborn**: Statistical visualization (`sns`).
- **Scikit-learn**: Machine learning:
  - `train_test_split`, `StandardScaler`, `KNeighborsClassifier`, `classification_report`, `confusion_matrix`.

## Dataset
- Expected in `overstimulation_dataset.csv`.
- Features: `Age`, `Sleep_Hours`, `Screen_Time`, `Stress_Level`, `Noise_Exposure`, `Social_Interaction`, `Work_Hours`, `Exercise_Hours`, `Caffeine_Intake`, `Multitasking_Habit`, `Anxiety_Score`, `Depression_Score`, `Sensory_Sensitivity`, `Meditation_Habit`, `Overthinking_Score`, `Irritability_Score`, `Headache_Frequency`, `Sleep_Quality`, `Tech_Usage_Hours`.
- Target: `Overstimulated` (0 or 1).

## Key Features
- **Visualization**: Heatmap and count plot for insights.
- **Feature Engineering**: Adds derived features for better prediction.
- **Tuning**: Optimizes k-value for KNN.
- **Scalability**: Normalizes features with `StandardScaler`.

## Installation
Ensure Python and the following libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Clone the repository:
bash

git clone <repository-url>
cd <repository-directory>

Ensure overstimulation_dataset.csv is in the directory.

Run the script:
bash

python main.py

Use the prediction function:
python

example_input = {
    'Age': 35, 'Sleep_Hours': 2, 'Screen_Time': 6.0, 'Stress_Level': 5,
    'Noise_Exposure': 2, 'Social_Interaction': 6, 'Work_Hours': 15,
    'Exercise_Hours': 1.5, 'Caffeine_Intake': 2, 'Multitasking_Habit': 1,
    'Anxiety_Score': 4, 'Depression_Score': 0, 'Sensory_Sensitivity': 2,
    'Meditation_Habit': 0, 'Overthinking_Score': 5, 'Irritability_Score': 4,
    'Headache_Frequency': 2, 'Sleep_Quality': 3, 'Tech_Usage_Hours': 7.0
}
result = predict_overstimulation(example_input)
print(f"Overstimulated: {result['Overstimulated']}, Probability: {result['Probability']:.2f}")

