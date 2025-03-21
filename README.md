# Overstimulation Predictor

This Jupyter Notebook implements a machine learning model to predict overstimulation based on various lifestyle and psychological factors. The model uses K-Nearest Neighbors (KNN) algorithm with feature engineering and data visualization capabilities.

## Features
- **Data Visualization**: Includes correlation heatmap and target variable distribution plots
- **Feature Engineering**: Creates derived features like Sleep Quality Ratio, Tech Stress, and Work-Life Balance
- **KNN Model**: Implements KNN classification with automatic k-value optimization
- **Prediction Function**: Allows users to input their own data for overstimulation prediction

## Dataset
- Uses `overstimulation_dataset.csv` (not included) containing 20+ features including:
  - Age, Sleep Hours, Screen Time
  - Stress Level, Noise Exposure
  - Anxiety Score, Depression Score
  - And more...

## Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Usage
1. Place `overstimulation_dataset.csv` in the same directory
2. Run the notebook
3. Use the `predict_overstimulation()` function with your own input data

## Example Input
```python
example_input = {
    'Age': 35,
    'Sleep_Hours': 7.5,
    'Screen_Time': 6.0,
    # ... (all required features)
}
result = predict_overstimulation(example_input)
