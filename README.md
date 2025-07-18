# Wine Quality Prediction

This project uses a linear regression model to predict the quality of red wine based on various physicochemical features.

## Dataset

The dataset used is [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality) (red wine), loaded from a CSV file.

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

Install dependencies with:
```bash
pip install pandas matplotlib seaborn scikit-learn joblib
```

## Usage

1. **Train the Model**

   The notebook loads the dataset, splits it into training and test sets, trains a linear regression model, and saves the model as `wine_quality_model.pkl`.

2. **Predict with the Model**

   The notebook demonstrates loading the saved model and making predictions on test samples.

## Files

- `WineQuality.ipynb`: Main notebook for training and testing the model.
- `winequality-red.csv`: Dataset file (not included, download from UCI repository).
- `wine_quality_model.pkl`: Saved trained model.

## Example

```python
import joblib

# Load the saved model
model = joblib.load('wine_quality_model.pkl')

# Predict on new data
predictions = model.predict(X_test.head())
print("Predictions:", predictions)
```

## License

This project
