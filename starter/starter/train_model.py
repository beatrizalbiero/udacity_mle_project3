# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data
from starter.starter.ml import model
import pandas as pd
import pickle

# Add the necessary imports for the starter code.

# load in the data.
data = pd.read_csv('starter/data/census.csv')

# Split data
train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb = process_data(
    train, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, encoder=encoder, lb=lb, label="salary",
    training=False
)

# Train and save a model.
clf = model.train_model(X_train, y_train)
y_test_pred = model.inference(clf, X_test)
precision, recall, fbeta = model.compute_model_metrics(y_test, y_test_pred)
with open('model_metrics.txt', 'w') as f:
    f.write('Precision: {}, fbeta: {}, recall: {} \n'.format(precision, fbeta, recall))

# Save model
pickle.dump(clf, open('ml/clf_model.sav', 'wb'))

# Compute Model Metrics
model.compute_model_metrics_slices(
    y_test, y_test_pred, test, features=[
        'sex', 'race'])
