from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def prepare_data_for_sliced_metrics(
        data, y_target, y_preds, features=cat_features):
    """
    Prepare dataframe to be evaluated (precision, recall and fbeta).

    data : pd DataFrame
    y_target : np.array
        Know labels, binarized
    y_preds : np.array
        Predicted labels, binarized
    features : list - optional
        Features to slice

    Returns
    -------
    data : pd DataFrame
        Preprocessed data to be evaluated
    """
    data_to_slice = pd.merge(data[features],
                             pd.DataFrame(
                                 {'label_value': y_target, 'score': y_preds}),
                             left_index=True, right_index=True)
    return data_to_slice


def compute_model_metrics_slices(y, preds, data, features=cat_features):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    data : pd DataFrame
        The data that will be sliced
    features: list (optional), default = cat_features
        Feature used to compute sliced metrics, if None
        compute to all categorical features.

    Returns
    -------
    Writes metrics to file slice_output.txt
    """

    data_prep = prepare_data_for_sliced_metrics(
        data[features], y, preds, features)

    with open('slice_output.txt', 'w') as f:
        for feature in features:
            for cat in range(len(data_prep[feature].unique())):
                y_filtered = data_prep[data_prep[feature] == data_prep[feature].unique()[
                    cat]]['label_value']
                preds_filtered = data_prep[data_prep[feature] == data_prep[feature].unique()[
                    cat]]['score']

                fbeta = fbeta_score(
                    y_filtered, preds_filtered, beta=1, zero_division=1)
                precision = precision_score(
                    y_filtered, preds_filtered, zero_division=1)
                recall = recall_score(
                    y_filtered, preds_filtered, zero_division=1)

                f.write(
                    'Feature {}, precision: {}, fbeta: {}, recall: {} \n'.format(
                        feature, fbeta, precision, recall))


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def model_load():
    """Loads saved model.
    """
    return pickle.load(open('starter/ml/clf_model.sav', 'rb'))
