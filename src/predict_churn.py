import pandas as pd
from pycaret.classification import predict_model, load_model


def load_data(filepath):
    """
    Loads churn data into a dataframe from a string filepath
    """
    data = pd.read_csv(filepath, index_col='customerID')
    return data


def make_predictions(dataframe):
    """
    Uses pycaret best model to make predictions on data in the df dataframe
    """
    model = load_model('LR')
    prediction = predict_model(model, data=dataframe)
    prediction.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    prediction['Churn_prediction'].replace({1: 'Churn', 0: 'No churn'}, inplace=True)
    return prediction['Churn_prediction']


if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
