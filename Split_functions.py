# This file is used as a placeholder for functions varying in their necessity

# Importing necessary libraries for functions below
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import pickle


def data_split(predictor, target, train_size = 0.8):
    """ This is given predictor and target dataframes and splits them by the training size on train and test datasets
    :param predictor: dataframe with columns the model will be trained on
    :param target: dataframe with columns the model will be tested on
    :param train_size: what percentage of data to use for training and testing
    :returns: 4 dataframes 2 (X,y) for training and 2 (X_test, y_test) for testing 
    """
    # Splitting the data on train and test
    X, X_test, y, y_test = train_test_split(predictor,
                                          target,
                                          train_size = train_size,
                                          shuffle = False)
    
    # Set of values where train data time overlap with target data time
    drop_set = list(set(X[X['time:timestamp'].between(X_test['time:timestamp'].min(), X_test['time:timestamp'].max())]['case:concept:name']))

    # Clean train
    X = X[X['case:concept:name'].isin(drop_set) == False]
    y = y[y['case:concept:name'].isin(X['case:concept:name'].unique())]
    
    # Clean test
    X_test = X_test[X_test['case:concept:name'].isin(drop_set) == False]
    y_test = y_test[y_test['case:concept:name'].isin(X_test['case:concept:name'].unique())]
    
    return X, X_test, y, y_test

def data_split_viz(predictor, target, train_size = 0.8):
    """ This is given predictor and target dataframes and splits them by the training size on train and test datasets
    :param predictor: dataframe with columns the model will be trained on
    :param target: dataframe with columns the model will be tested on
    :param train_size: what percentage of data to use for training and testing
    :returns: 4 dataframes 2 (X,y) for training and 2 (X_test, y_test) for testing, and the set of dropped applications for visualization purposes
    """
    X, X_test, y, y_test = train_test_split(predictor,
                                          target,
                                          train_size = train_size,
                                          shuffle = False)
    
    # Set of values where train data time overlap with target data time
    drop_set = list(set(X[X['time:timestamp'].between(X_test['time:timestamp'].min(), X_test['time:timestamp'].max())]['case:concept:name']))

    # Clean train
    X = X[X['case:concept:name'].isin(drop_set) == False]
    y = y[y['case:concept:name'].isin(X['case:concept:name'].unique())]
    
    # Clean test
    X_test = X_test[X_test['case:concept:name'].isin(drop_set) == False]
    y_test = y_test[y_test['case:concept:name'].isin(X_test['case:concept:name'].unique())]
    
    return X, X_test, y, y_test, drop_set

def predict_sequence(X, model_time = None, model_event = None, features_time= None, features_event = None):
    """ Uses predictor data in order to predict the event sequence and the time sequence if the model_time with features_time is provided
    If the time model and time features are provided then we can predict both next sequence and time delta sequence. Otherwise we would predict 
    only the next event sequence

    :param X: data used to predict the next event
    :param model_time: model for the time prediction, given the event
    :param model_event: model for the event prediction, given the predictor
    :param features_time: features by which to predict the time sequence
    :param features_event: features by which to predict the event sequence
    :returns: dictionary with predicted sequences and dictionary with predicted time deltas
    """
    predicted_next_sequence = {}
    predicted_time_delta = {}

    if model_time != None or features_time != None:
        for trace in X[features_event[0]].unique():
            check = X[X[features_time[0]] == trace] # contains all features and rows of particular trace
            predicted_trace = [i for i in check[features_time[1:7]].values[0]] # contains only event values
            predicted_time = [i for i in check[features_time[7:-3]].values[0]] # contains only time values

            for _ in range(len(check)):
                input_event = predicted_trace[:len(features_event)-1]
                hour_month_holiday = [i for i in check[features_time[-3:]].values[_]]
                input_time = predicted_time[:len(features_event)-1]

                temp = np.reshape(np.array(input_event + input_time + hour_month_holiday), (1, len(features_time)-1))
                predicted_time.insert(0, model_time.predict(temp)[0])

                input_event = np.reshape(np.array(predicted_trace[:len(features_event)-1]), (1, len(features_event)-1))
                predicted_trace.insert(0, model_event.predict(input_event)[0])
            
            predicted_time_delta[trace] = predicted_time[:len(check)][::-1]
            predicted_next_sequence[trace] = predicted_trace[:len(check)][::-1]
        return predicted_next_sequence, predicted_time_delta
    
    elif model_time == None:
        for trace in X[features_event[0]].unique():
            check = X[X[features_event[0]] == trace] # contains all features and rows of particular trace
            predicted_trace = [i for i in check[features_event[1:]].values[0]]
            for _ in range(len(check)):

                input_event = np.reshape(np.array(predicted_trace[:len(features_event)-1]), (1, len(features_event)-1))
                predicted_trace.insert(0, model_event.predict(input_event)[0])
                
            predicted_next_sequence[trace] = predicted_trace[:len(check)][::-1]
    return predicted_next_sequence
    

def evaluate_trace_prediction(y, predicted_next_sequence = None, predicted_time_delta= None ,features_time= None, features_event= None):
    """ Uses test data in order to assess the predicted event sequence and predicted time sequence if the model_time with features_time is provided
    If the time model and time features are provided then we can assess both next sequence and time delta sequence. Otherwise we would assess 
    only the next event sequence
    
    :param y: data to be predicted
    :param predicted_next_sequence: predicted sequence of events
    :param predicted_time_delta: predicted time sequence
    :param features_time: features by which to predict the time sequence
    :param features_event: features by which to predict the event sequence
    :returns: accuracy scores for the predicted event sequence and predicted time sequence
    """
    accuracy_scores = []
    mean_absolute_errors = []
    if predicted_time_delta != None and features_time != None:
        for trace in y[features_event[0]].unique():
            true = y[y[features_event[0]] == trace][features_event[-1]].to_list()
            predicted = predicted_next_sequence[trace]
            correct = 0
            for i in range(len(true)):
                if true[i] == predicted[i]:
                    correct = correct + 1

            MAE = round(mean_absolute_error(np.exp(y[y['case:concept:name'] == trace][features_time[-1]]), np.exp(predicted_time_delta[trace])),3)

            accuracy_scores.append(correct/len(true))
            mean_absolute_errors.append(MAE)

        return round(np.mean(accuracy_scores)*100, 2), round(np.mean(mean_absolute_errors),3)
    
    elif predicted_time_delta == None:
        for trace in y[features_event[0]].unique():
            true = y[y[features_event[0]] == trace][features_event[-1]].to_list()
            predicted = predicted_next_sequence[trace]
            correct = 0
            for i in range(len(true)):
                if true[i] == predicted[i]:
                    correct = correct + 1

            accuracy_scores.append(correct/len(true))

        return round(np.mean(accuracy_scores)*100, 2)
    
def import_models(file_type_name, n):
    """ Imports trained models in a pickle format
    :param file_type_name: type of the model saved in the file
    :param n: particular model
    :returns: ML model
    """
    event_models = {}

    for i in range(1,n+1):
        loaded_model = pickle.load(open(f'{file_type_name}_{i}.pk1', 'rb'))
        event_models[f'{file_type_name}_{i}.pk1'] = loaded_model

    return event_models


def test_event_models_for_trace_pred(event_models, X, y, X_features_event, y_features_event):
    """ Implements both prediction of the event sequence as well as assessing the predicted trace of events
    :param event_models: list of models used to predict the trace of events
    :param X: dataframe of predcitor data
    :param y: dataframe of target data
    :param X_features_event: features used to predict the trace of events
    :param y_features_event: features to be predicted
    :returns: dictionary with scores of the models
    """
    models_scores = {}
    for model_event_name in event_models:
    
        predicted_next_sequence = predict_sequence(X = X, model_event = event_models[model_event_name], features_event = X_features_event)
        accuracy = evaluate_trace_prediction(y = y, predicted_next_sequence = predicted_next_sequence, features_event = y_features_event)
        print(f'Average predicted event sequence accuracy: {accuracy}%')
        models_scores[f'{model_event_name}'] = accuracy
    return models_scores

def test_suffix_with_time_pred(model_event_short, models_time_scores, time_models, event_models, X, y, X_features_event, X_features_time, y_features_event, y_features_time):
    """Implements the prediction of the trace of events as well as prediction of time for the predicted events and assessing both prediction types
    :param model_event_short: event models list
    :param models_time_scores: empty dictionary to store name of models used to predict the trace and time taken, and its scores
    :param time_models: list of models used to predict time taken for event to happen
    :param event_models: list of models used to predict the trace of events
    :param X: predictor data
    :param y: target data
    :param X_features_event: features used for training the event prediction model
    :param X_features_time: features used for training the time prediction model
    :param y_features_event: features used for testing the event prediction model
    :param y_features_time: features used for testing the time prediction model
    :returns: dictionary containing name: model_event/model_time with the score it had produced
    """
    for model_event_name in model_event_short:
        print(f"Model for event prediction: {model_event_name}")
        for model_time_name in time_models:
            print(f'Model for time predction: {model_time_name}')
            predicted_next_sequence, predicted_time_delta = predict_sequence(X, 
                                                                            time_models[model_time_name], 
                                                                            event_models[model_event_name],
                                                                            X_features_time, 
                                                                            X_features_event)
            accuracy, mae = evaluate_trace_prediction(y, 
                                                    predicted_next_sequence, 
                                                    predicted_time_delta, 
                                                    y_features_time, 
                                                    y_features_event)
            print(f'Average predicted event sequence accuracy: {accuracy}%')
            print(f'Root mean squared error for time prediction: {mae} in seconds, {round(mae/3600,2)} in hours')
            models_time_scores[f'{model_event_name}/{model_time_name}'] = mae
    return models_time_scores


def save_models(model_params, model_type, X, y, X_features, y_features, X_test, y_test):
    """Saving models with varying parameters
    :param model_params: best model parameters
    :param model_type: model used to predict
    :param X: training data used for prediction
    :param y: training data used to test the prediction
    :param X_test: test data used for checking model on unseen data
    :param y_test: test data used for testing the prediction
    :param X_features: dataframe training columns
    :param y_features: dataframe testing columns
    """
    count = 1
    for i in model_params:
        model = model_type(**i)
        model.fit(X[X_features[:-1]].values, np.ravel(y[y_features[0]].values))
        print(f"Model {count} with score ", model.score(X_test[X_features[:-1]].values, y_test[y_features[0]].values), "have been successfully saved!")
        
        pickle.dump(model , open(f'next_activity_prediction_rfc_{count}.pk1' , 'wb'))
        count+=1