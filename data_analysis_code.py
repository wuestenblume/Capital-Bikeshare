
__author__ = 'Zeynep Vatandas'
 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
import sklearn
import pickle

# Converting date 
def convert_date (data):
    data['dteday'] = pd.to_datetime(data['dteday'],infer_datetime_format=True) 

def data_check(data):
    data.isna().sum()
    data.duplicated().sum()

def set_column_names(data):
    data.rename(columns={'instant':'id',
                        'dteday':'date',
                        'weathersit':'weather',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_rental',
                        'atemp':'feeling_temp',
                        'hr':'hour',
                        'yr':'year',
                        'actual_windspeed':'windspeed',
                        'actual_temp':'temperature'},inplace=True)


def convert_temperature(data):
    data['actual_temp'] = (data['temp'] * 47) - 8 
        
def convert_windspeed(data):
    data['actual_windspeed'] = (data['windspeed'] * 67) 
    
def drop_features(data, features):
    data.drop(features, inplace=True, axis=1)
    return data
    
def one_hot_encoding(data, categorical_features):
    data = pd.get_dummies(data, columns=categorical_features)
    return data

def split_data(data):
    labels = np.array(data['total_rental'])
    data= data.drop('total_rental', axis = 1)
    fdata = np.array(data)
    train_features, test_features, train_labels, test_labels = train_test_split(fdata, labels, test_size = 0.25, random_state = 0)
    return train_features, test_features, train_labels, test_labels

def check_nan_data(x):
    sklearn.utils.assert_all_finite(x)

def check_data_shape(x, y):
    sklearn.utils.check_X_y(x, y,
                            accept_sparse=False,
                            dtype='numeric',
                            order=None,
                            copy=False,
                            force_all_finite=True,
                            ensure_2d=True,
                            allow_nd=False,
                            multi_output=False,
                            ensure_min_samples=1,
                            ensure_min_features=1,
                            y_numeric=False,
                            estimator=None)



def train_model(train_features, train_labels, filename):
    regressor = RandomForestRegressor(n_estimators = 250, random_state = 0)
    regressor.fit(train_features, train_labels)
    pickle.dump(regressor, open(filename, 'wb'))

def test_model(saved_model, test_features):
    loaded_model = pickle.load(open(saved_model, 'rb'))
    predictions = loaded_model.predict(test_features)
    return predictions

def model_evaluation(test_labels, predictions):
    rmsle = mean_squared_log_error (test_labels, predictions, squared=False)
    mae = mean_absolute_error (test_labels, predictions)
    return rmsle, mae

def main():
    data = pd.read_csv('./Bike-Sharing-Dataset/hour.csv')
    saved_model = 'forest_of_trees.sav'
    convert_date (data)
    data_check (data)
    convert_temperature (data)
    convert_windspeed (data)
    set_column_names(data)
    data = drop_features(data, ['temperature','windspeed','id','feeling_temp','date','registered','casual','month'])
    data = one_hot_encoding(data, ["season", "weather"])
    train_features, test_features, train_labels, test_labels = split_data(data)
    print(train_features)
    print(train_labels)
    check_nan_data(train_features)
    check_nan_data(train_labels)
    check_nan_data(test_features)
    check_nan_data(test_labels)
    check_data_shape(train_features, train_labels)
    check_data_shape(test_features, test_labels)
    train_model(train_features, train_labels, saved_model)
    test = test_model(saved_model, test_features)
    error_loss = model_evaluation(test_labels, test)
    print('Here are top 100 prediction by our model')
    print(test[:100])
    print('Here is the error loss (RMSLE and MAE):', error_loss)


if __name__ == "__main__":
    main()

