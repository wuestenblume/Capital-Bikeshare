from data_analysis_code import *

import pandas as pd
import unittest
from datetime import date
from os.path import exists
import os

class TestStringMethods(unittest.TestCase):
    def test_convert_values_returns_expected_value(self):
        data = pd.DataFrame({"temp": [0.24], "actual_temp" : [0]})          
        convert_temperature(data)                
        assert round(data['actual_temp'][0], 2) == 3.28  

    def test_convert_values_returns_expected_value(self):
        data = pd.DataFrame({"windspeed": [0.5], "actual_windspeed" : [0]})          
        convert_windspeed(data)                
        assert round(data['actual_windspeed'][0], 2) == 33.5

    def test_drop_features(self):
        data = pd.DataFrame({"temp": [0.24, 1], "actual_temp" : [0, 1]})
        drop_features(data, ["temp"])
        assert len(data.columns) == 1
    
    def test_one_hot_encoding (self):
        data = pd.DataFrame({"season": [1,2,3,4]})
        data = one_hot_encoding(data,["season"])
        assert len(data.columns) == 4

    def test_train_model (self):
        train_features = [[0,1,2], [3,4,5]]
        train_labels = [0,1]
        train_model(train_features, train_labels, "test.sav")
        assert exists("test.sav") == True
        os.remove("test.sav")    

    def test_test_model(saved_model):        
        script_dir = os.path.dirname(__file__)    
        path_to_dummy_file = os.path.join(script_dir, 'test-model.sav')      
        test_features = [[1,3,5], [0,2,1]]
        predictions = test_model(path_to_dummy_file, test_features)
        assert predictions[0] == 0.632
        assert predictions[1] == 0.296


if __name__ == '__main__':
    unittest.main()