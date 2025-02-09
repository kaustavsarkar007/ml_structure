# all codes related to data transformation code goes here

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
    
class DataTranformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        
        '''
        Responsible for data transformation 
        
        '''
        try:
            numerical_features =['reading_score', 'writing_score']
            categorical_features = ['gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course']
            
            num_pipeline = Pipeline(                              # numerical pipeline
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # impute numerical data using median strategy
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(                              # categorical pipeline
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")), 
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Categorical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),  # name , transformer(as mentione in pipeline) ,features needs to transform 
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )
            
            logging.info("preprocessing done")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
def initiate_data_tranformation(self,train_path,test_path):
    
    '''
    
    Starting datatransformation here
    
    '''
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        logging.info("Train and test data read completed")
        logging.info("Obtainng preprocessor object")
        
        preprocessing_obj = self.get_data_transformer_obj()  # getting whatever preprocessing done above
        
        target_colum_name = "math_score"
        numerical_features =['reading_score', 'writing_score']
        
        input_feature_train_df  = train_df.drop(columns=[target_colum_name],axis=1)  # input features for training apart from target column
        target_feature_train_df = train_df[target_colum_name] # get target feature
        
        input_feature_test_df  = test_df.drop(columns=[target_colum_name],axis=1)  # input features for test apart from target column
        target_feature_test_df = test_df[target_colum_name] # get target feature
        
        
        logging.info(
            f"applying preprocessing on training and test datframe"
        )
        
        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
        
        
        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] 
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] 
        
        logging.info(f"Saving preprocessing object")
        
        save_object(                                    # to save as pkl file
            file_path = self.Data_transformation_config.preprocessing_obj_file_path,
            obj= preprocessing_obj
        )
        
        return ( train_arr, 
                test_arr, 
                self.Data_transformation_config.preprocessing_obj_file_path,)
    except:
        pass