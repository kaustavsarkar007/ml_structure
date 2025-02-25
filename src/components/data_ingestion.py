# all the code realated to reading the data

import os
import sys


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_transformation import DataTranformation, DataTransformationConfig



import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass # good for only for variable save
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # given input is saved in specific path
    test_data_path: str = os.path.join('artifacts', "test.csv")  # given input is saved in specific path
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # given input is saved in specific path
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # above paths will be saved into ingestion_config
    
    def initiate_data_ingestion(self):
        logging.info("Enterd the data ingestion method or component")
        
        try: 
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info('Read the data set as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train Test split initialted")
            
            train_set , test_set = train_test_split(df, test_size=0.2 , random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header = True)
            
            logging.info("Ingetion of data completed")
            
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTranformation()
    data_transformation.initiate_data_tranformation(train_data, test_data)
    