# any fuctionality to be used in a common way throughout all application

import os
import sys

import numpy as np
import pandas as pd
import dill  # help import pickle file

from src.exception import CustomException

def save_object(file_path, obj):  # takes file path and obj
    try:
        dir_path = os.path.dirname(file_path)  # make a directory
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path, "wb") as file_obj: # open file in write by mode
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)