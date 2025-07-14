import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
def save_object(obj, file_path):
    """
    Save an object to a file using pickle.
    
    Parameters:
    obj: The object to save.
    file_path: The path where the object will be saved.
    """
    import pickle
    try:
         dir_path = os.path.dirname(file_path)
         os.makedirs(dir_path, exist_ok=True)
         with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    
    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_object(file_path):
    """
    Load an object from a file using pickle.
    
    Parameters:
    file_path: The path from which the object will be loaded.
    
    Returns:
    The loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e

