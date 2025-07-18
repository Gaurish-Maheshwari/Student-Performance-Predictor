import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import os
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class data_transformation_config:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl') 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = data_transformation_config()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info("Numerical and categorical feature transformation pipelines created.")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            logging.info("Preprocessor object created.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data loaded successfully for transformation.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying transformations to training and testing data.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.preprocessor_obj_file_path}")
            

            logging.info("Data transformation completed successfully.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
