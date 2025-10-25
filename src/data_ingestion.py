import pandas as pd
from logger import logging
from exception import CustomException
import sys


def load_data(csv_path=r"D:\PROJECTS\ML\Recommendation-System\data\myntra.csv"):
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["brand", "colour", "description", "p_attributes", "name", "img"],
        )
        logging.info(f" Data loaded successfully with shape {df.shape}")
        return df

    except Exception as e:
        logging.info("Exception occured at Data Ingestion stage")
        raise CustomException(e, sys)


"""
if __name__ == "__main__":
    c = load_data()
    print(c.head())
    
    
    
"""
