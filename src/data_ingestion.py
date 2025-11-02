import pandas as pd
from logger import logging
from exception import CustomException
import sys
from sqlalchemy import create_engine
import os

try:

    user = "root"
    password = "root"
    host = "localhost"
    database = "product_db"

    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

    query = """
    SELECT brand, colour, description, p_attributes, name, img
    FROM products;
    """

    df = pd.read_sql(query, engine)
    datapath = os.path.join("data", "raw")
    os.makedirs(datapath, exist_ok=True)
    df.to_csv(os.path.join(datapath, "dataset.csv"), index=False)
    logging.info("CSV exported successfully!")

    logging.info(f"Data:{df.head()}")
except Exception as e:
    logging.info("Exception occured at Data Ingestion stage")
    raise CustomException(e, sys)
