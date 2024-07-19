import sqlite3

import datasets
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Union
from datasets import Dataset, DatasetDict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    db_path: str
    table_name: str = "data"
    subset_name: Optional[str] = "train"

class ReturnTypeDatabase(Enum):
    DATAFRAME = "dataframe"
    DATASET = "dataset"

class Database:
    def __init__(self, cfg: DatabaseConfig):
            self.config = cfg
            self.conn = sqlite3.connect(self.config.db_path)
            self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def read_rows(self, column: Optional[str], limit: Optional[int],
                  offset: Optional[int],
                  return_type: Optional[ReturnTypeDatabase] = ReturnTypeDatabase.DATAFRAME) \
            -> Union[pd.DataFrame, DatasetDict]:
        if column is None:
            column = "*"
        logger.info(f"Reading {column} from {self.config.table_name}")
        query = f"SELECT {column} FROM {self.config.table_name} LIMIT {limit} OFFSET {offset}"

        logger.info(f"Executing query: {query}")

        if return_type == ReturnTypeDatabase.DATASET:
            if self.config.subset_name is None:
                raise ValueError("subset_name must be provided to return a dataset")
            dataset: Dataset = Dataset.from_pandas(pd.read_sql_query(query, self.conn))
            dataset_dict: DatasetDict = datasets.DatasetDict(train=dataset)
            return dataset_dict

        return pd.read_sql_query(query, self.conn)

    def read_all(self, column: Optional[str], return_type: Optional[ReturnTypeDatabase]) -> pd.DataFrame:
        return self.read_rows(column, None, None, return_type)

    def write_rows(self, data: Union[pd.DataFrame, Dataset]) -> None:
        logger.info(f"Writing data to {self.config.table_name}")
        data.to_sql(self.config.table_name, self.conn, if_exists="append", index=False)
        self.conn.commit()
        logger.info("Data written")


