# data_loader.py
import pandas as pd
import logger_setup
from conf.config import RAW_DATA_FILE_01, RAW_DATA_FILE_02

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self):
        try:
            logger_setup.logger.debug("START ...")
            logger_setup.logger.debug(f'Loading data from path: {self.data_path}')
            df_train_users = pd.read_csv(f'{self.data_path}{RAW_DATA_FILE_01}')
            df_test_users = pd.read_csv(f'{self.data_path}{RAW_DATA_FILE_02}')
            logger_setup.logger.info(f'Loaded {RAW_DATA_FILE_01} with shape: {df_train_users.shape}')
            logger_setup.logger.info(f'Loaded {RAW_DATA_FILE_02} with shape: {df_test_users.shape}')
            logger_setup.logger.info(f'Successfully loaded data from path: {self.data_path}')
            logger_setup.logger.debug("... FINISH")
            return df_train_users, df_test_users
        except Exception as e:
            logger_setup.logger.error(f'Failed to load data from {self.data_path}: {str(e)}')
            raise