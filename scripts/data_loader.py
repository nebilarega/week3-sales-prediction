import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('../logs/loading.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class FetchData():
    def __init__(self):
        self.dfdict = {}
        self.dfdict['train'] = self.get_train_data()
        self.dfdict['test'] = self.get_test_data()
        self.dfdict['sample'] = self.get_sample_data()
        self.dfdict['store'] = self.get_store_data()

        logger.info('Loaded train, test, sample and store data')

    def get_train_data(self, name='train'):
        filename = f'../data/data/{name}.csv'
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            logger.error("{} not found. ".format(filename))
        else:
            return df

    def get_test_data(self, name='test'):
        filename = f'../data/data/{name}.csv'
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            logger.error("{} not found. ".format(filename))
        else:
            return df

    def get_store_data(self, name='store'):
        filename = f'../data/data/{name}.csv'
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            logger.error("{} not found. ".format(filename))
        else:
            return df

    def get_sample_data(self, name='sample_submission'):
        filename = f'../data/data/{name}.csv'
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            logger.error("{} not found. ".format(filename))
        else:
            return df
