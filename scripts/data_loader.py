import pandas as pd


class FetchData():
    def __init__(self):
        self.dfdict = {}
        self.dfdict['train'] = self.get_train_data()
        self.dfdict['test'] = self.get_test_data()
        self.dfdict['sample'] = self.get_sample_data()
        self.dfdict['store'] = self.get_store_data()

    def get_train_data(self, name='train'):
        filename = f'../data/data/{name}.csv'
        return pd.read_csv(filename)

    def get_test_data(self, name='test'):
        filename = f'../data/data/{name}.csv'
        return pd.read_csv(filename)

    def get_store_data(self, name='store'):
        filename = f'../data/data/{name}.csv'
        return pd.read_csv(filename)

    def get_sample_data(self, name='sample_submission'):
        filename = f'../data/data/{name}.csv'
        return pd.read_csv(filename)
