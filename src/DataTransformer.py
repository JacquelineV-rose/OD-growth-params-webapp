import pandas as pd

class DataTransformer:
    pass  # Base class placeholder

class TecanDataTransformer(DataTransformer):
    @staticmethod
    def load_data(file_path, sep="\t"):
        return pd.read_csv(file_path, sep=sep)

    @staticmethod
    def transform_data(initial_data, columns_to_drop=['Cycle', 'Temp. [°C]', 'Time_individual[s]']):
        transposed_data = initial_data.set_index('Cycle Nr.').T
        transposed_data = transposed_data.reset_index().rename(columns={'index': 'Cycle'})
        return transposed_data.loc[:, ~transposed_data.columns.isin(columns_to_drop)]

    @staticmethod
    def get_transformed_data(transformed_data):
        return transformed_data
