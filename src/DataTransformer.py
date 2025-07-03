from abc import ABC, abstractmethod
import pandas as pd

class DataTransformer(ABC):
    @abstractmethod
    def load_data(filepath, sep):
        pass

    @abstractmethod
    def transform_data(initial_data, columns_to_drop=None):
        pass

    @abstractmethod
    def get_transformed_data(transformed_data):
        pass


class TecanDataTransformer(DataTransformer):
    @staticmethod
    def load_data(file_path, sep="\t"):
        return pd.read_csv(file_path, sep=sep)

    @staticmethod
    def transform_data(initial_data, columns_to_drop=['Cycle', 'Temp. [°C]', 'Time_individual[s]', 'T° 600']):
        # Convert HH:MM:SS to seconds
        initial_data['Time [s]'] = pd.to_timedelta(initial_data['Time']).dt.total_seconds()
    

   
        data = initial_data.drop(columns=columns_to_drop + ['Time'], errors='ignore')

        
        cols = ['Time [s]'] + [col for col in data.columns if col != 'Time [s]']
        data = data[cols]

        return data

    @staticmethod
    def get_transformed_data(transformed_data):
        return transformed_data
