from abc import ABC, abstractmethod
import pandas as pd


class DataTransformer(ABC):
    """
    Abstract base class for data transformation.
    """

    @staticmethod
    @abstractmethod
    def load_data(filepath: str, sep: str = "\t") -> pd.DataFrame:
        """
        Load raw data from file.
        """
        pass

    @staticmethod
    @abstractmethod
    def transform_data(initial_data: pd.DataFrame, columns_to_drop=None) -> pd.DataFrame:
        """
        Transform raw data into analysis-ready format.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transformed_data(transformed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Return the transformed data (can include postprocessing if needed).
        """
        pass


class TecanDataTransformer(DataTransformer):
    """
    Concrete implementation of DataTransformer for Tecan plate reader data.
    """

    @staticmethod
    def load_data(file_path: str, sep: str = "\t") -> pd.DataFrame:
        return pd.read_csv(file_path, sep=sep)

    @staticmethod
    def transform_data(initial_data, columns_to_drop=['Cycle', 'Temp. [°C]', 'Time_individual[s]', 'T° 600']):
        initial_data['Time [s]'] = pd.to_timedelta(initial_data['Time']).dt.total_seconds()

    # Drop unneeded columns
        data = initial_data.drop(columns=columns_to_drop + ['Time'], errors='ignore')

    # Move 'Time [s]' to front
        cols = ['Time [s]'] + [col for col in data.columns if col != 'Time [s]']
        data = data[cols]

    # Fill NaNs in well columns with 0 or another sensible default
        for col in data.columns:
            if col != 'Time [s]':
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.0)

        return data


    @staticmethod
    def get_transformed_data(transformed_data: pd.DataFrame) -> pd.DataFrame:
        return transformed_data
