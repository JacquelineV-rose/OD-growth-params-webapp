from abc import ABC, abstractmethod
import pandas as pd

class DataTransformer(ABC):
    @abstractmethod
    def load_data(self, file_path: str, sep: str = ",") -> pd.DataFrame:
        pass

    @abstractmethod
    def transform_data(self, initial_data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_transformed_data(self, transformed_data: pd.DataFrame) -> pd.DataFrame:
        pass

class TecanDataTransformer(DataTransformer):
    def load_data(self, file_path: str, sep: str = ",") -> pd.DataFrame:
        df = pd.read_csv(file_path, sep=sep)
        return df

    def transform_data(self, initial_data: pd.DataFrame) -> pd.DataFrame:
        # Example transformation: rename time column, drop temp columns
        time_cols = [col for col in initial_data.columns if "Time" in col]
        if time_cols:
            initial_data = initial_data.rename(columns={time_cols[0]: "Time [s]"})
        temp_cols = [col for col in initial_data.columns if "T°" in col or "Temp" in col]
        initial_data = initial_data.drop(columns=temp_cols, errors="ignore")
        return initial_data

    def get_transformed_data(self, transformed_data: pd.DataFrame) -> pd.DataFrame:
        return transformed_data

    # New method to transform a dataframe directly (e.g., from uploaded file)
    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        time_cols = [col for col in df.columns if "Time" in col]
        if time_cols:
            df = df.rename(columns={time_cols[0]: "Time [s]"})
        temp_cols = [col for col in df.columns if "T°" in col or "Temp" in col]
        df = df.drop(columns=temp_cols, errors="ignore")
        return df
