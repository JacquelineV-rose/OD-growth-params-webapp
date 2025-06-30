import pandas as pd
import matplotlib.pyplot as plt

class Wellplate:
    def __init__(self, layout, well_data, well_plate_name=None, start_time=None):
        self.layout = layout
        self.well_data = well_data
        self.well_plate_name = well_plate_name
        self.start_time = start_time
        self.growth_params = pd.DataFrame()

    def compute_params(self):
        # Dummy example: just count wells as growth rates
        self.growth_params = pd.DataFrame({
            "Well": self.well_data.columns.drop("Time [s]", errors="ignore"),
            "GrowthRates": [0.1] * (len(self.well_data.columns) - 1),
            "Tau": [0] * (len(self.well_data.columns) - 1),
            "SaturationOD": [0.5] * (len(self.well_data.columns) - 1),
        })

    def plot_raw_data(self, save_path=None):
        # Example plot of first well
        if "Time [s]" in self.well_data.columns:
            time = self.well_data["Time [s]"]
            well_cols = self.well_data.columns.drop("Time [s]")
            plt.figure(figsize=(6,4))
            for well in well_cols:
                plt.plot(time, self.well_data[well], label=well)
            plt.legend(fontsize=6)
            plt.xlabel("Time [s]")
            plt.ylabel("OD")
            if save_path:
                plt.savefig(save_path, dpi=150)
            plt.close()

    def output_csv(self, filename):
        self.growth_params.to_csv(filename, sep='\t', index=False)

    def get_growth_params(self):
        return self.growth_params
