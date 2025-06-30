import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Wellplate():
    def __init__(self, layout, well_data, well_plate_name=None, start_time=None):
        self.layout = layout  # tuple (rows, cols), e.g. (16, 24)
        self.time_points = well_data['Cycle'].astype(np.float64).values  # using 'Cycle' as time (adjust if needed)
        self.well_data = well_data.drop(['Cycle'], axis=1)
        self.start_time = start_time
        self.well_plate_name = well_plate_name
        self.growth_params = pd.DataFrame()

    def find_tau(self, od_readings, time_points):
        od_readings = np.array(od_readings).astype(np.float64)
        if len(od_readings) < 5:
            return None, None
        for i in range(len(od_readings) - 4):
            if all(od_readings[j] < od_readings[j + 1] for j in range(i, i + 4)):
                return time_points[i], i
        return None, None

    def calculateSaturate(self, od_readings):
        od_readings = np.array(od_readings).astype(np.float64)
        max_index = np.argmax(od_readings)
        if max_index == 0:
            K = np.mean(od_readings[:3])
        elif max_index == len(od_readings) - 1:
            K = np.mean(od_readings[-3:])
        else:
            K = np.mean(od_readings[max_index - 1:max_index + 2])
        return K, max_index

    def calculateInitialGrowthRate(self, od_readings, time_points, start_index, end_index):
        od_readings = np.array(od_readings).astype(np.float64)
        try:
            log_od_readings = np.log(od_readings[start_index:end_index + 1])
            slopes = np.gradient(log_od_readings, time_points[start_index:end_index + 1])

            Q1, Q3 = np.percentile(slopes, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            valid_indices = np.where((slopes > lower_bound) & (slopes < upper_bound))[0]
            filtered_slopes = slopes[valid_indices]

            max_slope_index = np.argmax(filtered_slopes)
            original_max_slope_index = valid_indices[max_slope_index] + start_index

            if original_max_slope_index == start_index:
                r = np.mean(slopes[:start_index + 3])
            elif original_max_slope_index == end_index:
                r = np.mean(slopes[end_index - 2:])
            else:
                r = np.mean(slopes[original_max_slope_index - 1:original_max_slope_index + 2])

            return r, original_max_slope_index
        except Exception:
            return None, None

    def calculateGrowth(self, od_readings, time_points):
        time_point, start_index = self.find_tau(od_readings, time_points)
        K, end_index = self.calculateSaturate(od_readings)
        r, _ = self.calculateInitialGrowthRate(od_readings, time_points, start_index, end_index - 1)
        return r, time_point, K

    def compute_params(self):
        growth_rates = []
        tau_values = []
        saturate_values = []

        for col in self.well_data.columns:
            od = self.well_data[col]
            r, tau, K = self.calculateGrowth(od, self.time_points)
            growth_rates.append(r)
            tau_values.append(tau)
            saturate_values.append(K)

        self.growth_params = pd.DataFrame({
            'Well': self.well_data.columns,
            'GrowthRates': growth_rates,
            'Tau': tau_values,
            'SaturationOD': saturate_values
        })

    def plot_raw_data(self, save_path=None):
        row_num, col_num = self.layout
        fig, axs = plt.subplots(row_num, col_num, figsize=(col_num * 2, row_num * 1.5), sharey=True)
        rows = "".join([chr(ord('A') + i) for i in range(row_num)])
        for i, row in enumerate(rows):
            for j in range(1, col_num + 1):
                well_id = f"{row}{j}"
                ax = axs[i, j - 1] if row_num > 1 else axs[j - 1]
                if well_id in self.well_data.columns:
                    ax.plot(self.time_points, self.well_data[well_id])
                    ax.set_title(well_id, fontsize=6)
                    ax.tick_params(labelsize=6)
                else:
                    ax.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    def output_csv(self, filename):
        self.growth_params.to_csv(filename, sep='\t', index=False)

    def get_growth_params(self):
        return self.growth_params
