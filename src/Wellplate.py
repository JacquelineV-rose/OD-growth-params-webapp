import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Wellplate():
    def __init__(self, layout, well_data, well_plate_name=None, start_time=None):
        self.layout = layout  # (row,col), e.g., (16,24)
        self.time_points = well_data['Time [s]'].astype(np.float64).values
        self.well_data = well_data.drop(['Time [s]'], axis=1)
        self.start_time = start_time
        self.well_plate_name = well_plate_name
        self.growth_params = pd.DataFrame()
        self.compute_params()

    def find_tau(self, od_readings, time_points, window=3, threshold=0.03):
        od_readings = np.array(od_readings).astype(np.float64)
        time_points = np.array(time_points).astype(np.float64)

        if len(od_readings) < 5:
            return None, None

        smoothed = np.convolve(od_readings, np.ones(window) / window, mode='same')
        baseline = np.median(smoothed[:window * 2])

        dynamic_threshold = max(threshold, 0.05 * baseline)

        for i, val in enumerate(smoothed):
            if val > baseline + dynamic_threshold:
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
        if start_index is None or end_index is None or start_index >= end_index:
            return None, None

        od_readings = np.array(od_readings).astype(np.float64)

        try:
            log_od = np.log(od_readings[start_index:end_index + 1])
            slopes = np.gradient(log_od, time_points[start_index:end_index + 1])

            Q1, Q3 = np.percentile(slopes, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            valid_indices = np.where((slopes > lower_bound) & (slopes < upper_bound))[0]
            if len(valid_indices) == 0:
                return None, None

            filtered_slopes = slopes[valid_indices]
            max_slope_index = np.argmax(filtered_slopes)
            original_max_index = valid_indices[max_slope_index] + start_index

            if original_max_index == start_index:
                r = np.mean(slopes[:3])
            elif original_max_index == end_index:
                r = np.mean(slopes[-3:])
            else:
                r = np.mean(slopes[max(0, original_max_index - 1):original_max_index + 2])

            return r, original_max_index
        except Exception:
            return None, None

    def compute_params(self):
        results = []

        for well_name, col in self.well_data.items():
            tau_val, tau_idx = self.find_tau(col, self.time_points)
            K_val, K_idx = self.calculateSaturate(col)

            if tau_idx is None:
                tau_idx = 0
            if K_idx is None or K_idx <= tau_idx + 1:
                K_idx = len(col) - 1

            r_val, r_idx = self.calculateInitialGrowthRate(col, self.time_points, tau_idx, K_idx)
            print(f"{well_name} -> tau_idx: {tau_idx}, K_idx: {K_idx}, growth_rate: {r_val}, growth_rate_idx: {r_idx}")

            results.append({
                'Well': well_name,
                'tau_values': tau_val if tau_val is not None else np.nan,
                'tau_index': tau_idx if tau_idx is not None else np.nan,
                'GrowthRates': r_val if r_val is not None else np.nan,
                'growth_rates_index': r_idx if r_idx is not None else np.nan,
                'saturate_values': K_val if K_val is not None else np.nan,
                'saturate_index': K_idx if K_idx is not None else np.nan,
                'saturation_time': self.time_points[int(K_idx)] if pd.notna(K_idx) else np.nan
            })

        self.growth_params = pd.DataFrame(results)

    def get_growth_params(self):
        return self.growth_params

    def output_csv(self, filename):
        self.growth_params.to_csv(filename, sep='\t')

    def plot_raw_data(self, save_path=None, wells=None):
        """
        Plot raw OD data for specified wells.
        If wells=None, plot all wells.
        If save_path is provided, save plot(s) to file(s).
        If multiple wells, saves one PNG per well if save_path ends with '.png',
        appending well name before .png.
        """
        if wells is None:
            wells = list(self.well_data.columns)

        for well_id in wells:
            if well_id not in self.well_data.columns:
                continue

            plt.figure(figsize=(6, 4))
            plt.plot(self.time_points, self.well_data[well_id], label='Raw OD')

            if not self.growth_params.empty:
                try:
                    tau_index = self.growth_params.loc[self.growth_params['Well'] == well_id, 'tau_index'].values[0]
                    saturate_index = self.growth_params.loc[self.growth_params['Well'] == well_id, 'saturate_index'].values[0]

                    if not pd.isna(tau_index):
                        plt.axvline(self.time_points[int(tau_index)], color='red', linestyle='--', label='Tau')

                    if not pd.isna(saturate_index):
                        plt.axvline(self.time_points[int(saturate_index)], color='green', linestyle='--', label='Saturation')
                except Exception:
                    pass

            plt.title(f'Well {well_id}')
            plt.xlabel('Time (s)')
            plt.ylabel('Optical Density (OD)')
            plt.legend()
            plt.tight_layout()

            if save_path:
                # If multiple wells, create a file per well by inserting well_id before extension
                if len(wells) > 1 and save_path.endswith('.png'):
                    base = save_path[:-4]
                    ext = '.png'
                    file_path = f"{base}_{well_id}{ext}"
                    plt.savefig(file_path)
                else:
                    plt.savefig(save_path)
            plt.close()
