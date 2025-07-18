import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Wellplate:
    def __init__(self, layout, well_data, well_plate_name=None, start_time=None):
        """
        Wellplate represents a plate reader experiment.
        """
        self.layout = layout  # tuple of (rows, cols), e.g., (16,24) for 384-well
        self.time_points = well_data['Time [s]'].astype(np.float64).values
        self.well_data = well_data.drop(['Time [s]'], axis=1)
        self.start_time = start_time

        self.well_plate_name = well_plate_name
        self.growth_params = pd.DataFrame()
        self.compute_params()

    @staticmethod
    def detect_plate_layout(columns):
        """
        Detect plate layout (rows, columns) from column names.
        Assumes columns use standard well ID format (e.g., A1, B12, etc.).
        """
        well_cols = [col for col in columns if col not in ('Time', 'Time [s]', 'T° 600')]
        rows = sorted(set(col[0] for col in well_cols))
        cols = sorted(set(int(col[1:]) for col in well_cols))
        return len(rows), len(cols)

    def find_tau(self, od_readings, time_points, window=3, threshold=0.03):
        od_readings = np.array(od_readings, dtype=np.float64)
        time_points = np.array(time_points, dtype=np.float64)

    
        if len(od_readings) < window * 2:
            print("Too few data points to compute baseline")
            return None, None

    
        if np.all(od_readings == 0) or np.all(np.isnan(od_readings)):
            print("All readings are zero or NaN, skipping tau detection")
            return None, None


   
        smoothed = np.convolve(od_readings, np.ones(window) / window, mode='same')

    
        baseline = np.mean(smoothed[:window * 2])

    
        if np.isnan(baseline):
            print("Baseline is NaN, skipping tau")
            return None, None

        print(f"Baseline OD: {baseline:.4f}, Threshold: {threshold}")


        for i, val in enumerate(smoothed):
            if val > baseline + threshold:
                return time_points[i], i

        print("Tau not detected")
        return None, None


    def calculateSaturate(self, od_readings):
        od_readings = np.array(od_readings, dtype=np.float64)
        max_index = np.argmax(od_readings)

        if max_index == 0:
            K = np.mean(od_readings[:3])
        elif max_index == len(od_readings) - 1:
            K = np.mean(od_readings[-3:])
        else:
            K = np.mean(od_readings[max_index - 1:max_index + 2])

        return K, max_index

    def calculateInitialGrowthRate(self, od_readings, time_points, start_index, end_index):
        od_readings = np.array(od_readings, dtype=np.float64)

        try:
            log_od = np.log(od_readings[start_index:end_index + 1])
            slopes = np.gradient(log_od, time_points[start_index:end_index + 1])

            Q1, Q3 = np.percentile(slopes, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            valid_indices = np.where((slopes > lower_bound) & (slopes < upper_bound))[0]
            filtered_slopes = slopes[valid_indices]

            if len(filtered_slopes) == 0:
                return None, None

            max_idx = np.argmax(filtered_slopes)
            original_idx = valid_indices[max_idx] + start_index

            if original_idx == start_index:
                r = np.mean(slopes[:start_index + 3])
            elif original_idx == end_index:
                r = np.mean(slopes[end_index - 2:])
            else:
                r = np.mean(slopes[original_idx - 1:original_idx + 2])

            return r, original_idx

        except Exception:
            return None, None

    def compute_params(self):
        results = []

        for well_name, col in self.well_data.items():
        # Convert to numpy array for numeric checks
            od_readings = np.array(col, dtype=np.float64)

        # Skip wells where all values are NaN or all zero
            if np.all(np.isnan(od_readings)) or np.all(od_readings == 0):
                print(f"Skipping well {well_name}: empty or invalid readings.")
                continue

            tau_val, tau_idx = self.find_tau(od_readings, self.time_points)
            K_val, K_idx = self.calculateSaturate(od_readings)

        # Safeguard against invalid tau/K
            if tau_idx is None or K_idx is None or K_idx <= tau_idx:
                print(f"Skipping well {well_name}: could not compute valid tau or saturation.")
                continue

            r_val, r_idx = self.calculateInitialGrowthRate(
                od_readings,
                self.time_points,
                tau_idx,
                K_idx - 1
            )

            results.append({
                'Well': well_name,
                'tau_values': tau_val,
                'tau_index': tau_idx,
                'GrowthRates': r_val,
                'growth_rates_index': r_idx,
                'saturate_values': K_val,
                'saturate_index': K_idx,
                'saturation_time': self.time_points[int(K_idx)] if pd.notna(K_idx) else np.nan
            })

        self.growth_params = pd.DataFrame(results)

    def output_csv(self, filename):
        self.growth_params.to_csv(filename, sep='\t', index=False)

    def get_growth_params(self):
        return self.growth_params

    def plot_raw_data(self, save_path=None, wells=None, max_cols=12):
        import math

        if self.growth_params is None or self.well_data.empty:
            print("No data to plot. Please compute parameters first.")
            return

        if wells is None:
            wells = self.well_data.columns.tolist()

        n_wells = len(wells)
        n_cols = min(max_cols, n_wells)
        n_rows = math.ceil(n_wells / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(n_cols * 2, n_rows * 2),
                                squeeze=False,
                                sharex=True, sharey=True)

        for idx, well_id in enumerate(wells):
            i, j = divmod(idx, n_cols)
            ax = axs[i][j]

            if well_id not in self.well_data.columns:
                ax.axis("off")
                continue

            y_data = self.well_data[well_id]
            ax.plot(self.time_points, y_data, label="OD")

            growth_row = self.growth_params[self.growth_params["Well"] == well_id]
            if not growth_row.empty:
                tau_idx = growth_row["tau_index"].iloc[0]
                sat_idx = growth_row["saturate_index"].iloc[0]
                if pd.notna(tau_idx):
                    ax.axvline(self.time_points[int(tau_idx)], color='red', linestyle='--', lw=0.8, label='Tau')
                if pd.notna(sat_idx):
                    ax.axvline(self.time_points[int(sat_idx)], color='green', linestyle='--', lw=0.8, label='Saturation')

            ax.set_title(well_id, fontsize=8)
            ax.tick_params(labelsize=6)

        for idx in range(n_wells, n_rows * n_cols):
            i, j = divmod(idx, n_cols)
            axs[i][j].axis("off")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_single_well(self, well_id):
        growth_data_parameters = self.growth_params.set_index('Well').loc[well_id]
        growth_rate = growth_data_parameters["GrowthRates"]
        growth_rates_index = growth_data_parameters["growth_rates_index"]

        data_plot = self.well_data[well_id]
        original_time_point = self.time_points[int(growth_rates_index)]
        growth_rate_point = data_plot[growth_rates_index]

        plt.plot(self.time_points, data_plot)
        plt.plot(original_time_point, growth_rate_point, 'ro')

        slope_in_original = growth_rate * growth_rate_point
        x_vals = np.linspace(original_time_point - 2000.0, original_time_point + 2000.0, 100)
        y_vals = slope_in_original * (x_vals - original_time_point) + growth_rate_point
        plt.plot(x_vals, y_vals, 'r-', label='Gradient Line at Point')

        plt.show()


if __name__ == "__main__":
    from src.DataTransformer import TecanDataTransformer

    data_path = "../data/2023-12-13_task_plate_read_data_rep1.csv"
    initial_data = TecanDataTransformer.load_data(data_path)
    transformed_data = TecanDataTransformer.transform_data(initial_data)
    well_data = TecanDataTransformer.get_transformed_data(transformed_data)

    # 🔷 Auto-detect layout
    plate_shape = Wellplate.detect_plate_layout(well_data.columns)
    well_plate_rep_1 = Wellplate(plate_shape, well_data, 'repetition_one')

    well_plate_rep_1.plot_raw_data()

