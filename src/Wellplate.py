import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

class Wellplate(): 
    def __init__(self,layout,well_data,well_plate_name=None,start_time=None): 
        self.layout = layout #tuple of (row,col) for example 384 wellplate has layout of (16,24)
        self.time_points = well_data['Time [s]'].astype(np.float64).values
        self.well_data = well_data.drop(['Time [s]'],axis=1)
        self.start_time = start_time
        
        self.well_plate_name = well_plate_name
        self.growth_params = pd.DataFrame()
        self.compute_params()

    def find_tau(self, od_readings, time_points, window=3, threshold=0.03):
        od_readings = np.array(od_readings).astype(np.float64)
        time_points = np.array(time_points).astype(np.float64)

        if len(od_readings) < 5:
            return None, None

   
        smoothed = np.convolve(od_readings, np.ones(window)/window, mode='same')

   
        baseline = np.mean(smoothed[:window*2])
        print(f"Baseline OD: {baseline:.4f}, Threshold: {threshold}")
   
        for i, val in enumerate(smoothed):
            if val > baseline + threshold:
                return time_points[i], i
        print("Tau not detected")
        return None, None
    
    def calculateSaturate(self, od_readings):
        od_readings = np.array(od_readings).astype(np.float64)
      
        max_index = np.argmax(od_readings)
      
        if max_index == 0:
            K = np.mean(od_readings[:3])
        elif max_index == len(od_readings) - 1:
            K = np.mean(od_readings[-3:])
        else:
            K = np.mean(od_readings[max_index-1:max_index+2])
        return K, max_index
    
    def calculateInitialGrowthRate(self,od_readings, time_points, start_index, end_index):
        od_readings = np.array(od_readings).astype(np.float64)

        try:
        # Calculate logarithmic slopes
            log_od_readings = np.log(od_readings[start_index:end_index+1])
            slopes = np.gradient(log_od_readings, time_points[start_index:end_index+1])

            #Remove outliers
            Q1, Q3 = np.percentile(slopes, [25, 75])
            
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            valid_indices = np.where((slopes > lower_bound) & (slopes < upper_bound))[0]
            filtered_slopes = slopes[valid_indices]
            
#             log_od_readings = np.log(od_readings)
#             slopes = np.gradient(log_od_readings, time_points)

#             valid_indices = np.arange(0,len(od_readings)+1)
#             filtered_slopes = slopes
            
            # Choose the maximal slope
            max_slope_index = np.argmax(filtered_slopes)
            original_max_slope_index = valid_indices[max_slope_index] + start_index

            # Average the maximal slope with its two neighboring slopes in the original data
            if original_max_slope_index == start_index:
                r = np.mean(slopes[:start_index+3])
            elif original_max_slope_index == end_index:
                r = np.mean(slopes[end_index-2:])
            else:
                r = np.mean(slopes[original_max_slope_index-1:original_max_slope_index+2])

            return r, original_max_slope_index
        except Exception as e:
          # handle the exception
          return None,None
        
    def calculateGrowth(self, od_readings, time_points):
        time_point, start_index = self.find_tau(od_readings, time_points)
        K, end_index = self.calculateSaturate(od_readings)
        r, growth_rate_index = self.calculateInitialGrowthRate(od_readings, time_points, start_index, end_index-1)
        return r, growth_rate_index

        
    def compute_params(self):
        results = []

        for well_name, col in self.well_data.items():
            tau_val, tau_idx = self.find_tau(col, self.time_points)
            K_val, K_idx = self.calculateSaturate(col)
            r_val, r_idx = self.calculateInitialGrowthRate(col, self.time_points, tau_idx, K_idx - 1)

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

    
        
    def plot_raw_data(self, save_path=None, wells=None, max_cols=12):
        """
        Plot the raw data of the wells, optionally only a subset.

        Args:
            save_path (str): if provided, save the figure to this path.
            wells (list): list of well IDs to plot. If None, plot all available.
            max_cols (int): number of columns in the figure grid.
        """
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

        # draw tau and saturation if available
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

    # Hide unused axes
        for idx in range(n_wells, n_rows * n_cols):
            i, j = divmod(idx, n_cols)
            axs[i][j].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_single_well(self,well_id):
        
        ### issue to be fixed with this plot: 
        """
        when calculating the growth rate, we are calculating between the tau index and saturation index, however, 
        when plotting, we are plotting on whole time frame. so the growth plot might not be accurate at this point
        """
        #extract the growth parameters
        growth_data_parameters = self.growth_params.set_index('Well').loc[well_id]
        growth_rate = growth_data_parameters["GrowthRates"]
        growth_rates_index = growth_data_parameters["growth_rates_index"]
        
        #get raw data point
        data_plot = self.well_data[well_id]
        original_time_point = self.time_points[int(growth_rates_index)]
        
        # start_index = growth_data_parameters["tau_index"]
        # end_index = growth_data_parameters["saturate_index"]
        growth_rate_point = data_plot[growth_rates_index] #when representing as a point , (x,y) => (original_time_point,growth_rate_point)
         
        #plot the raw data and the growth point
        plt.plot(self.time_points,data_plot)
        plt.plot(original_time_point, growth_rate_point, 'ro')
        
        slope_in_original = growth_rate * growth_rate_point #calculate slope at that time
        # Extend the range of x_vals symmetrically
        extended_range = 2000.0  # Adjust for desired line length
        x_vals = np.linspace(original_time_point - extended_range, original_time_point + extended_range, 100)
        
        # Calculate y_vals using the linear equation (y = mx + c)
        y_vals = slope_in_original * (x_vals - original_time_point) + growth_rate_point
        plt.plot(x_vals, y_vals, 'r-', label='Gradient Line at Point')
        
    
    def get_growth_params(self): 
        return self.growth_params
    
    def match_data_with_layout(self):
        
        row_num,col_num = self.layout
        rows = "".join([ chr(ord('A') + i ) for i in range(row_num)])
        unavailable_wells = [] 
        for i, row in enumerate(rows):
            for j in range(1, col_num+1):
                well_id = f"{row}{j}"
                if well_id in self.well_data.columns:
                    continue
                else:
                    unavailable_wells.append(well_id)
        if unavailable_wells:
            print("Following well id are not available in the data",unavailable_wells)
        else:
            print("All of the well id are available")
    
    def output_csv(self,filename): 
        self.growth_params.to_csv(filename, sep='\t')
    

if __name__ =="__main__":
    from src.DataTransformer import TecanDataTransformer
    data_path = "../data/2023-12-13_task_plate_read_data_rep1.csv"
    initial_data = TecanDataTransformer.load_data(data_path)
    transformed_data = TecanDataTransformer.transform_data(initial_data)
    well_data = TecanDataTransformer.get_transformed_data(transformed_data)

    well_plate_rep_1 = Wellplate((16,24),well_data,'repetition_one')
    well_plate_rep_1.plot_raw_data()