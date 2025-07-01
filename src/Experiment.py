import numpy as np 
import matplotlib.pyplot as plt

class Experiment():
    def __init__(self, wellplates,title=None,owner=None):
        self.wellplates = wellplates
        self.experiment_title = title 
        self.owner = owner
        self.base_layout = self.wellplates[0].layout
    
    def plot_combined_data(self,shared_y=False):
        row_num,col_num = self.base_layout
        #expects the input to be list of tuple.[(plate_readings_well_name_as_columns,time_points),...]
        fig, axs = plt.subplots(row_num, col_num, figsize=(col_num, row_num),sharey = shared_y)  # Adjust figsize as needed
        # Loop over the DataFrame to plot each well
        rows = "".join([ chr(ord('A') + i ) for i in range(row_num)])
        for i, row in enumerate(rows):
            for j in range(1, col_num+1):
                well_id = f"{row}{j}"
                if well_id in self.wellplates[0].well_data.columns:
                    for well_plate in self.wellplates :
                        axs[i, j - 1].plot(well_plate.time_points,well_plate.well_data[well_id])  # Adjust this line for your specific plot type
                    axs[i, j - 1].set_title(well_id, fontsize=8)
                    axs[i, j - 1].tick_params(labelsize=6)

        # Adjust the layout
        plt.subplots_adjust(hspace=1, wspace=1)  # Adjust the spacing as needed

        # Show or save the plot
        plt.show()
    
    def validate_same_layout(self):
        for well in self.wellplates:
            if well.layout != self.base_layout:
                return False 
        return True
    
    def analyze_data(self):
        
        """
        TODO , This function will be used to compare the statistics and difference for different experiments 
        this will be the continuous imporvement work
        """
        pass 
    