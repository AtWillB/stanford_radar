from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import collections  as mc
from scipy.signal import argrelextrema # whats different?
from scipy.signal import find_peaks # Whats different?
from scipy import fft



@dataclass(frozen = True)
class EuteticInterface:
    df: pd.DataFrame
    relative_depth: np.ndarray = field(metadata={'unit':'Relative Depth Penetration Percentage'})
    depth: np.ndarray = field(metadata={'unit':'Kilometers'})
    longitude_span: np.ndarray
    d_ice: float = field(metadata = {'unit': 'Kilometers'})

    minumum_depth: float = field(metadata={'unit': 'Kilometers'}, init = False)
    maximum_depth: float = field(metadata={'unit': 'Kilometers'}, init = False)
    mean_depth: float = field(metadata={'unit': 'Kilometers'}, init = False)
    std: float = field(metadata={'unit': 'Kilometers'}, init = False)
    upwelling_heights: np.ndarray = field(metadata={'unit': 'Kilometers'}, init = False)
    downwelling_heights: np.ndarray = field(metadata={'unit': 'Kilometers'}, init = False)
    chir: np.ndarray = field(init = False) # Convective Height Imbalance Ratio
    num_of_convection_cells: int = field(init = False)
    avg_upwelling_depth: float = field(metadata = {'unit': 'Kilometers'}, init =False)


    def __post_init__(self):
        object.__setattr__(self, 'minumum_depth', np.min(self.depth))
        object.__setattr__(self, 'maximum_depth', np.max(self.depth))
        object.__setattr__(self, 'mean_depth', np.mean(self.depth))
        object.__setattr__(self, 'std', np.std(self.depth))
        depth_to_ocean = self.d_ice - self.depth
        object.__setattr__(self, 'upwelling_heights', depth_to_ocean[find_peaks(depth_to_ocean)[0]])
        object.__setattr__(self, 'downwelling_heights', depth_to_ocean[find_peaks(-depth_to_ocean)[0]])
        if self.std > 0.01:
            object.__setattr__(self, 'num_of_convection_cells', self.upwelling_heights.shape[0])
            object.__setattr__(self, 'avg_upwelling_depth', self.d_ice - np.mean(self.upwelling_heights))
        else:
            object.__setattr__(self, 'num_of_convection_cells', 0)
            object.__setattr__(self, 'avg_upwelling_height', 0)
        object.__setattr__(self, 'chir', (sum(self.upwelling_heights) - sum(self.downwelling_heights))/self.num_of_convection_cells)
    

    @classmethod
    def from_temp2D(cls, TempProfile2D) -> 'EuteticInterface': # should go in eutetic class
        df_list = []
        big_depth_array = TempProfile2D.depth_array.reshape([TempProfile2D.nCellsPerShell,  TempProfile2D.nShells])
        big_temp_array =  TempProfile2D.temp_array.reshape([TempProfile2D.nCellsPerShell, TempProfile2D.nShells])
        shell_indexes = np.linspace(1, TempProfile2D.nShells, TempProfile2D.nShells)

        for radial_depth_array, radial_temp_array, shell_number in zip(big_depth_array.T, big_temp_array.T, shell_indexes):
            small_df_dict = {
                'shell_number': np.linspace(shell_number, shell_number, TempProfile2D.nCellsPerShell),
                'depth': radial_depth_array[::-1],
                'temp': radial_temp_array[::-1]
            }
            small_df = pd.DataFrame(small_df_dict)
            df_list.append(small_df)

        big_df = pd.concat(df_list)
        eutectic_df = (big_df.query("temp <= 240").groupby(['shell_number'])['depth'].max()).reset_index()
        eutectic_df['relative_depth'] = (eutectic_df['depth']/TempProfile2D.d_ice)*100
        eutectic_df['longitude'] = (eutectic_df['shell_number']/TempProfile2D.nShells)*62
        eutectic_df['temp'] = big_df.query("temp <= 240").groupby(['shell_number'])['temp'].max().values

        return cls(
        df = eutectic_df,
        relative_depth = eutectic_df['relative_depth'].values, 
        depth = eutectic_df['depth'].values,
        longitude_span = eutectic_df['longitude'].values,
        d_ice = TempProfile2D.d_ice
        )
    

    def calc_sparse_eutectic(self, plot = True, depth_limit = True):

        number_of_groups = np.linspace(0, self.relative_depth.shape[0]-1, 25, dtype = int)
        line_segment_list = []


        for group_index in number_of_groups[:-1]:
            curr_index = np.where(number_of_groups == group_index)[0][0]
            dice_roll = np.random.randint(1, 10)

            line_segment_depth = self.relative_depth[number_of_groups[curr_index]:number_of_groups[curr_index+1]+1]
            line_segment_angular = self.longitude_span[number_of_groups[curr_index]:number_of_groups[curr_index+1]+1]

            if dice_roll < 7:
                line_segment_depth = np.zeros(len(line_segment_depth))

            line_segment = np.array([(a_r, depth) for depth, a_r in zip(line_segment_depth, line_segment_angular)])
            line_segment_list.append(line_segment)
        
        # if depth_limit is True:
        #     depth_min = np.min(self.relative_depth)
        #     depth_max = np.max(self.relative_depth)
        #     if int(depth_min) - int(depth_min) == 0:
        #         relative_pen_limit = np.random.randint(int(depth_min)-2, int(depth_max) + 3)
        #     else:
        #         relative_pen_limit = np.random.randint(depth_min, depth_max)
        #     for line_segment in line_segment_list:
        #         for i in range(len(line_segment)):
        #             if line_segment[i][1] < relative_pen_limit:
        #                 line_segment[i][1] = 0

        plotting_list = [line_segment for line_segment in line_segment_list if np.max(line_segment[:, 1]) != 0]

        new_plotting_list = []
        for plotting_line_segment in plotting_list:
            new_segment = np.array([(x, y) for x, y in plotting_line_segment if y != 0])
            new_plotting_list.append(new_segment)

        if plot == True:
            self.plot_sparse_eutectic(new_plotting_list)
        return np.vstack(line_segment_list)


    def predict_eutetic(self, depth_array):

        prediction_dict ={"signal_std": 0,
                          "scaled_frequency_std": 0, 
                          "num_low_freq_impules": 0, 
                          "prediction": "empty"}

        if np.sum(depth_array) == 0:
            return prediction_dict
        prediction_dict['signal_std'] = np.round(np.std(depth_array[np.nonzero(depth_array)]), 5)
        yf = np.abs(fft.rfft(depth_array))

        yf[0] = 0
        stdf = np.std(yf)
        prediction_dict['scaled_frequency_std'] = np.round(stdf*15, 5)

        local_maxima_index = argrelextrema(yf, np.greater)
        yf_local = yf[local_maxima_index]
       
        num_low_freq_impules = yf_local[yf_local > stdf*15].shape[0]
        prediction_dict['num_low_freq_impules'] = num_low_freq_impules

        if prediction_dict['signal_std'] < 0.001:
            prediction_dict['prediction'] = 'conductive'
        else:
            if  prediction_dict['num_low_freq_impules'] < 2:
                prediction_dict['prediction'] = 'vigorously convective'
            else:
                prediction_dict['prediction'] = 'sluggishly convective'
            
        return prediction_dict


    def plot_sparse_eutectic(self, new_plotting_list):
        lc = mc.LineCollection(new_plotting_list, linewidths=2)
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.set_ylim(np.min(self.relative_depth)-0.5, np.max(self.relative_depth)+0.5)
        ax.set_xlim(0,np.max(self.longitude_span))
        ax.invert_yaxis()

        plt.xlabel('Longitude [°]', fontsize = 12)
        plt.ylabel('Relative Pen. Depth [%]', fontsize = 12)
        plt.title("Sparse Eutectic Depth")
        plt.grid()
        plt.show()

    
    def plot_fft(self, depth_array): # might be worth it to look at seting up a class just for this & for sparsity stuff
        yf = np.abs(fft.rfft(depth_array))
        xf = fft.rfftfreq(len(depth_array))
        yf[0] = 0
        stdf = np.std(yf)

        local_maxima_index = argrelextrema(yf, np.greater)
        yf_local = yf[local_maxima_index]
        xf_local = xf[local_maxima_index]


        plt.plot(xf, yf)
        plt.plot(np.linspace(0, 0.5, 1000), np.linspace(stdf, stdf, 1000)*15)
        plt.scatter(xf_local, yf_local)
        ax = plt.gca()

        plt.title("FFT of Relative Eutectic Depth")
        plt.xlabel("Frequency [non-dimensionalized]")
        plt.ylabel("Power")
        plt.grid()
        ax.set_xlim(0, .01)
        plt.show()


        
    def __repr__(self):
        return f'''Longitude: {np.max(self.longitude_span)},
Standard Deviation: {self.std} Km,
Relative Depth Penetration Range: {self.minumum_depth} - {self.maximum_depth} Km'''
    


