from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

@dataclass(frozen = True)
class euteticInterface:
    eutectic_df: pd.DataFrame
    relative_eutectic_depth: np.ndarray = field(metadata={'unit':'Relative Depth Penetration Percentage'})
    eutectic_depth: np.ndarray = field(metadata={'unit':'Kilometers'})
    longitude: np.ndarray

    minumum_depth: float = field(metadata={'unit': 'Kilometers'}, init = False)
    maximum_depth: float = field(metadata={'unit': 'Kilometers'}, init = False)
    mean_depth: float = field(metadata={'unit': 'Kilometers'}, init = False)
    std: float = field(metadata={'unit': 'Kilometers'}, init = False)


    def __post_init__(self):
        object.__setattr__(self, 'minumum_depth', np.min(self.eutectic_depth))
        object.__setattr__(self, 'maximum_depth', np.max(self.eutectic_depth))
        object.__setattr__(self, 'mean_depth', np.mean(self.eutectic_depth))
        object.__setattr__(self, 'std', np.std(self.eutectic_depth))
    
    @classmethod
    def from_temp2D(cls, TempProfile2D) -> 'euteticInterface': # should go in eutetic class
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
        eutectic_df = eutectic_df,
        relative_eutectic_depth = eutectic_df['relative_depth'], 
        eutectic_depth = eutectic_df['depth'],
        longitude = eutectic_df['longitude']
        )
    

    def save_eutectic_data(self, folderpath):
        to_save_df = self.eutectic_df

        to_save_df.rename(columns = {'depth': 'Depth [Km]', 
                                        'shell_number': 'Shell Number', 
                                        'relative_depth':'Relative Depth [%]', 
                                        'longitude': 'Longitude [°]',
                                        'temp': 'Temperature [K]'}, inplace = True)

        eutetic_df_filename = self.filename.replace("_2D_data.txt", '_2D_eutectic_data.txt') 
        to_save_df.to_csv(folderpath+eutetic_df_filename, index = False)

        
    def __repr__(self):
        return f'''Angular Ratio: {np.max(self.angular_ratio)},
Standard Deviation: {self.std} Km,
Relative Depth Penetration Range: {self.minumum_depth} - {self.maximum_depth} Km'''
    



@dataclass(frozen = True)
class TempProfile1D:
    filename: str
    temp_array: np.ndarray = field(metadata={'unit':'kelvin'})
    radius_array: np.ndarray = field(metadata={'unit':'meters'})
    density_array: np.ndarray = field(metadata={'unit':'kg/m3'})
    salt_concentration_array: np.ndarray = field(metadata={'unit': 'wt%'})

    Ro: float = field(metadata = {'unit': 'meters'}, init = False, repr=False)
    Rs: float = field(metadata = {'unit': 'meters'}, init = False, repr=False)
    To: float = field(metadata = {'unit': 'kelvin'}, init = False, repr=False)
    Ts: float = field(metadata = {'unit': 'kelvin'}, init = False, repr=False)
    d_ice: float = field(metadata = {'unit': 'kelvin'}, init = False, repr=False)
    depth_array: np.ndarray = field(metadata={'unit':'meters'}, init=False, repr=False)
    resolution: float = field(metadata= {'unit': 'meters'}, init=False, repr=False)


    def __post_init__(self):
        object.__setattr__(self, 'Ro', np.min(self.radius_array))
        object.__setattr__(self, 'Rs', np.max(self.radius_array))
        object.__setattr__(self, 'To', np.max(self.temp_array))
        object.__setattr__(self, 'Ts', np.min(self.temp_array))
        object.__setattr__(self, 'd_ice', np.round(self.Rs) - np.round(self.Ro))
        object.__setattr__(self, 'depth_array', abs(self.radius_array - self.Rs))
        object.__setattr__(self, 'resolution', np.median(np.diff(np.unique(self.radius_array)))*1000)


    @classmethod
    def from_filepath(cls, filepath: str) -> 'TempProfile1D':
        profile_array = np.loadtxt(filepath, comments='#')

        return cls(
            filename = filepath.split("/")[-1],
            radius_array=profile_array[:, 0],
            temp_array=profile_array[:, 1],
            density_array = profile_array[:, 2],
            salt_concentration_array = profile_array[:, 3]
        )
    


    def __repr__(self):
        return f'''Temp range: {np.round(self.Ts, 1)} K - {np.round(self.To, 1)} K,
Ice Shell Depth: {self.d_ice} Km,   
Resolution: {np.round(self.resolution)} m'''




@dataclass(frozen = True)
class TempProfile2D(TempProfile1D):
    x_array: np.ndarray = field(metadata={'unit':'Kilometers'})
    y_array: np.ndarray = field(metadata={'unit':'Kilometers'})
    z_array: np.ndarray = field(metadata={'unit':'Kilometers'})

    radius_array: np.ndarray = field(metadata={'unit':'Kilometers'}, init = False)
    unique_radius_array: np.ndarray = field(metadata={'unit':'Kilometers'}, init = False)
    nCellsPerShell: int = field(init=False)
    nShells: int = field(init=False)
    eutectic_interface: euteticInterface = field(init = False)


    def __post_init__(self):
        object.__setattr__(self, 'radius_array', np.round(np.sqrt(self.x_array**2 + self.y_array**2 + self.z_array**2), 3))
        super().__post_init__()
        object.__setattr__(self, 'unique_radius_array', np.unique(self.radius_array, return_counts=True))
        object.__setattr__(self, 'nShells', np.max(self.unique_radius_array[1]))
        object.__setattr__(self, 'nCellsPerShell', int(len(self.radius_array)/self.nShells))
        object.__setattr__(self, 'eutectic_interface', euteticInterface.from_temp2D(self))

    
    def plot_relative_eutectic_depth(self, folderpath = False):
        sns.lineplot(x = self.eutectic_interface.longitude, y = self.eutectic_interface.relative_eutectic_depth)

        plt.xlabel('Longitude [°]', fontsize = 12)
        plt.ylabel('Relative Pen. Depth [%]', fontsize = 12)
        filename_split = self.filename.split('_')
        viscosity = filename_split[2].split("eta")[-1]

        plt.title(f'Ice Shell Dpeth: {self.d_ice} Km, Viscosity: {viscosity} Pa s')
        plt.grid()

        ax = plt.gca()
        ax.set_ylim(np.min(self.eutectic_interface.relative_eutectic_depth)-0.5, np.max(self.eutectic_interface.relative_eutectic_depth)+0.5)
        ax.set_xlim(0,62)
        ax.invert_yaxis()

        if folderpath:
            plot_filename = self.filename.replace('2D_data.txt', '2D_eutectic_plot.png')
            plot_filepath = folderpath+plot_filename
            plt.savefig(plot_filepath)
        plt.show()


    def plot_temp_twod(self):
        twod_x_array = self.x_array.reshape([self.nCellsPerShell,  self.nShells])
        twod_y_array = self.y_array.reshape([self.nCellsPerShell,  self.nShells])
        twod_temp_array = self.temp_array.reshape([self.nCellsPerShell,  self.nShells])

        fig, ax = plt.subplots(1, 1, figsize = (2, 4), dpi = 250)
        fontsize = 8

        cmap = sns.color_palette("inferno", as_cmap = True)
        cbar_ax = [0, 0.085, 0.9, 0.015]

        conjtourplot = ax.contourf(twod_x_array, twod_y_array, twod_temp_array, np.linspace(self.Ts, self.To,256), cmap = cmap, extend="both")
        ax.axis('off')
        plt.show()
    


    @classmethod
    def from_filepath(cls, filepath: str) -> 'TempProfile2D':
        profile_array = np.loadtxt(filepath, comments='#')

        return cls(
            filename = filepath.split('/')[-1],
            x_array=profile_array[:, 0],
            y_array=profile_array[:, 1],
            z_array = profile_array[:, 2],
            temp_array = profile_array[:, 3],
            density_array = profile_array[:, 4],
            salt_concentration_array = profile_array[:, 5]
        )

    
    def __repr__(self):
        return f'''Temp range: {np.round(self.Ts, 1)} K - {np.round(self.To, 1)} K,
Ice Shell Depth: {self.d_ice} Km,   
Resolution: {np.round(self.resolution)} m,
nShells: {self.nShells},
nCellsPerShell: {self.nCellsPerShell}'''