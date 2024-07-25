from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


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
    x_array: np.ndarray = field(metadata={'unit':'meters'})
    y_array: np.ndarray = field(metadata={'unit':'meters'})
    z_array: np.ndarray = field(metadata={'unit':'meters'})

    radius_array: np.ndarray = field(metadata={'unit':'meters'}, init = False)
    unique_radius_array: np.ndarray = field(metadata={'unit':'meters'}, init = False)
    nCellsPerShell: int = field(init=False)
    nShells: int = field(init=False)

    relative_eutectic_depth: np.ndarray = field(metadata={'unit':'Relative Depth Penetration Percentage'}, init = False)
    angular_ratio: np.ndarray = field(init = False)


    def __post_init__(self):
        object.__setattr__(self, 'radius_array', np.round(np.sqrt(self.x_array**2 + self.y_array**2 + self.z_array**2), 3))
        super().__post_init__()
        object.__setattr__(self, 'unique_radius_array', np.unique(self.radius_array, return_counts=True))
        object.__setattr__(self, 'nCellsPerShell', self.unique_radius_array[0].shape[0])
        object.__setattr__(self, 'nShells', np.max(self.unique_radius_array[1]))
        eutectic_df = self.get_eutectic_data()
        object.__setattr__(self, 'relative_eutectic_depth', eutectic_df['depth'])
        object.__setattr__(self, 'angular_ratio', eutectic_df['a_r'])


    def get_eutectic_data(self) -> pd.DataFrame:
        df_list = []
        big_depth_array = self.depth_array.reshape([self.nCellsPerShell,  self.nShells])
        big_temp_array =  self.temp_array.reshape([self.nCellsPerShell, self.nShells])
        shell_indexes = np.linspace(1, self.nShells, self.nShells)

        for radial_depth_array, radial_temp_array, shell_number in zip(big_depth_array.T, big_temp_array.T, shell_indexes):
            small_df_dict = {
                'shell_number': np.linspace(shell_number, shell_number, self.nCellsPerShell),
                'depth': radial_depth_array[::-1],
                'temp': radial_temp_array[::-1]
            }
            small_df = pd.DataFrame(small_df_dict)
            df_list.append(small_df)

        big_df = pd.concat(df_list)
        eutectic_df = ((big_df.query("temp <= 240").groupby(['shell_number'])['depth'].max()/self.d_ice)*100).reset_index()
        eutectic_df['a_r'] = (eutectic_df['shell_number']/self.nShells)*25

        return eutectic_df


    def plot_temp_twod(self):
        twod_x_array = self.x_array.reshape([self.nCellsPerShell,  self.nShells])
        twod_y_array = self.y_array.reshape([self.nCellsPerShell,  self.nShells])
        twod_temp_array = self.temp_array.reshape([self.nCellsPerShell,  self.nShells])

        fig, ax = plt.subplots(1, 1, figsize = (2, 4), dpi = 500)
        fontsize = 8

        cmap = sns.color_palette("inferno", as_cmap = True)
        cbar_ax = [0, 0.085, 0.9, 0.015]

        conjtourplot = ax.contourf(twod_x_array, twod_y_array, twod_temp_array, np.linspace(self.Ts, self.To,256), cmap = cmap, extend="both")
        # ticksstep = 40
        # cbarticks = np.arange(self.Ts - (self.Ts%ticksstep), self.To - (self.To%ticksstep) + ticksstep, ticksstep)
        # cbarticks = np.arange(self.Ts, self.To, ticksstep)
        # cbar_ax = fig.add_axes(cbar_ax)
        # cb = fig.colorbar(conjtourplot, ticks=cbarticks, orientation = 'horizontal', cax = cbar_ax)
        # cb.set_label(label="Temperature [K]", fontsize = fontsize) #22 # 28
        # cb.set_ticklabels(ticklabels=cbarticks, fontsize = fontsize-1.5) #20 # 26

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
