import os
import scipy.io
import h5py
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import easygui as eg
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class EVizTool:
    def __init__(self,event_file_name, event_data_frame, sensor_size):
        """
        # initialization of EVizToll class containing custom methods for data visualisation
        :param event_file_name: string - "dvs_test.mat" or "dvs_test.h5"
        :param event_data_frame: pandas data frame containing event based data - t, x, y, p (0 / 1)
        :param sensor_size: tuple
        """
        self.event_file_name = event_file_name

        self.event_data_frame = event_data_frame
        self.t = event_data_frame["t"].values
        self.x = event_data_frame["x"].values
        self.y = event_data_frame["y"].values
        self.p = event_data_frame["p"].values

        # print(self.t, self.t[0], self.t[-1])

        # making OFF polarity convention to always 0
        self.p[self.p == -1] = 0
        self.sensor_size = sensor_size
        self.sensor_x_max, self.sensor_y_max = sensor_size[0], sensor_size[1]

    def plot_event_histogram(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.p, bins=3, edgecolor="black", alpha=0.7)
        plt.title(f'Event Polarity Histogram - {self.event_file_name}')
        # plt.xticks([1, -1], labels=["ON (1)", "OFF (-1)"])
        plt.ylabel("Count")

    def plot_temporal_kernel_density(self):
        plt.figure(figsize=(10, 5))
        sns.kdeplot(self.t[self.p == 1], label="ON Events", fill=True, cmap="Blues")
        sns.kdeplot(self.t[self.p == 0], label="OFF Events", fill=True, cmap="Reds")
        plt.title(f'Temporal Kernel Density Visualization - {self.event_file_name}')
        plt.xlabel("Time")
        plt.ylabel("Density")
        plt.legend()

    def plot_spatial_kernel_density(self):
        pass

    def plot_event_on_off_map(self):
        x_on, y_on = self.x[self.p == 1], self.y[self.p == 1]
        x_off, y_off = self.x[self.p == 0], self.y[self.p == 0]
        fig1, ax = plt.subplots(1, 2, figsize=(12, 5))
        # map for ON events
        hb1 = ax[0].hexbin(x_on, y_on, gridsize=50, cmap="Blues", mincnt=1)
        ax[0].set_title(f"ON Events Density - {self.event_file_name}")
        ax[0].set_xlabel("X Coordinate")
        ax[0].set_ylabel("Y Coordinate")
        plt.colorbar(hb1, ax=ax[0])

        # map for OFF events
        hb2 = ax[1].hexbin(x_off, y_off, gridsize=50, cmap="Reds", mincnt=1)
        ax[1].set_title(f"OFF Events Density - {self.event_file_name}")
        ax[1].set_xlabel("X Coordinate")
        plt.colorbar(hb2, ax=ax[1])
        plt.tight_layout()

    def plot_polarity_count_at_given_pixel(self):
        polarity_counts = self.event_data_frame.groupby(['x', 'y'])['p'].value_counts().unstack(fill_value=0)
        polarity_counts.columns = ['p=0', 'p=1']
        polarity_counts['intensity'] = 255 * (polarity_counts['p=1'] - polarity_counts['p=0'])
        max_pixel = polarity_counts.sum(axis=1).idxmax()
        min_pixel = polarity_counts.sum(axis=1).idxmin()

        selected_pixels = [max_pixel, min_pixel]
        # selected_pixels = [(0, 32), (0, 34), (0, 36), (117, 68)]

        fig, axes = plt.subplots(1, len(selected_pixels), figsize=(15, 4))

        for ax, (x, y) in zip(axes, selected_pixels):
            if (x, y) in polarity_counts.index:
                data = polarity_counts.loc[(x, y)]
                ax.bar(['p=0', 'p=1'], [data['p=0'], data['p=1']], color=['red', 'blue'])
                ax.set_title(f'Pixel ({x},{y})')
                ax.set_ylabel('Count')
            else:
                ax.set_title(f'Pixel ({x},{y}) - No Events')
                ax.bar(['p=0', 'p=1'], [0, 0], color=['red', 'blue'])

        plt.tight_layout()

    def plot_event_intensity_map(self):
        # occurrences of p=1 and p=0 ON/OFF at each (x, y)
        polarity_counts = self.event_data_frame.groupby(['x', 'y'])['p'].value_counts().unstack(fill_value=0)
        polarity_counts.columns = ['p=0', 'p=1']

        polarity_counts['intensity'] = 255 * (polarity_counts['p=1'] - polarity_counts['p=0'])
        intensity_map = np.zeros(self.sensor_size)
        for (x, y), row in polarity_counts.iterrows():
            if 0 <= x < self.sensor_size[0] and 0 <= y < self.sensor_size[1]:
                intensity_map[x, y] = row['intensity']

        plt.figure(figsize=(8, 6))
        plt.imshow(intensity_map.T, cmap='seismic_r', origin='upper', interpolation='nearest')
        plt.colorbar(label="Net Intensity Change")
        plt.title(f"Event Intensity Map - {self.event_file_name}")
        plt.xlabel("X Pixel")
        plt.ylabel("Y Pixel")
        plt.show()


def load_event_file(path):
    """
    # takes file path, loads the data and converts it to data frame for further processing
    :param path: string - "home/data/analysis/event_data.h5"
    :return: output data frame - pandas data frame, filename  - string, sensor size - tuple
    """

    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_path)[1]
    output_df = pd.DataFrame()
    x_max, y_max = None, None

    if file_extension == ".mat":
        mat_data = scipy.io.loadmat(path)
        mat_td_data = mat_data['TD']
        mat_obj_data = mat_data['Obj']
        mat_n_obj = mat_data['nObj']
        mat_x_max = mat_data['xMax']
        mat_y_max = mat_data['yMax']

        # mat_file_entire_data = mat_obj_data
        mat_file_entire_data = mat_td_data
        mat_file_x_pix = mat_file_entire_data[0][0][0].flatten()
        mat_file_y_pix = mat_file_entire_data[0][0][1].flatten()
        mat_file_p = mat_file_entire_data[0][0][2].flatten()
        mat_file_t = mat_file_entire_data[0][0][3].flatten()
        x_max = int(mat_x_max.flatten()[0])
        y_max = int(mat_y_max.flatten()[0])

        output_df = pd.DataFrame({'t': mat_file_t, 'x': mat_file_x_pix, 'y': mat_file_y_pix, 'p': mat_file_p})

    if file_extension == ".h5":
        with h5py.File(path, 'r') as file:
            h5_dataset = file['events']
            h5_file_entire_data = h5_dataset[:]

        h5_file_entire_data_T = h5_file_entire_data.T
        h5_file_x_pix = h5_file_entire_data_T[1]
        h5_file_y_pix = h5_file_entire_data_T[2]
        h5_file_p = h5_file_entire_data_T[3]
        h5_file_t = h5_file_entire_data_T[0]

        # we know the sensor size from simulation and rendering
        x_max, y_max = 128, 128
        # x_max, y_max = 1024, 768
        # temporal resolution in microseconds - 1e6

        output_df = pd.DataFrame({'t': h5_file_t/1e6, 'x': h5_file_x_pix, 'y': h5_file_y_pix, 'p': h5_file_p})
        # output_df = pd.DataFrame({'t': h5_file_t, 'x': h5_file_x_pix, 'y': h5_file_y_pix, 'p': h5_file_p})


    return file_name, output_df, (x_max, y_max)

try:
    base_path = "/home/amulya/PycharmProjects/myProjects/Neuromorphic_Image_Processing/DATA/"
    file_path = eg.fileopenbox(title="eVizTool dialog says select an event data file",
                                   filetypes=["*.mat", "*.h5"], default=base_path)
    # file_path = ""      # give file path manually
    if file_path:
        event_name, event_df, sensor_dim = load_event_file(file_path)
        print("sensor dimension = ", sensor_dim)
        eviz_obj = EVizTool(event_name, event_df, sensor_dim)
        # eviz_obj.plot_event_histogram()
        eviz_obj.plot_temporal_kernel_density()
        eviz_obj.plot_event_on_off_map()
        eviz_obj.plot_polarity_count_at_given_pixel()
        eviz_obj.plot_event_intensity_map()
        plt.show()

    else:
        print("No event file was selected")
except Exception as e:
    print(f"Error while loading event file - {e}")
    raise e


