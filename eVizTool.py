import cv2
import scipy.io
import scipy.signal
import numpy as np
import seaborn as sns
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

    @staticmethod
    def peak_estimation(time_bins, density):
        # finding ON and OFF peaks
        # parameters are adjusted to detect significant peaks only
        # peaks, properties = scipy.signal.find_peaks(density, height=0.1, prominence=0.05, width=2)
        peaks, properties = scipy.signal.find_peaks(density, height=0)
        peak_densities = properties["peak_heights"]

        peak_times = time_bins[peaks]

        # num_peaks = len(peaks)

        # print the number of peaks and where the
        # for i in range(num_peaks):
        #     print(f"Peak {i + 1}: Density={peak_densities[i]:.4f}, Time={peak_times[i]:.2f}")

        return peak_times, peak_densities

    def estimate_and_plot_temporal_kernel_density(self):
        plt.figure(figsize=(10, 5))

        time_bins = np.linspace(self.t.min(), self.t.max(), 100)

        kde_on = scipy.stats.gaussian_kde(self.t[self.p == 1])
        kde_off = scipy.stats.gaussian_kde(self.t[self.p == 0])

        density_on = kde_on(time_bins)
        density_off = kde_off(time_bins)

        on_peak_times, on_peak_densities = self.peak_estimation(time_bins, density_on)
        off_peak_times, off_peak_densities = self.peak_estimation(time_bins, density_off)

        sns.lineplot(x=time_bins, y=density_on, label="ON Events", color="blue")
        sns.lineplot(x=time_bins, y=density_off, label="OFF Events", color="red")

        plt.fill_between(time_bins, density_on, alpha=0.3, color="blue")
        plt.fill_between(time_bins, density_off, alpha=0.3, color="red")

        plt.title(f'Temporal Kernel Density Visualization - {self.event_file_name}')
        plt.xlabel("Time")
        plt.ylabel("Density")
        plt.legend()

        return time_bins, on_peak_times, on_peak_densities, off_peak_times, off_peak_densities

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
        ax[0].invert_yaxis()
        plt.colorbar(hb1, ax=ax[0])

        # map for OFF events
        hb2 = ax[1].hexbin(x_off, y_off, gridsize=50, cmap="Reds", mincnt=1)
        ax[1].set_title(f"OFF Events Density - {self.event_file_name}")
        ax[1].set_xlabel("X Coordinate")
        ax[1].invert_yaxis()
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

    def event_to_video(self, integration_time, start, end, apply_bounding_box):

        self.event_data_frame['time_bin'] = (self.event_data_frame['t'] // integration_time).astype(int)

        time_bins = sorted(self.event_data_frame['time_bin'].unique())[start:end]

        frames = []

        for t_bin in time_bins:
            df_bin = self.event_data_frame[self.event_data_frame['time_bin'] == t_bin]

            counts_bin = df_bin.groupby(['x', 'y'])['p'].value_counts().unstack(fill_value=0)

            for col in [0, 1]:
                if col not in counts_bin.columns:
                    counts_bin[col] = 0

            counts_bin.columns = ['p=0', 'p=1']

            counts_bin['intensity'] = 255 * (counts_bin['p=1'] - counts_bin['p=0'])

            frame = np.zeros(self.sensor_size)
            for (px, py), row in counts_bin.iterrows():
                if 0 <= px < self.sensor_size[1] and 0 <= py < self.sensor_size[0]:
                    frame[py, px] = row['intensity']

            video_filename = f"../{self.event_file_name}_{integration_time}_{start}_{end}.mp4"
            # for bounding box around the events
            if apply_bounding_box:
                video_filename = f"../bb_{self.event_file_name}_{integration_time}_{start}_{end}.mp4"
                downsample_factor = 4
                frame_size = self.sensor_size
                x_vals, y_vals = df_bin["x"].values, df_bin["y"].values
                x_vals = x_vals + np.random.normal(0, 1e-3, size=x_vals.shape)
                y_vals = y_vals + np.random.normal(0, 1e-3, size=y_vals.shape)
                # skip if too few events
                if len(x_vals) < 2:
                    continue

                small_grid_x, small_grid_y = np.meshgrid(
                    np.linspace(0, frame_size[1], frame_size[1] // downsample_factor),
                    np.linspace(0, frame_size[0], frame_size[0] // downsample_factor)
                )

                kde = scipy.stats.gaussian_kde(np.vstack([x_vals, y_vals]))
                density = kde(np.vstack([small_grid_x.ravel(), small_grid_y.ravel()]))
                density = density.reshape(small_grid_x.shape)

                # normalize the density
                density = (density - density.min()) / (density.max() - density.min())

                # gaussian smoothing
                density_resized = cv2.resize(density, (frame_size[1], frame_size[0]), interpolation=cv2.INTER_CUBIC)
                blurred_density = cv2.GaussianBlur(density_resized, (5, 5), 0)

                # adaptive thresholding for better detection
                _, binary_mask = cv2.threshold(blurred_density, 0.3, 1, cv2.THRESH_BINARY)

                # uint8 for open cv compatibility
                binary_mask = (binary_mask * 255).astype(np.uint8)

                # apply contours to find bounding boxes
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # to exclude small variations and noise. Try tweaking this parameter
                    if w > 5 and h > 5:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            frames.append(frame)

        # for plotting time bins - for only less-number of time bins
        if end <= 10:
            fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
            if len(frames) == 1:
                axes = [axes]

            for ax, (frame, t_bin) in zip(axes, zip(frames, time_bins[start:end])):
                # TODO: try with frame and frame.T
                ax.imshow(frame, cmap='gray', origin='upper', interpolation='nearest')
                ax.set_title(f'Time Bin {t_bin}')
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        # for saving output
        fps = 30
        frame_size = (self.sensor_size[1], self.sensor_size[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size, isColor=True)

        for frame in frames:
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # frame_colored = cv2.applyColorMap(frame_normalized, cv2.COLORMAP_PLASMA)
            frame_colored = cv2.applyColorMap(frame_normalized, cv2.COLORMAP_JET)
            video_writer.write(frame_colored)

        video_writer.release()
        print(f"Video saved as {video_filename}")

