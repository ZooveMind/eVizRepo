import cv2
import scipy.io
import scipy.signal
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import pandas as pd
import imageio
from matplotlib import _cm
from io import BytesIO
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

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

    def save_figure(self, folder_name, file_suffix):
        output_dir = os.path.join("visualization_output", folder_name)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.event_file_name))[0]
        output_path = os.path.join(output_dir, f"{base_name}_{file_suffix}.png")
        plt.savefig(output_path)
        print(f"Saved: {output_path}")



    def save_visual_output(self, folder_name, file_suffix, content, ext="png"):
        """
        Save a visual output (e.g., figure, gif, or future video).

        :param folder_name: Name of the subfolder inside visualization_output.
        :param file_suffix: Descriptive suffix for the filename.
        :param content: 'plt.Figure' for PNG or list of np.ndarray/PIL.Image for GIF.
        :param ext: 'png' or 'gif'
        """
        output_dir = os.path.join("visualization_output", folder_name)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.event_file_name))[0]
        output_path = os.path.join(output_dir, f"{base_name}_{file_suffix}.{ext}")

        try:
            if ext == "png":
                if isinstance(content, plt.Figure):
                    content.savefig(output_path)
                    plt.close(content)
                else:
                    raise TypeError("For PNG, 'content' must be a matplotlib Figure object.")

            elif ext == "gif":
                if isinstance(content, list) and len(content) > 0:
                    if all(isinstance(frame, (np.ndarray, Image.Image)) for frame in content):
                        imageio.mimsave(output_path, content, fps=10)
                    else:
                        raise TypeError("All frames in GIF content must be numpy arrays or PIL Images.")
                else:
                    raise ValueError("GIF content must be a non-empty list of frames.")

            else:
                raise ValueError(f"Unsupported extension: {ext}")

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error while saving visual output: {e}")



    def bin_events(self, bin_width=10000):
        """
        Bins event data from self.t using the specified bin width.
        """
        min_time, max_time = self.t.min(), self.t.max()
        bin_edges = np.arange(min_time, max_time, bin_width)

        # Assign each event to a bin based on timestamps
        bin_indices = np.digitize(self.t, bin_edges)

        # Group full event tuples (t, x, y, p) into bins
        full_events = np.stack((self.t, self.x, self.y, self.p), axis=1)
        binned_events = [full_events[bin_indices == i] for i in range(1, len(bin_edges))]

        return binned_events

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

    @staticmethod
    def bin_events_static(events, bin_width=10000):
        """
        Static method to bin events based on timestamp.

        :param events: np.ndarray of shape (N, 4) -> (t, x, y, p)
        :param bin_width: Width of time bin in microseconds
        :return: List of np.ndarrays, each a bin of events
        """
        if len(events) == 0:
            return []

        min_time, max_time = events[:, 0].min(), events[:, 0].max()
        bin_edges = np.arange(min_time, max_time + bin_width, bin_width)

        bin_indices = np.digitize(events[:, 0], bin_edges)
        binned_events = [events[bin_indices == i] for i in range(1, len(bin_edges))]

        return binned_events

    ##################################################################################################
    ####################### Visualization plots starts from here#####################################

    #1. Histogram visulaization

    def plot_event_histogram(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.p, bins=3, edgecolor="black", alpha=0.7)
        plt.title(f'Event Polarity Histogram - {self.event_file_name}')
        plt.ylabel("Count")
        # plt.xticks([1, -1], labels=["ON (1)", "OFF (-1)"])
        self.save_figure("histogram_visualization", "event_histogram")
        plt.close()
    
    def plot_histograms(self, bin_width=10000):
        """
        Plots both raw timestamp histogram and binned event histogram.
        """
        binned_events = self.bin_events(bin_width=bin_width)

        plt.figure(figsize=(12, 5))

        # Raw timestamp histogram
        plt.subplot(1, 2, 1)
        plt.hist(self.t, bins=50, color='blue', alpha=0.7)
        plt.xlabel("Timestamp")
        plt.ylabel("Event Count")
        plt.title("Raw Event Timestamp Distribution")

        # Binned event histogram
        plt.subplot(1, 2, 2)
        bin_counts = [len(bin_) for bin_ in binned_events]
        plt.bar(range(len(bin_counts)), bin_counts, color='green', alpha=0.7)
        plt.xlabel("Bin Index")
        plt.ylabel("Event Count per Bin")
        plt.title(f"Binned Events (Bin Width = {bin_width})")

        plt.suptitle("Binned Time Event Visualization", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        self.save_figure("plot_histograms", "binned_time_event_visualization")
        plt.close()

    # 3-d histogram:
    def plot_3d_voxel_grid(self, time_bins=10):
        """
        Generate a 3D voxel grid of event activity across X, Y, and time,
        and save the resulting figure.
        """
        x, y, t = self.x, self.y, self.t

        x_max, y_max = x.max() + 1, y.max() + 1
        t_min, t_max = t.min(), t.max()

        voxel_grid = np.zeros((x_max, y_max, time_bins), dtype=np.int32)

        # Bin the time dimension
        t_bins = np.linspace(t_min, t_max, time_bins + 1)

        for i in range(time_bins):
            mask = (t >= t_bins[i]) & (t < t_bins[i + 1])
            np.add.at(voxel_grid, (x[mask], y[mask], i), 1)

        filled = voxel_grid > 0  # Where to draw voxels

        # Normalize for color mapping
        norm = voxel_grid / voxel_grid.max()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Generate colors based on density
        cmap = cm.viridis
        colors = cmap(norm)

        ax.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Time Bin')
        ax.set_title('3D Voxel Grid of Event Data')

        # Add colorbar
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array(voxel_grid)
        fig.colorbar(mappable, ax=ax, label='Event Density')

        self.save_figure("generate_3d_voxel_grid", "events_chunk_voxel_grid_3d")
        plt.close()

    # optical flow of events gradient:
    def plot_event_optical_flow(self, batch_size=50000):
        """
        Estimate and visualize event-based optical flow using gradient method.
        Saves the figure in the output folder.
        """
        x, y, t = self.x, self.y, self.t
        x_max, y_max = x.max() + 1, y.max() + 1
        flow_map_x = np.zeros((y_max, x_max), dtype=np.float32)
        flow_map_y = np.zeros((y_max, x_max), dtype=np.float32)
        count_map = np.zeros((y_max, x_max), dtype=np.float32)

        # Process events in batches
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            t_batch = t[i:i + batch_size]

            heat = np.zeros((y_max, x_max), dtype=np.float32)
            np.add.at(heat, (y_batch, x_batch), t_batch)  # Accumulate timestamp
            mask = (heat > 0)

            grad_y, grad_x = np.gradient(heat)

            flow_map_x += grad_x
            flow_map_y += grad_y
            count_map += mask.astype(np.float32)

        count_map[count_map == 0] = 1  # Avoid divide by zero
        flow_map_x /= count_map
        flow_map_y /= count_map

        # Plot flow
        plt.figure(figsize=(8, 6))
        plt.imshow(count_map, cmap='gray', origin='upper')
        step = max(x_max // 30, 1)  # To avoid too dense arrows
        plt.quiver(
            np.arange(0, x_max, step),
            np.arange(0, y_max, step),
            flow_map_x[::step, ::step],
            flow_map_y[::step, ::step],
            color='red', angles='xy', scale_units='xy', scale=1.5
        )

        plt.title("Event-based Optical Flow (Gradient Method)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        self.save_figure("optical_flow", "event_optical_flow")
        plt.close()

    # 2. Full heatmap visualisation:

    def plot_full_heatmap(self, batch_size=50000):
        """
        Generate a full heatmap of event activity across all timestamps in batches
        and save the resulting figure.
        """
        x, y = self.x, self.y
        x_max, y_max = x.max() + 1, y.max() + 1
        heatmap = np.zeros((y_max, x_max), dtype=np.int32)

        # Process in batches
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            np.add.at(heatmap, (y_batch, x_batch), 1)

        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest', origin='upper')
        plt.colorbar(label="Event Density")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Full Heatmap of Event Data")

        self.save_figure("generate_full_heatmap", "event_density_map")
        plt.close()

    

    #4. batch wise heatmap of event flow:
    def plot_event_batches(self, num_batches=8, x_bins=360, y_bins=260):
        """
        Plot batch-wise 2D heatmaps of event accumulation across the entire event stream.
        
        :param num_batches: Number of batches to divide the full event stream
        :param x_bins: Number of bins for X axis in the histogram
        :param y_bins: Number of bins for Y axis in the histogram
        """
        timestamps, x_positions, y_positions = self.t, self.x, self.y
        num_events = len(timestamps)
        batch_size = num_events // num_batches

        fig = plt.figure(figsize=(20, num_batches // 4 * 4))

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size if i < num_batches - 1 else num_events

            x_batch = x_positions[start:end]
            y_batch = y_positions[start:end]

            heatmap, _, _ = np.histogram2d(x_batch, y_batch, bins=(x_bins, y_bins))

            ax = fig.add_subplot(num_batches // 4, 4, i + 1)
            im = ax.imshow(heatmap.T, origin='upper', cmap='hot', interpolation='nearest')
            fig.colorbar(im, ax=ax, label="Event Density")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_title(f"Batch {i+1}: Events {start} to {end}")

        plt.tight_layout()
        self.save_figure("plot_event_batches", "batch_heatmaps")
        plt.close()

    # heatmap gif:
    def generate_time_lapse_heatmap_gif(self, bin_width=10000, x_bins=360, y_bins=260):
        try:
            print(f"Generating time-lapse heatmap GIF with bin width: {bin_width}µs")

            # Step 1: Create event array
            events_array = np.column_stack((self.t, self.x, self.y, self.p))

            # Step 2: Bin events by timestamp
            min_time = self.t.min()
            max_time = self.t.max()
            bin_edges = np.arange(min_time, max_time + bin_width, bin_width)

            if len(bin_edges) <= 2:
                print("❗ Bin width too large, only one bin generated.")
                return

            frames = []
            for i in range(len(bin_edges) - 1):
                t_start = bin_edges[i]
                t_end = bin_edges[i + 1]

                bin_mask = (self.t >= t_start) & (self.t < t_end)
                x_bin = self.x[bin_mask]
                y_bin = self.y[bin_mask]

                if len(x_bin) == 0:
                    continue  # Skip empty frames

                heatmap, _, _ = np.histogram2d(x_bin, y_bin, bins=(x_bins, y_bins))

                fig, ax = plt.subplots()
                im = ax.imshow(
                    heatmap.T, origin="lower", cmap="hot", interpolation="nearest"
                )
                plt.colorbar(im, ax=ax, label="Event Count")
                ax.set_title(f"Time: {t_start/1e6:.2f}s - {t_end/1e6:.2f}s")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)

                plt.close(fig)

            if not frames:
                print("⚠️ No frames generated — check event data or bin width.")
                return

            self.save_visual_output("time_lapse", "heatmap", frames, ext="gif")
            print("✅ Time-lapse heatmap GIF saved successfully.")

        except Exception as e:
            print(f"❌ Error generating GIF: {e}")

    # . 3-d scatter plot:
    def plot_3d_scatter(self):
        """
        3D scatter plot of event data using timestamps, x, y values.
        """
        t, x, y = self.t, self.x, self.y

        # 3D scatter plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(x, y, t, c=t, cmap="plasma", alpha=0.5, s=1)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Timestamps")
        ax.set_title("3D Scatter Plot of Event Data")

        plt.colorbar(scatter)

        self.save_figure("plot_3d_scatter", "3d_event_plot")
        plt.close()

    #. plot temporal kernel desnity
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
        self.save_figure("temporal_kernel_density", "temporal_density")
        plt.close()

    def plot_spatial_kernel_density(self):
        pass

    #. on off event:
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
        self.save_figure("on_off_map", "on_off_density")
        plt.close()

    #  
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
        self.save_figure("polarity_count_pixel", "polarity_count")
        plt.close()

    #. event desnity map
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
        #plt.show()
        self.save_figure("intensity_map", "event_intensity")
        plt.close()

    # . video to event with some bounding box approach
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

    
    def event_to_video1(self, integration_time, start, end, apply_bounding_box):
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

            if apply_bounding_box:
                downsample_factor = 4
                frame_size = self.sensor_size
                x_vals, y_vals = df_bin["x"].values, df_bin["y"].values
                x_vals = x_vals + np.random.normal(0, 1e-3, size=x_vals.shape)
                y_vals = y_vals + np.random.normal(0, 1e-3, size=y_vals.shape)

                if len(x_vals) < 2:
                    continue

                small_grid_x, small_grid_y = np.meshgrid(
                    np.linspace(0, frame_size[1], frame_size[1] // downsample_factor),
                    np.linspace(0, frame_size[0], frame_size[0] // downsample_factor)
                )

                kde = scipy.stats.gaussian_kde(np.vstack([x_vals, y_vals]))
                density = kde(np.vstack([small_grid_x.ravel(), small_grid_y.ravel()]))
                density = density.reshape(small_grid_x.shape)
                density = (density - density.min()) / (density.max() - density.min())

                density_resized = cv2.resize(density, (frame_size[1], frame_size[0]), interpolation=cv2.INTER_CUBIC)
                blurred_density = cv2.GaussianBlur(density_resized, (5, 5), 0)
                _, binary_mask = cv2.threshold(blurred_density, 0.3, 1, cv2.THRESH_BINARY)
                binary_mask = (binary_mask * 255).astype(np.uint8)

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 5 and h > 5:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            frames.append(frame)

        # Optional plot of individual time bins
        if end <= 10:
            fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
            if len(frames) == 1:
                axes = [axes]

            for ax, (frame, t_bin) in zip(axes, zip(frames, time_bins[start:end])):
                ax.imshow(frame, cmap='gray', origin='upper', interpolation='nearest')
                ax.set_title(f'Time Bin {t_bin}')
                ax.axis('off')
            plt.tight_layout()
            plt.show()

        # Output setup
        fps = 30
        frame_size = (self.sensor_size[1], self.sensor_size[0])
        output_dir = "visualization_output/event_to_video"
        os.makedirs(output_dir, exist_ok=True)

        base_name = f"{'bb_' if apply_bounding_box else ''}{self.event_file_name}_{integration_time}_{start}_{end}"
        mp4_path = os.path.join(output_dir, f"{base_name}.mp4")
        avi_path = os.path.join(output_dir, f"{base_name}.avi")

        # Save MP4
        mp4_writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size, isColor=True)
        # Save AVI
        avi_writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size, isColor=True)

        for frame in frames:
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame_colored = cv2.applyColorMap(frame_normalized, cv2.COLORMAP_JET)
            mp4_writer.write(frame_colored)
            avi_writer.write(frame_colored)

        mp4_writer.release()
        avi_writer.release()

        print(f"MP4 video saved at: {mp4_path}")
        print(f"AVI video saved at: {avi_path}")

        return mp4_path, avi_path


