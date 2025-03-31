import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loadFile import load_event_data_file
from eVizTool import EVizTool

if __name__ == "__main__":
    filenames = []
    event_start_times = []
    event_end_times = []
    no_of_events = []
    max_on_peak_times = []
    max_on_peak_densities = []
    min_on_peak_times = []
    min_on_peak_densities = []
    no_of_on_peaks = []
    max_off_peak_times = []
    max_off_peak_densities = []
    min_off_peak_times = []
    min_off_peak_densities = []
    no_of_off_peaks = []

    # file_path = "../../DATA/EBSSA_DATA/ISS_2_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/SL8RB_1957_davis_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/20170214-20-58_22285_SL-16RB_labelled.mat"
    # file_path = "../../DATA/Test_Data/SL14RB_leap_1933_davis_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/SL8RB_1957_atis_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/square_linear.mat"
    # file_path = "../../DATA/Test_Data/A7P18C1-2021_11_06_17_07_31.mat"
    # file_path = "../../DATA/Test_Data/test_complex.mat"

    # file_path = "../../DATA/Test_Data/moving_source.h5"
    # file_path = "../../DATA/Test_Data/blink_1024_768.h5"

    # file_path = "../../DATA/Test_Data/A0P16C2-2021_11_06_14_00_03.npy"
    # file_path = "../../DATA/Test_Data/A0P12C0-2021_11_05_14_02_50.npy"
    # file_path = "../../DATA/EB_Action_Recognition/THU-EACT-50-CHL/A48P15C0-2021_11_06_10_22_44.npy"
    # file_path = "../../DATA/EB_Action_Recognition/THU-EACT-50-CHL/MAT_Files/A9P13C0-2021_11_05_15_13_49.mat"

    # file_path = "../../DATA/Test_Data/square_1_linear_50_2.h5.csv"

    # folder_path = "../../DATA/EB_Action_Recognition/THU-EACT-50-CHL/test"
    folder_path = "../../DATA/EB_Action_Recognition/THU-EACT-50-CHL"
    # folder_path = "../../DATA/EBSSA_DATA/ISS"
    event_df = pd.DataFrame()
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        file_name = os.path.basename(file_path)
        event_df = load_event_data_file(file_path)

        if event_df is None:
            print("No valid event data was found in the file!")
            print(f"Skipped file - {file_name}")
        elif event_df.empty:
            print("No valid event data was found in the file!")
            print(f"Skipped file - {file_name}")
        else:
            sensor_dim = (int(max(event_df["x"].values) + 1), int(max(event_df["y"].values) + 1))
            # print("Valid event data was found. Analyzing plots...")
            eviz_obj = EVizTool(file_name, event_df, sensor_dim)
            # eviz_obj.plot_event_histogram()
            # eviz_obj.plot_temporal_kernel_density()
            _, on_peak_time, on_peak_density, off_peak_time, off_peak_density = eviz_obj.estimate_and_plot_temporal_kernel_density()
            # eviz_obj.plot_event_on_off_map()
            # eviz_obj.plot_polarity_count_at_given_pixel()
            # eviz_obj.plot_event_intensity_map()
            # eviz_obj.event_to_video(integration_time=0.01, start=0, end=1000, apply_bounding_box=True)
            # eviz_obj.event_to_video(integration_time=0.1/2, start=0, end=1000, apply_bounding_box=True)

            # plt.show()

            filenames.append(filename)
            event_start_times.append(event_df["t"].values[0]*1e6)
            event_end_times.append(event_df["t"].values[-1]*1e6)
            no_of_events.append(len(event_df["t"].values)*1e6)
            max_on_density_loc = np.argmax(on_peak_density)
            max_on_peak_times.append(on_peak_time[max_on_density_loc]*1e6)
            max_on_peak_densities.append(np.max(on_peak_density))
            min_on_density_loc = np.argmin(on_peak_density)
            min_on_peak_times.append(on_peak_time[min_on_density_loc]*1e6)
            min_on_peak_densities.append(np.min(on_peak_density))
            no_of_on_peaks.append(len(on_peak_density))

            max_off_density_loc = np.argmax(off_peak_density)
            max_off_peak_times.append(off_peak_time[max_off_density_loc])
            max_off_peak_densities.append(np.max(off_peak_density))
            min_off_density_loc = np.argmin(off_peak_density)
            min_off_peak_times.append(off_peak_time[min_off_density_loc])
            min_off_peak_densities.append(np.min(off_peak_density))
            no_of_off_peaks.append(len(off_peak_density))

        df = pd.DataFrame({'file_name ': filenames, 'event_start_t': event_start_times, 'event_end_t': event_end_times,
                           'no_of_events': no_of_events, 'max_on_peak': max_on_peak_densities, 'max_on_peak_t': max_on_peak_times,
                           'min_on_peak': min_on_peak_densities, "min_on_peak_t": min_on_peak_times, 'no_of_on_peaks': no_of_on_peaks,
                           'max_off_peak': max_off_peak_densities, 'max_off_peak_t': max_off_peak_times,
                           'min_off_peak': min_off_peak_densities, 'min_off_peak_t': min_off_peak_times,
                           'no_of_off_peaks': no_of_off_peaks})
        df.to_csv('../analytics.csv', index=False)