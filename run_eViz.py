import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loadFile import load_event_data_file
from eVizTool import EVizTool

if __name__ == "__main__":
    # file_path = "../../DATA/EBSSA_DATA/ISS_2_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/SL8RB_1957_davis_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/20170214-20-58_22285_SL-16RB_labelled.mat"
    # file_path = "../../DATA/Test_Data/SL14RB_leap_1933_davis_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/SL8RB_1957_atis_td_labelled.mat"
    # file_path = "../../DATA/Test_Data/square_linear.mat"
    # file_path = "../../DATA/Test_Data/A7P18C1-2021_11_06_17_07_31.mat"
    # file_path = "../../DATA/Test_Data/test_complex.mat"

    # file_path = "../../DATA/Test_Data/moving_source.h5"
    #file_path = "C:\Zoove\object_detection_exploration\square_wave\Event_Square\square_1_circle_25_2.h5"
    #file_path = "C:\Zoove\object_detection_exploration\Test_data\A0P15C1-2021_11_06_09_59_24.npy"
    #file_path = "C:\Zoove\object_detection_exploration\Test_data\slider_depth_events\events_chunk.txt"
    #file_path = "C:\Zoove\object_detection_exploration\Test_data\prophesee_events_mini\GOPR9647_ts.npy"
    file_path = "C:\Zoove\object_detection_exploration\Test_data\Real_Flash_On_Off.h5"

    # file_path = "../../DATA/Test_Data/A0P16C2-2021_11_06_14_00_03.npy"
    # file_path = "../../DATA/Test_Data/A0P12C0-2021_11_05_14_02_50.npy"
    # file_path = "../../DATA/EB_Action_Recognition/THU-EACT-50-CHL/A48P15C0-2021_11_06_10_22_44.npy"
    # file_path = "../../DATA/EB_Action_Recognition/THU-EACT-50-CHL/MAT_Files/A9P13C0-2021_11_05_15_13_49.mat"

    # file_path = "../../DATA/Test_Data/square_1_linear_50_2.h5.csv"
    event_df = pd.DataFrame()
    event_df = load_event_data_file(file_path)
    file_name = os.path.basename(file_path)
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
        eviz_obj.plot_histograms()
        eviz_obj.plot_temporal_kernel_density()
        eviz_obj.plot_event_on_off_map()
        eviz_obj.plot_polarity_count_at_given_pixel()
        eviz_obj.plot_event_intensity_map()
        eviz_obj.plot_full_heatmap()
        eviz_obj.plot_event_batches(num_batches= 16)
        eviz_obj.plot_event_intensity_map()
        #eviz_obj.plot_3d_voxel_grid()
        eviz_obj.plot_event_optical_flow()
        #eviz_obj.generate_time_lapse_heatmap_gif(bin_width=1000)
        #eviz_obj.plot_3d_scatter()

        #mp4_path, avi_path = eviz_obj.event_to_video1(integration_time=5000, start=0, end=10, apply_bounding_box=True)
        #plt.show()
