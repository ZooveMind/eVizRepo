
import os
import h5py             # for .h5
import scipy.io         # for .mat
import pandas as pd     # for .csv
import numpy as np      # for .npy, .txt

# start with this empty array and based on file extension the potential data for respective fields would be extracted
timestamp_array = np.array([])
x_y_counter = 0
x_array = np.array([])
y_array = np.array([])
polarity_array = np.array([])

def is_valid_timestamp_array(timestamps):
    # TODO: find a better way to filter and decide timestamp array
    timestamps = np.array(timestamps)
    if len(str(int(max(timestamps)))) > 4:
        return True
    else:
        return False

def is_valid_x_y_array(x_y_array):
    # assuming a sensor dimension is maximum till 4 digits (512, 512), (1024, 1024) etc
    if 1 < len(str(int(max(x_y_array)))) <= 4:
        return True
    else:
        return False

def is_valid_polarity_array(arr):
    valid_combinations = [{0, 1}, {1, -1}, {0}, {1}, {-1}]
    return set(arr) in valid_combinations

def load_event_data_file(file_path):
    file_name = os.path.basename(file_path)

    _, extension = os.path.splitext(file_name)

    global timestamp_array, x_y_counter, x_array, y_array, polarity_array
    timestamp_array = np.array([])
    x_y_counter = 0
    x_array = np.array([])
    y_array = np.array([])
    polarity_array = np.array([])

    if extension == ".mat":
        # mat_data will always be a dictionary
        # '__header__', '__version__', '__globals__' are the common keys and can be skipped, find other keys if present
        mat_data = scipy.io.loadmat(file_path)
        filtered_mat_data = {key: value for key, value in mat_data.items() if not key.startswith("__")}

        def find_valid_1d_array(data, key):
            global timestamp_array, x_y_counter, x_array, y_array, polarity_array
            # TODO: make this function only generic for all kind of data and find the valid 1d arrays using this
            # TODO: handle 'Obj' properly in later version
            # this is for EBSSA data not to process 'Obj' key
            if isinstance(data, np.ndarray) and key != 'Obj':
                n_dim = data.ndim
                while n_dim > 1:
                    data = data.flatten()
                    n_dim = data.ndim

                if n_dim == 1 and len(data) == 1:
                    if len(data) == 1:
                        data = data[0]
                        if isinstance(data, np.void) and len(data.shape) == 0 and len(data) != 0:
                            for i in range(len(data)):
                                if data[i].ndim != 1:
                                    find_valid_1d_array(data[i], key)
                        else:
                            if isinstance(data, np.ndarray) and len(data) == data.shape[0]:
                                pass

                if data.ndim == 1 and 1 < len(data) == data.shape[0]:
                    # print(f"    Found a valid 1D array: {data.shape} at {key}")
                    if is_valid_timestamp_array(data):
                        # print('     -labelling as timestamp array')
                        timestamp_array = data
                    elif is_valid_x_y_array(data):
                        # print('     -labelling as x or y array')
                        x_y_counter += 1
                        if x_y_counter - 1 == 0:
                            x_array = data
                        if x_y_counter - 1 == 1:
                            y_array = data

                    elif is_valid_polarity_array(data):
                        # print('     - labelling as polarity array')
                        polarity_array = data

        # iterate through all keys in the .mat file
        for key, value in filtered_mat_data.items():
            # print(f"\nSearching in key: {key}")
            find_valid_1d_array(value, key)

    if extension == '.h5':
        with h5py.File(file_path, 'r') as file:
            def get_structure(name, obj):
                return name

            key = file.visititems(get_structure)
            h5_data = file[key][:]
            rows, cols = h5_data.shape
            if cols == 4 and isinstance(h5_data, np.ndarray):
                h5_data_T = h5_data.T
                for data in h5_data_T:
                    # print(f"    Found a valid 1D array: {data.shape}")
                    if is_valid_timestamp_array(data):
                        # print('     -labelling as timestamp array')
                        timestamp_array = data
                    elif is_valid_x_y_array(data):
                        # print('     -labelling as x or y array')
                        x_y_counter += 1
                        if x_y_counter - 1 == 0:
                            x_array = data
                        if x_y_counter - 1 == 1:
                            y_array = data

                    elif is_valid_polarity_array(data):
                        # print('     - labelling as polarity array')
                        polarity_array = data

    if extension == '.txt':
        data = np.loadtxt(file_path)

        if data.shape[1] < 4:
            raise ValueError("Expected 4 columns in .txt file: timestamp, x, y, polarity")

        timestamp_array = data[:, 0]
        x_array = data[:, 1]   
        y_array = data[:, 2]   
        polarity_array = data[:, 3].astype(int)  

    if extension == '.npy':
        try:
            npy_data = np.load(file_path)
            rows, cols = npy_data.shape
            if cols == 4 and isinstance(npy_data, np.ndarray):
                npy_data_T = npy_data.T
                for data in npy_data_T:
                    # print(f"    Found a valid 1D array: {data.shape}")
                    if is_valid_timestamp_array(data):
                        # print('     -labelling as timestamp array')
                        timestamp_array = data
                    elif is_valid_x_y_array(data):
                        # print('     -labelling as x or y array')
                        x_y_counter += 1
                        if x_y_counter - 1 == 0:
                            x_array = data
                        if x_y_counter - 1 == 1:
                            y_array = data

                    elif is_valid_polarity_array(data):
                        # print('     - labelling as polarity array')
                        polarity_array = data
            else:
                raise ValueError("Expected 4 columns in .npy file: timestamp, x, y,p")
        except Exception as e:
            print(f"Error loading .npy file: {e}")
            
    if extension == '.csv':
        input_df = pd.read_csv(file_path).values
        rows, cols = input_df.shape
        if cols == 4 and isinstance(input_df, np.ndarray):
            input_df_T = input_df.T
            for data in input_df_T:
                # print(f"    Found a valid 1D array: {data.shape}")
                if is_valid_timestamp_array(data):
                    # print('     -labelling as timestamp array')
                    timestamp_array = data
                elif is_valid_x_y_array(data):
                    # print('     -labelling as x or y array')
                    x_y_counter += 1
                    if x_y_counter - 1 == 0:
                        x_array = data
                    if x_y_counter - 1 == 1:
                        y_array = data

                elif is_valid_polarity_array(data):
                    # print('     - labelling as polarity array')
                    polarity_array = data

    if timestamp_array.size != 0:
        pass
        # print("\nFound timestamp array")
        # print(timestamp_array)

    if x_array.size != 0:
        pass
        # print("\nFound x array")
        # print(x_array)

    if y_array.size != 0:
        pass
        # print("\nFound y array")
        # print(y_array)

    # swap x_array and y_array if x_array maximum is less - assumption is, if not equal sensor size is more along x
    if x_array.size != 0 and y_array.size != 0:
        if np.max(y_array) > np.max(x_array):
            # print("Maximum of x array is smaller than maximum of y array")
            # print("Swapping x and y!")
            temp = x_array
            x_array = y_array
            y_array = temp
        else:
            pass
            # print("Maximum of x array is larger than maximum of y array or equal to maximum of y array")
            # print("No swapping is needed!")

    if polarity_array.size != 0:
        pass
        # print("\nFound polarity array")
        # print(polarity_array)

    if timestamp_array.size != 0 and x_array.size != 0 and y_array.size != 0 and polarity_array.size != 0:
        # print(f"\nFound all possible arrays for the input file - {file_name}. "
        #       f"\nMaking final data frame for passing to eVizTool")

        # /1e6 for us time resolution
        event_df = pd.DataFrame({
            't': timestamp_array/1e6,
            'x': x_array.astype(int),
            'y': y_array.astype(int),
            'p': polarity_array.astype(int)
        })
        # print(event_df)

        return event_df

    else:
        pd.DataFrame()