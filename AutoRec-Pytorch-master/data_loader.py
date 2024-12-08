import os
import sys
import numpy as np

path_prefix = './ml-1M/'

def load_data(dataset='ratings', train_ratio=0.9):
    """
    Load and preprocess dataset from a specified file.

    Args:
        dataset (str): Name of the dataset file (without extension).
        train_ratio (float): Proportion of data to use for training (default is 0.9).

    Returns:
        tuple: Contains the following:
            - train_list (list): List of training records (user, item, rating).
            - test_list (list): List of testing records (user, item, rating).
            - max_uid (int): Maximum user ID in the dataset.
            - max_vid (int): Maximum item ID in the dataset.
    """
    fname = path_prefix + dataset + '.dat'  # Construct the full file path.
    max_uid = 0  # Initialize maximum user ID.
    max_vid = 0  # Initialize maximum item ID.
    records = []  # List to store all records.

    # Check if the dataset file exists.
    if not os.path.exists(fname):
        print('[Error] File %s not found!' % fname)
        sys.exit(-1)

    first_line_flag = True  # Flag to handle the first line separately.

    # Open the file for reading.
    with open(fname, encoding="ISO-8859-1") as f:
        for line in f:
            # Split each line into components based on '::'.
            tks = line.strip().split('::')
            if first_line_flag:
                # Initialize max_uid and max_vid using the first line's data.
                max_uid = int(tks[0])
                max_vid = int(tks[1])
                first_line_flag = False
                continue

            # Update max_uid and max_vid with the current line's data.
            max_uid = max(max_uid, int(tks[0]))
            max_vid = max(max_vid, int(tks[1]))

            # Append the user, item, and rating to the records list.
            records.append((int(tks[0]) - 1, int(tks[1]) - 1, int(tks[2])))

    # Print dataset statistics.
    print("Max user ID {0}. Max item ID {1}. In total {2} ratings.".format(
        max_uid, max_vid, len(records)))

    # Shuffle the records randomly.
    np.random.shuffle(records)

    # Split the data into training and testing sets based on the train_ratio.
    train_list = records[0:int(len(records) * train_ratio)]
    test_list = records[int(len(records) * train_ratio):]

    return train_list, test_list, max_uid, max_vid
