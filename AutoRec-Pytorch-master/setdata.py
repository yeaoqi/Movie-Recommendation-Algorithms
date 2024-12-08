import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, rating_list, n_user, n_item, user_based=True):
        """
        Initialize the dataset.

        Args:
            rating_list (list of tuples): List of (user, item, rating) tuples.
            n_user (int): Total number of users.
            n_item (int): Total number of items.
            user_based (bool): Whether the dataset is user-centric. If False, it will be item-centric.
        """
        self.data = rating_list  # Store the list of ratings.
        self.user_based = user_based  # Flag to determine user-based or item-based configuration.
        self.n_user = n_user  # Number of users.
        self.n_item = n_item  # Number of items.

        # Initialize the rating matrix with zeros.
        self.x_mat = np.ones((n_user, n_item)) * 0

        # Initialize the mask matrix to indicate observed ratings.
        self.mask = np.zeros((n_user, n_item))

        # Populate the rating matrix and mask matrix using the provided ratings.
        for u, v, r in self.data:
            self.x_mat[u][v] = r  # Set the rating value.
            self.mask[u][v] = 1  # Mark the rating as observed.

        # Convert the matrices to PyTorch tensors.
        self.x_mat = torch.from_numpy(self.x_mat).float()
        self.mask = torch.from_numpy(self.mask).float()

        # If not user-based, transpose the matrices for item-based configuration.
        if not self.user_based:
            self.x_mat = self.x_mat.t()
            self.mask = self.mask.t()

    def __getitem__(self, index):
        """
        Get a specific row of the rating and mask matrices.
        """
        return self.x_mat[index], self.mask[index]

    def __len__(self):
        """
        Get the number of rows in the dataset.
        """
        if self.user_based:
            return self.n_user
        return self.n_item

    def get_mat(self):
        """
        Get the full rating matrix and mask, along with the user-based flag.
        """
        return self.x_mat, self.mask, self.user_based



