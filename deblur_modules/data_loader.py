import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2 as cv
import os


class GoProDataLoader(Dataset):
    """
        Loading the dataset for training
    """

    def __init__(self, root_path, dataset_path):
        """
        Initializing the dataset arguments.
        
        Args:
            root_path: Path of the dataset directory.
            dataset_path: Path of the dataset list.

        Example:
        train_dataset = GoProDataLoader(root_path="gopro", dataset_path="gopro.train.csv")
        """
        
        super(GoProDataLoader, self).__init__()
        self.root_path = root_path
        self.dataset = pd.read_csv(dataset_path)

    def __getitem__(self, index):
        # select
        selected_image = self.dataset.iloc[index]

        # path
        sharp_image_path = os.path.join(self.root_path, selected_image["Folder_name"] + "/sharp/" + selected_image["File_name"])
        blur_image_path = os.path.join(self.root_path, selected_image["Folder_name"] + "/blur/" + selected_image["File_name"])

        # Read an image and change the channel axis to be in the first position instead of last: 
        # (Width,Height, channel) => (channel, Width, Height)
        sharp_image = np.moveaxis(cv.imread(sharp_image_path), -1, 0)
        blur_image = np.moveaxis(cv.imread(blur_image_path), -1, 0)

        # Covert it to torch tensor
        sharp_image = torch.from_numpy(sharp_image)
        blur_image = torch.from_numpy(blur_image)

        return sharp_image, blur_image

    def __len__(self):
        # return length of the dataset (Number of samples)
        return len(self.dataset)