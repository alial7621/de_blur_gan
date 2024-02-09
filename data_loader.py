import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2 as cv
import os


class DataLoader(Dataset):
    """
        Loading the dataset for training
    """

    def __init__(self, root_path, dataset_path, image_size=(256, 256)):
        """
        Initializing the dataset arguments.
        
        Args:
            root_path: Path of the dataset directory.
            dataset_path: Path of the dataset list.
            image_size: Size (Resolution) of the input image to the models. default value: 256x256

        Example:
        train_dataset = DataLoader(root_path="gopro", dataset_path="gopro.train.csv", image_size=(256, 256))
        """
        
        super(DataLoader, self).__init__()
        self.root_path = root_path
        self.dataset = pd.read_csv(dataset_path)
        self.image_size = image_size

    def __getitem__(self, index):
        # select
        selected_image = self.dataset.iloc[index]

        # path
        sharp_image_path = os.path.join(self.root_path, selected_image["Folder_name"] + "/sharp/" + selected_image["File_name"])
        blur_image_path = os.path.join(self.root_path, selected_image["Folder_name"] + "/blur/" + selected_image["File_name"])
        
        # Read and resize
        sharp_image = cv.resize(cv.imread(sharp_image_path), self.image_size)
        blur_image = cv.resize(cv.imread(blur_image_path), self.image_size)

        # Convert to torch tensor
        sharp_image = torch.from_numpy(sharp_image)
        blur_image = torch.from_numpy(blur_image)

        return sharp_image, blur_image

    def __len__(self):
        # return length of the dataset (Number of samples)
        return len(self.dataset)