from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from PIL import Image
import os


class GoProDataLoader(Dataset):
    """
        Loading the dataset for training
    """

    def __init__(self, root_path, dataset_path, transform=None, image_size=256):
        """
        Initializing the dataset arguments.
        
        Args:
            root_path: Path of the dataset directory.
            dataset_path: Path of the dataset list.
            transform: Transforms to be applied on the dataset.
            image_size: Size of the images.

        Example:
        train_dataset = GoProDataLoader(root_path="gopro", dataset_path="gopro.train.csv")
        """
        
        super(GoProDataLoader, self).__init__()
        self.root_path = root_path
        self.dataset = pd.read_csv(dataset_path)
        self.image_size = image_size
        self.transform = transform

        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])


    def __getitem__(self, index):
        # select
        selected_image = self.dataset.iloc[index]

        # path
        sharp_image_path = os.path.join(self.root_path, selected_image["Folder_name"] + "/sharp/" + selected_image["File_name"])
        blur_image_path = os.path.join(self.root_path, selected_image["Folder_name"] + "/blur/" + selected_image["File_name"])

        # load
        sharp_image = Image.open(sharp_image_path).convert('RGB')
        blur_image = Image.open(blur_image_path).convert('RGB')

        # Apply transform
        if self.transform:
            # Apply same transform to both images
            sharp_image = self.transform(sharp_image)
            blur_image = self.transform(blur_image)
        else:
            sharp_image = transforms.ToTensor()(sharp_image)
            blur_image = transforms.ToTensor()(blur_image)

        return sharp_image, blur_image

    def __len__(self):
        # return length of the dataset (Number of samples)
        return len(self.dataset)
    
def get_data_loaders(root_path, dataset_path, batch_size=4, image_size=256, num_workers=4):
    """
    Create data loaders for training and testing.
    
    Args:
        data_dir (str): Directory with images (for synthetic noise).
        batch_size (int): Batch size for data loaders.
        image_size (int): Size to resize images to.
        num_workers (int): Number of workers for data loading.
        
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing.
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    
    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = GoProDataLoader(
            root_path=os.path.join(root_path, "train"), #root_path,
            dataset_path=os.path.join(dataset_path, "train_samples.csv"), #dataset_path,
            image_size=image_size,
            transform=train_transform
        )
        
    test_dataset = GoProDataLoader(
        root_path=os.path.join(root_path, "test"), #root_path,
        dataset_path=os.path.join(dataset_path, "test_samples.csv"), #dataset_path,
        image_size=image_size,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader