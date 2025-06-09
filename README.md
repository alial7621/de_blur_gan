# de_blur_gan

This repository contains the implementation of a Generative Adversarial Network (GAN) model for image restoration (deblurring). The code has been written in PyTorch.


## Installation

This project supports two methods for setting up the environment: native Python and Docker.

### 1. Native Python Environment
To set up the project using a native Python environment, install the required dependencies with the following command:
```
pip install -r requirements.txt
```
This project uses **PyTorch 2.5.0** with CUDA support for GPU acceleration. To install PyTorch 2.5.0 with CUDA, run:
```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```
Ensure you have a compatible CUDA version installed on your system.

### 2. Docker Container
To set up the project using Docker, follow these steps:

1. Build the Docker image using the provided `Dockerfile` and `docker-compose.yaml`:
   ```
   docker-compose build
   ```
2. Start the container in detached mode:
   ```
   docker-compose up -d
   ```
3. Access the container's shell for running commands:
   ```
   docker exec -it de_blur_gan bash
   ```
The `Dockerfile` and `docker-compose.yaml` handle the installation of dependencies, including PyTorch 2.5.1 with CUDA support. Ensure Docker and Docker Compose are installed on your system.


## Modules

`train.py`: Python script to train the model.

`deblur_modules/` folder:

`config.py`: To set configuration options and hyperparameter values.

`data_loader.py`: Dataloader class.

`losses.py`: All the needed loss functions to train the dataset.

`models.py`: Implemented neural network architectures for the generator and the discriminator models.

`organize_gopro_dataset.py`: Convert subfolders of the dataset into sharp and blurry images.



## Dataset and Data Preparation

The [GoPro Dataset](https://seungjunnah.github.io/Datasets/gopro) has been used to train the model. There are two versions of the dataset available, with differences in their volume. You can use either one of them based on your hardware and needs.

The dataset contains two folders, `train` and `test`. Each folder contains several subfolders, and each subfolder has its corresponding images. `train_samples.csv` and `test_samples.csv` in `/data` include a list of subfolders and their corresponding images.

If you want to organize the dataset in a way that separates sharp and blurry images into two folders, you can run the `deblur_modules/organize_gopro_dataset.py` module. This module will convert subfolders into sharp and blurred images. The following is the code to run this module for your convenience.

```
cd deblur_modules/
python organize_gopro_dataset.py --dir_in="dataset directory" --dir_out="target directory"
```

A sample of the dataset

Blurry image
![Blurry image](images/blury_image_sample.png)
Sharp image
![Sharp image](images/sharp_image_sample.png)

## Usage
The `train.py` module serves as the entry point for training.

### Training
To train the model, use the following command:
```
python train.py
```
Additional arguments can be provided to customize the training process, as defined in the `config.py` module. These arguments can be modified either in the `config.py` file or via the command line. For example:
```
python train.py --batch_size=8
```
Note that arguments like `--batch_size` are prefixed with double dashes (`--`).
