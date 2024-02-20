import os
from shutil import copyfile

import click
import tqdm


@click.command()
@click.option('--dir_in', default='GOPRO_Large')
@click.option('--dir_out')
def reorganize_gopro_files(dir_in, dir_out):
    """
    This function can be used to organize dataset files.
    orginal dataset file directory:
        gopro
            |
             train 
            |     |
            |     bunch of sub folders --- sharp
            |                          --- blur
            |
            |
             test --"like train folder"
        
    Result:
        gopro
            |
            train -- sharp images
            |     -- blur images
            |
            test -- sharp images
                 -- blur images
    
    Args:
        dir_in: orginal directory path of the dataset
        dir_out: path of the organized dataset

    """
    if not os.path.exists(dir_out):
        # Create a directory for the output dataset
        os.makedirs(dir_out)

    # Cover both train and test folders
    for train_test_folder in tqdm.tqdm(os.listdir(dir_in), desc='dir'):
        # Create directory for sharp and blur images in organized directory.
        output_directory = os.path.join(dir_out, train_test_folder)
        out_dir_blur = os.path.join(output_directory, 'blur')
        out_dir_sharp = os.path.join(output_directory, 'sharp')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        if not os.path.exists(out_dir_blur):
            os.makedirs(out_dir_blur)
        if not os.path.exists(out_dir_sharp):
            os.makedirs(out_dir_sharp)

        current_folder_path = os.path.join(dir_in, train_test_folder)
        # For the following section we are in either train or test directory
        # For each sub folder in above directory, the code copy sharp or blur image from orginal directory to the corresponding target directory
        # The final name of each image is "name of sub folder + _ + name of the image in orginal directory"
        for image_folder in tqdm.tqdm(os.listdir(current_folder_path), desc='image_folders'):

            current_sub_folder_path = os.path.join(current_folder_path, image_folder)

            for image_blurred in os.listdir(os.path.join(current_sub_folder_path, 'blur')):
                current_image_blurred_path = os.path.join(current_sub_folder_path, 'blur', image_blurred)
                output_image_blurred_path = os.path.join(out_dir_blur, image_folder + "_" + image_blurred)
                copyfile(current_image_blurred_path, output_image_blurred_path)

            for image_sharp in os.listdir(os.path.join(current_sub_folder_path, 'sharp')):
                current_image_sharp_path = os.path.join(current_sub_folder_path, 'sharp', image_sharp)
                output_image_sharp_path = os.path.join(out_dir_sharp, image_folder + "_" + image_sharp)
                copyfile(current_image_sharp_path, output_image_sharp_path)


if __name__ == "__main__":
    reorganize_gopro_files()
