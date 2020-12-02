"""
This file prepares the data set for the training process of the CNN. 
For this purpose 3 steps are performed:

- Increasing the examples by mirroring
- Adjust the number of examples by duplicating
- Rotate the images along the X-axis
"""

import argparse
import logging
import random
import shutil
from pathlib import Path
from shutil import copyfile

import numpy as np
from matplotlib import image as mpimg
from PIL import Image
from tqdm import tqdm

# Logging config
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)


def rotated_training_dataset(dataset_folder: Path, scale_rotation: int) -> None:
    """
    Rotates the images based on a normal distribution

    Parameters
    ----------
    dataset_folder : Path
        Path to datasetfolder
    scale_rotation : int
        Width of the bell of normal distribution when rotating in degree
    """
    # Get the number of files
    number_files = len([y for y in dataset_folder.rglob("*.ppm")])

    # Create and show normal distribution with as many values as images
    # scale adjusts the Standard Deviation
    norm_dist = np.random.normal(loc=0, scale=scale_rotation, size=number_files)

    # Loop over all images and pick the corresponding entry in the normal_dist list
    # This gives each image an random value from our normal distribution
    for i, path in tqdm(enumerate(dataset_folder.rglob("*.ppm")), total=number_files):
        image = Image.open(path)
        rotated = image.rotate(norm_dist[i])
        rotated.save(path)


def flip_images(directories: Path, n_classes: int) -> None:
    """
    Increases number of images in certain classes by copying and mirroring the signs

    Parameters
    ----------
    directories : Path
        Path to datasetfolder
    n_classes : int
        Number of classes
    """
    flip_horizontal = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35, 40]
    flip_vertical = [1, 5, 12, 15, 17]
    flip_to_each_other = {
        19: 20,
        33: 34,
        36: 37,
        38: 39,
        20: 19,
        34: 33,
        37: 36,
        39: 38,
    }

    folder = sorted([x for x in directories.iterdir() if x.is_dir()])

    for c in tqdm(range(n_classes)):
        files_in_folder = [x for x in folder[c].iterdir() if x.is_file()]

        if c in flip_horizontal:
            for image in files_in_folder:
                image_path = image.parents[0].joinpath(
                    str(image.stem) + "_f" + str(image.suffix)
                )
                img = mpimg.imread(image)
                flipped_img = np.fliplr(img)
                flipped_img = flipped_img.astype(np.uint8)
                flipped_img = Image.fromarray(flipped_img)
                flipped_img.save(str(image_path))

        if c in flip_vertical:
            for image in files_in_folder:
                image_path = image.parents[0].joinpath(
                    str(image.stem) + "_f" + str(image.suffix)
                )
                img = mpimg.imread(image)
                flipped_img = np.flipud(img)
                flipped_img = flipped_img.astype(np.uint8)
                flipped_img = Image.fromarray(flipped_img)
                flipped_img.save(str(image_path))

        if c in flip_to_each_other.keys():
            for image in files_in_folder:
                image_path = Path(
                    str(image.parents[0]).replace(
                        str(c), str(flip_to_each_other.get(c))
                    )
                ).joinpath(str(image.stem) + "_f" + str(image.suffix))
                img = mpimg.imread(image)
                flipped_img = np.fliplr(img)
                flipped_img = flipped_img.astype(np.uint8)
                flipped_img = Image.fromarray(flipped_img)
                flipped_img.save(str(image_path))


def get_number_of_files(directory: str, cnt: int = 0) -> int:
    """
    Traverses passed folder and contained subfolders and returns the total number of contained files

    Parameters
    ----------
    directory : str
        Parent directory
    cnt : int, optional
        Starting point of the counter, by default 0

    Returns
    -------
    int
        Number of files in directory
    """
    for path in Path(directory).iterdir():
        if path.is_dir():
            # Recursiv call for subfolders
            cnt = get_number_of_files(path, cnt)
        elif path.is_file():
            cnt += 1
    return cnt


def get_max_number_of_files(path: str) -> int:
    """
    Returns the maximum amount of files in any of the subfolders in path

    Parameters
    ----------
    path : str
        Parent directory

    Returns
    -------
    int
        Maximum amount in any subfolder
    """
    directories = Path(path)
    lst = []
    for directory in [x for x in directories.iterdir() if x.is_dir()]:
        num_files_source = get_number_of_files(directory, 0)
        lst.append(num_files_source)

    return max(lst)


def upsample_files_in_folder(folder_path: Path, target_num: int) -> None:
    """
    Duplicates random files from current folder until the number of files in this folder matches target_num

    Parameters
    ----------
    folder_path : Path
        Folder to work in
    target_num : int
        Target number of files in folder
    """

    file_list = list(Path(folder_path).glob("**/*.ppm"))
    num_req_duplicates = target_num - len(file_list)
    for i in range(0, num_req_duplicates):
        sel_file = random.choice(file_list)
        copyfile(
            str(sel_file),
            str(
                sel_file.parents[0].joinpath(
                    sel_file.stem + "_" + str(i) + sel_file.suffix
                )
            ),
        )


def copy_folder(src_dataset_folder: Path, target_dataset_folder: Path) -> None:
    """
    Creates a duplicate from source folder

    Parameters
    ----------
    src_dataset_folder : Path
        Path to source folder
    target_dataset_folder : Path
        Path to target folder
    """
    if not target_dataset_folder.is_dir():
        shutil.copytree(src_dataset_folder, target_dataset_folder)


def parse_arguments() -> argparse.Namespace:
    """
    Parses bash arguments

    Returns
    -------
    argparse.Namespace
        Namespace with parsed arguments
    """

    parser = argparse.ArgumentParser(description="Initializes Dataset for training")
    parser.add_argument(
        "-p",
        "--source_path",
        type=str,
        default=None,
        required=True,
        help="Source path of the dataset (default: None)",
    )
    parser.add_argument(
        "-t",
        "--target_path",
        type=str,
        default=None,
        required=True,
        help="Target path for initialized dataset (default: None)",
    )
    parser.add_argument(
        "-n",
        "--num_classes",
        type=int,
        default=43,
        required=False,
        help="Number of classes in dataset (default: 43)",
    )
    parser.add_argument(
        "-u",
        "--num_upsampling",
        type=int,
        default=4500,
        required=False,
        help="Number of files to which each folder should be amplified (default: 4500)",
    )
    parser.add_argument(
        "-r",
        "--rotation",
        type=int,
        default=20,
        required=False,
        help="Width of the bell of normal distribution when rotating in degree (default: 20)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    src_folder = Path(args.source_path)
    trg_folder = Path(args.target_path)
    # Create dataset at target location
    log.info("Copy data to target location...")
    copy_folder(src_folder, trg_folder)
    # Increase number of examples through flipping
    log.info("Flip images...")
    flip_images(trg_folder, args.num_classes)
    # Upsample dataset for balanced classes
    log.info("Upsample dataset...")
    for directory in tqdm([x for x in trg_folder.iterdir() if x.is_dir()]):
        upsample_files_in_folder(str(directory), args.num_upsampling)
    # Rotate images
    log.info("Rotate images...")
    rotated_training_dataset(trg_folder, args.rotation)
