"""
Provides the original files in Training/Test folder. Can crop images (optional).
If necessary, the zip files are downloaded first.
"""

import argparse
import csv
import glob
import logging
import os
import shutil
import zipfile
from collections import namedtuple
from pathlib import Path
from shutil import move

import pandas as pd
import requests
from PIL import Image

# Logging config
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# Constants
WORKING_DIR = r"../data"
FINAL_TRAINING_URL = r"https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
FINAL_TEST_URL = r"https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
FINAL_TEST_GT_URL = r"https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"


def _download_url(url: str, save_path: str, chunk_size=128) -> None:
    """
    Downloads file from url

    Parameters
    ----------
    url : str
        File URL
    save_path : str
        Path to download target
    chunk_size : int, optional
        Chunk size while downloading, by default 128
    """
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def download_data(train_url: str, test_url: str, test_gt_url: str, target_path: str) -> None:
    """
    Downloads and unzips the necessary datasets

    Parameters
    ----------
    train_url : str
        URL to train dataset zip
    test_url : str
        URL to test dataset zip
    test_gt_url : str
        URL to test GT (Ground Truth) dataset zip
    target_path : 
        Path to target dir of datasets
    """
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    if Path.is_dir(target_path.joinpath("GTSRB", "Final_Training", "Images")) and Path.is_dir(target_path.joinpath("GTSRB", "Final_Test", "Images")):
        log.info("Source folder found, skip download/unzipping!")
    else:
        # Download Files
        log.info("Download training Dataset...")
        training_dataset = str(target_path.joinpath("GTSRB_Final_Training_Images.zip"))
        if not Path(training_dataset).is_file():
            _download_url(train_url, training_dataset)

        log.info("Download test Dataset...")
        test_dataset = str(target_path.joinpath("GTSRB_Final_Test_Images.zip"))
        if not Path(test_dataset).is_file():
            _download_url(test_url, test_dataset)

        log.info("Download test_GT dataset...")
        test_get_dataset = str(target_path.joinpath("GTSRB_Final_Test_GT.zip"))
        if not Path(test_get_dataset).is_file():
            _download_url(test_gt_url, test_get_dataset)

        # Unzip files
        log.info("Unzip training images")
        with zipfile.ZipFile(training_dataset, "r") as zip_ref:
            zip_ref.extractall(str(target_path))
        log.info("Unzip test images...")
        with zipfile.ZipFile(test_dataset, "r") as zip_ref:
            zip_ref.extractall(str(target_path))
        log.info("Unzip test GT")
        with zipfile.ZipFile(test_get_dataset, "r") as zip_ref:
            zip_ref.extractall(str(target_path))


def read_annotations(
    csv_file: str,
    dir: str,
    filename_header_txt: str = "Filename",
    label_header_txt: str = "ClassId",
    roi_labels: list = ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"],
) -> list:
    """
    Runs over csv file and read out filename, label and roi for each row. Stores it inside a namedtuple
    with the three keys "filename", "label" and roi.

    Parameters
    ----------
    csv_file:
        CSV file with information for all images in folder (filename and classId)
    filename_header_txt:
        Column header for filename column in CSV file
    label_header_txt :
        Column header for label column in CSV file
    roi_labels:
        List with the four header for the roi columns in CSV file

    Returns
    -------
    List with Annotation (namedtuple) for every image listed in csv.
    Tuple contains filename (str), label (str) and roi (namedtuple).
    """

    # Define a new namedtuple type to store the annotations
    Annotation = namedtuple("Annotation", ["filename", "label", "folder", "roi"])
    Roi = namedtuple("Roi", ["x1", "y1", "x2", "y2"])

    annotations = []
    with open(csv_file) as file:
        # Read in csv file
        reader = csv.DictReader(file, delimiter=";")

        # Iterate over all entries and collect filenames and labels
        next(reader)  # skip header row
        for row in reader:
            roi = Roi(
                int(row.get(roi_labels[0])),
                int(row.get(roi_labels[1])),
                int(row.get(roi_labels[2])),
                int(row.get(roi_labels[3])),
            )
            annotations.append(
                Annotation(
                    row.get(filename_header_txt), row.get(label_header_txt), dir, roi
                )
            )
    return annotations


def load_training_annotations(training_source_path: str) -> list:
    """
    Loops over all folders in training_source_path and searches for the annotation file (.csv).
    Passed the current annotation-file to read_annotation and unpacks the returned list into a list
    (collecting all returns). Returns a list with Annotations (namedtuple) for every image in the training dataset.

    Parameters
    ----------
    training_source_path
        Path to training dataset folder

    Returns
    -------
    List with Annotations (namedtuple) for every image in the training dataset
    """
    annotations = []
    # Loop over all folders in trainingdata folder
    directories = Path(training_source_path)
    for directory in [x for x in directories.iterdir() if x.is_dir()]:
        # Get all annotation-files (one in each folder)
        for file in glob.glob(str(Path(directory).joinpath("*.csv"))):
            # read_annotations returns a list, .extend() unpacks that before appending it to annotaions
            annotations.extend(read_annotations(file, directory))

    return annotations


def crop_images(annotations: list, output_path: str, crop: bool, size: int) -> None:
    """
    Converts all images listed in annotations into the passed format (size*size).
    Crops the images first in respect to the roi from the annotation and then
    up- or downscales the image.
    
    annotations:
        List with all annotations
    output_path:
        Path where the new images are to be stored
    crop: 
        Should the images be cropped?
    size:
        Image size in pixel
    """
    log.info("Start cropping images")

    # Variables to log how many images have already been processed
    i = 0
    cnt = 0
    divider = 5000

    # Loop over all annotations (Images in Dataset) and crop it
    for annotation in annotations:
        # Periodically logging
        if i % divider == 0:
            cnt += 1
            log.info(f"Annotation {cnt*divider} of {len(annotations)}")

        # Create paths for each image dynamically
        relative_path = Path(annotation.folder).joinpath(annotation.filename)

        target_folder = Path(output_path).joinpath(Path(annotation.folder).name)
        target_path = Path(target_folder).joinpath(annotation.filename)

        # Make sure that the folder exists, but the file not
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        Path(target_path).unlink(missing_ok=True)

        # Crop image and rescale it to size*size
        # Using LANCZOS for resampling, it has the worst performance but the best quality of all filters
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
        image = Image.open(relative_path)
        if crop:
            image = image.crop((annotation.roi.y1, annotation.roi.x1, annotation.roi.y2, annotation.roi.x2))
        image = image.resize(size=(size, size), resample=Image.LANCZOS)
        image.save(target_path)

        # Increase i to keep logging functional
        i += 1

    log.info("Finished cropping images")


def sort_test_data(folder: str, csv: str) -> None:
    """
    The test dataset comes unsorted (all images in one folder) with the labels given in the CSV. 
    To load it with torchvision.datasets.ImageFolder, the individual classes must be given in the folder structure. 
    This method creates these folders and moves the single images into the correct folders

    Parameters
    ----------
    folder : str
        Folder with test dataset
    csv : str
        Path to GT-final_test.csv
    """
    target = Path(folder)
    file_list = list(Path(folder).glob("**/*.ppm"))

    df = pd.read_csv(csv, sep=";")

    for file in file_list:
        label = str(df.loc[df["Filename"] == file.name, "ClassId"].iloc[0]).zfill(5)
        target_folder = target.joinpath(label)
        target_folder.mkdir(exist_ok=True, parents=True)
        move(str(file), str(target_folder.joinpath(file.name)))


def parse_args():
    """
    Parse shell arguments

    Returns
    -------
    argsparse.Namespace
        Namespace with all arguments
    """
    parser = argparse.ArgumentParser(
        description="Get datasets and (optional) crop them"
    )
    parser.add_argument(
        "--crop",
        dest="crop",
        metavar="Crop images",
        required=True,
        type=str,
        help="Should the images be cropped? (True/False)",
    )
    parser.add_argument(
        "--working_dir",
        dest="working_dir",
        metavar="Working Directory",
        required=False,
        default=WORKING_DIR,
        type=str,
        help="URL to Test GT Dataset",
    )
    parser.add_argument(
        "--url_train",
        dest="url_train",
        metavar="URL Train",
        required=False,
        default=FINAL_TRAINING_URL,
        type=str,
        help="Path to directory where datasets will be located",
    )
    parser.add_argument(
        "--url_test",
        dest="url_test",
        metavar="URL Test",
        required=False,
        default=FINAL_TEST_URL,
        type=str,
        help="URL to Test Dataset",
    )
    parser.add_argument(
        "--url_train_gt",
        dest="url_train_gt",
        metavar="URL Train GT",
        required=False,
        default=FINAL_TRAINING_URL,
        type=str,
        help="URL to Test GT Dataset",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Get arguments
    args = parse_args()

    # Define paths
    train_data = str(Path(args.working_dir).joinpath(r"GTSRB/Final_Training/Images"))
    test_data = str(Path(args.working_dir).joinpath(r"GTSRB/Final_Test/Images"))
    test_annotation = str(Path(args.working_dir).joinpath(r"GT-final_test.csv"))
    output_folder_train = str(Path(args.working_dir).joinpath(r"original_training"))
    output_folder_test = str(Path(args.working_dir).joinpath(r"original_test"))

    # Download data if necessary
    download_data(args.url_train, args.url_test, args.url_train_gt, args.working_dir)
    
    # Make crop argument to bool
    if args.crop.lower() == "true":
        crop = True
    else:
        crop = False
    
    if crop:
        output_folder_train = str(Path(args.working_dir).joinpath(r"cropped_training"))
        output_folder_test = str(Path(args.working_dir).joinpath(r"cropped_test"))

    # Read out annotations
    annotations = load_training_annotations(train_data)
    log.info(f"Image count: {len(annotations)}")

    # Convert images into the right shape, in this process we
    # first crop an image according to the roi in the annotations
    # and performs then the up- or downscaling
    crop_images(annotations, output_folder_train, crop, size=48)

    # Crop test images
    annotations = read_annotations(test_annotation, test_data)
    crop_images(annotations, output_folder_test, crop,  48)
    files_list = os.listdir(output_folder_test + "/Images")
    for files in files_list:
        shutil.move(
            output_folder_test + "/Images/" + files,
            output_folder_test + "/" + files,
        )
    shutil.rmtree(output_folder_test + "/Images")


    # Sort test images into folder
    sort_test_data(output_folder_test, test_annotation)
