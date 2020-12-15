import argparse
import logging
import os
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import splitfolders
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from model import LitModel, hparams

# Some of the methods we call have changed the default behavior and indicate this with a warning.
# Since the changes are not relevant for us, we hide the warnings.
warnings.filterwarnings("ignore")

# Logging config
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)


def cal_dir_stat(root: str, filetype: str) -> tuple:
    """
    Calculates the mean and standard deviation per channel for all images in the
    passed dateset. Do not calculate the statistics on the whole dataset,
    as per here http://cs231n.github.io/neural-networks-2/#datapre.

    root:
        Path to dataset
    filetype:
        Type of images in dataset, e.g. ppm or jpg

    """
    log.info("Calculate mean/std values")
    CHANNEL_NUM = 3
    cls_dirs = [x for x in Path(root).iterdir() if x.is_dir()]
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for idx, d in enumerate(cls_dirs):
        for image in tqdm(d.glob(f"*.{filetype}"), total=len(cls_dirs)):
            im = cv2.imread(
                str(image)
            )  # image in M*N*CHANNEL_NUM shape, channel in BGR order
            im = im / 255.0
            pixel_num += im.size / CHANNEL_NUM
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


def _unzip_file(filepath, target="/content/"):
    """
    Internal method for unzipping of files

    Parameters
    ----------
    filepath : str
        Path to zip file
    target : str, optional
        Target directory, by default "/content/"
    """
    filepath = Path(filepath)

    with zipfile.ZipFile(filepath, "r") as zip_ref:
        for file in tqdm(zip_ref.namelist()):
            folder_name = filepath.stem + "/"
            if file.startswith(folder_name) and not file == folder_name:
                tmp_target = Path(Path(target).joinpath(file))
                tmp_target.parent.mkdir(exist_ok=True, parents=True)

                if file.endswith("/"):
                    tmp_target.mkdir(exist_ok=True, parents=True)
                else:
                    with zip_ref.open(file) as zf, open(tmp_target, "wb") as f:
                        shutil.copyfileobj(zf, f)


def unzip_files(
    path_training: str, path_test: str, train_folder: str, test_folder: str
) -> None:
    """
    Unzips train and test dataset zip files

    Parameters
    ----------
    path_training : str
        Path to training zip file
    path_test : str
        Path to test zip file
    train_folder : str
        Target folder of train dataset (used to validate, if zip was already unzipped)
    test_folder : str
        Target folder of test dataset (used to validate, if zip was already unzipped)
    """

    if not train_folder.is_dir():
        log.info("Unzip train...")
        _unzip_file(path_training)
    else:
        log.info("Train dataset is already unzipped")
    if not test_folder.is_dir():
        log.info("Unzip test...")
        _unzip_file(path_test)
    else:
        log.info("Test dataset is already unzipped")


def split_dataset(train_dataset: str, target: str, seed: int, ratio: str) -> None:
    """
    Splits train dataset into train and validation folder

    Parameters
    ----------
    train_dataset : str
        Path to folder with trainigs dataset
    target : str
        Path to parent path of the new train/val folder
    seed : int
        Seed for random selection of files
    ratio : str
        Ratio of the train/val split. Must be in for of a tuple!
    """
    # Call the split_folders module (https://github.com/jfilter/split-folders)
    if not target.is_dir():
        log.info("Create train/val split")
        splitfolders.ratio(train_dataset, target, seed=seed, ratio=ratio)
    else:
        log.info("Train/val split already exists")


def train(litmodel: LitModel, trainer: pl.Trainer) -> LitModel:
    """
    Trains model

    Parameters
    ----------
    litmodel : model.LitModel
        Model to train
    trainer : pytorch_lightning.trainer.trainer.Trainer
        Trainer

    Returns
    -------
    model.LitModel
        Trained model
    """

    trainer.fit(litmodel)
    return litmodel


def test(litmodel: LitModel, trainer: pl.Trainer) -> None:
    """
    Runs test on trained model and logs result

    Parameters
    ----------
    litmodel : model.LitModel
        Trained model
    trainer : pytorch_lightning.trainer.trainer.Trainer
        Trainer
    """
    log.info("Test Result: " + str(trainer.test(litmodel)))


def setup_train_env(
    destination: str,
    hparams: dict,
    mean: list,
    std: list,
    train_val_folder: str,
    test_folder: str,
    epochs: str,
) -> Tuple[LitModel, pl.Trainer]:
    """
    Creates model and trainer with the passed settings

    Parameters
    ----------
    destination :
        Location where checkpoints will be stored
    hparams : dict
        Settings for model configuration(dropout rate, learning rate, momentum, optimizer, activationfunctions for features and classifier and stn parameters)
    mean : list
        Mean values for all three channels in train dataset
    std : list
        Standardderivation values for all three channels in train dataset
    train_val_folder : str
        Path to train/val dataset folder
    test_folder : str
        Path to test folder
    epochs : str
        Maximum number of epochs

    Returns
    -------
    model.LitModel, pytorch_lightning.trainer.trainer.Trainer
        Model and trainer ready for training
    """
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_acc", save_last=True, save_top_k=1, mode="max"
    )
    early_stop_callback = EarlyStopping(
        monitor="avg_val_acc", min_delta=0.00, patience=5, verbose=False, mode="max"
    )
    litmodel = LitModel(hparams, mean, std, train_val_folder, test_folder)
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        gpus=num_gpus,
        fast_dev_run=False,
        max_epochs=epochs,
        progress_bar_refresh_rate=200,
        default_root_dir=destination,
        profiler=False,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    return litmodel, trainer


def parse_args():
    """
    Parse shell arguments

    Returns
    -------
    argsparse.Namespace
        Namespace with all arguments
    """
    parser = argparse.ArgumentParser(description="Train and test model")
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        required=True,
        help="Path to train dataset",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
        required=True,
        help="Path to test dataset",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default=None,
        required=True,
        help="Path where the checkpoints should be stored",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2020,
        required=False,
        help="Seed for train/validation split (default: 2020)",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=str,
        default="(0.7, 0.3)",
        required=False,
        help='Ratio for train/validation split, must be in tuple format! (default: "(0.7, 0.3)")',
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=15,
        required=False,
        help="Max number of epoches(default: 15)",
    )
    parser.add_argument(
        "-d",
        "--dropout_rate",
        type=float,
        default=0.45,
        required=False,
        help="Dropout range (default: 0.45)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    args = parse_args()
    train_val_folder = Path(args.train_dataset).joinpath("train_val")

    # Split train dataset into train and validation
    split_dataset(args.train_dataset, train_val_folder, seed=2020, ratio=(0.7, 0.3))

    # Calculate mean
    mean, std = cal_dir_stat(str(Path(train_val_folder).joinpath("train")), "ppm")
    log.info("Mean: " + str(mean))
    log.info("Std: " + str(std))

    # Setup model/trainer
    hparams["dropout_rate"] = args.dropout_rate
    litmodel, trainer = setup_train_env(
        destination = args.destination,
        hparams=hparams,
        mean=mean,
        std=std,
        train_val_folder=train_val_folder,
        test_folder=args.test_dataset,
        epochs=args.epochs,
    )

    # Train/test
    litmodel = train(litmodel, trainer)
    test(litmodel, trainer)
