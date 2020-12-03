import logging
import os
import shutil
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import splitfolders
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

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


def __unzip_file(filepath, target="/content/"):
    filepath = Path(filepath)
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith(filepath.stem + "/"):
                zip_ref.extract(file, target)

    # Remove annoying child folder...
    file_names = os.listdir(Path(target).joinpath(filepath.stem))

    for file_name in file_names:
        shutil.move(
            os.path.join(Path(target).joinpath(filepath.stem), file_name), target
        )

    shutil.rmtree(Path(target).joinpath(filepath.stem))


def _unzip_file(filepath, target="/content/"):
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


def unzip_files(path_notebooks, path_training, path_test, train_folder, test_folder):
    if not Path(Path(path_notebooks).parent).joinpath("model.py").is_file():
        log.info("Unzip notebooks...")
        _unzip_file(path_notebooks)
    else:
        log.info("Notebook is already unzipped")
    if not train_folder.is_dir():
        log.info("Unzip train...")
        # _unzip_file(path_training)
    else:
        log.info("Train dataset is already unzipped")
    if not test_folder.is_dir():
        log.info("Unzip test...")
        # _unzip_file(path_test)
    else:
        log.info("Test dataset is already unzipped")


def move_notebook_files(notebook_folder):
    if not Path(Path(notebook_folder).parent).joinpath("model.py").is_file():
        file_names = os.listdir(notebook_folder)
        for file_name in file_names:
            shutil.move(
                os.path.join(notebook_folder, file_name), Path(notebook_folder).parent
            )

        shutil.rmtree(notebook_folder)
    else:
        log.info("Notebooks are also already moved")


def split_dataset(train_dataset, target, seed, ratio):
    # Call the split_folders module (https://github.com/jfilter/split-folders)
    if not target.is_dir():
        log.info("Create train/val split")
        splitfolders.ratio(train_dataset, target, seed=seed, ratio=ratio)
    else:
        log.info("Train/val split already exists")


def train(litmodel, trainer):
    trainer.fit(litmodel)
    return litmodel


def test(litmodel, trainer):
    log.info("Test Result: " + str(trainer.test(litmodel)))


def setup_train_env(hparams, mean, std, train_val_folder, test_folder):
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_acc", save_last=True, save_top_k=1, mode="max"
    )
    early_stop_callback = EarlyStopping(
        monitor="avg_val_acc", min_delta=0.00, patience=5, verbose=False, mode="max"
    )
    litmodel = LitModel(hparams, mean, std, train_val_folder, test_folder)
    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=False,
        max_epochs=1,
        progress_bar_refresh_rate=200,
        default_root_dir="/content/drive/My Drive/Master Thesis/checkpoints",
        profiler=False,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    return litmodel, trainer


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    base_folder = "/content"
    path_notebooks = "/content/drive/My Drive/Master Thesis/notebooks.zip"
    path_train = "/content/drive/My Drive/Master Thesis/rework/original_training.zip"
    path_test = "/content/drive/My Drive/Master Thesis/rework/original_test.zip"

    train_folder = Path(base_folder).joinpath(Path(path_train).stem)
    test_folder = Path(base_folder).joinpath(Path(path_test).stem)
    train_val_folder = train_folder.joinpath("train_val")

    # Setup files
    unzip_files(path_notebooks, path_train, path_test, train_folder, test_folder)

    move_notebook_files(str(Path(base_folder).joinpath(Path(path_notebooks).stem)))
    from model import LitModel, hparams  # Now we can import the unpacked files

    split_dataset(train_folder, train_val_folder, seed=2020, ratio=(0.7, 0.3))

    # Calculate mean
    # mean, std = cal_dir_stat(str(Path(train_val_folder).joinpath("train")), "ppm")
    # log.info(mean)
    # log.info(std)
    mean = [0.31212274001080575, 0.28787920805165645, 0.29833593771990724]
    std = [0.2788638916341848, 0.2672319741765941, 0.2756277781233801]

    # Setup model/trainer
    litmodel, trainer = setup_train_env(
        hparams, mean, std, train_val_folder, test_folder
    )

    # Train/test
    litmodel = train(litmodel, trainer)
    test(litmodel, trainer)
