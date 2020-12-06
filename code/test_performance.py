"""
This class runs performance tests for all passed datasets
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from tqdm import tqdm

from model import LitModel

# Some of the methods we call have changed the default behavior and indicate this with a warning.
# Since the changes are not relevant for us, we hide the warnings.
warnings.filterwarnings("ignore")


class TestPerformance:
    def __init__(
        self, mean: list = None, std: list = None, filetype_list: list = None
    ) -> None:
        self.mean = mean
        self.std = std
        self.FILETYPE_LIST = (
            ["pdf", "png", "svg"] if not filetype_list else filetype_list
        )

    def store_diagrams_and_df(
        self,
        data_dict: dict,
        colums: list,
        image_name: str,
        df_name: str,
        set_y_axis_range: bool = False,
    ) -> None:
        """
        Creates images in all passed formats and stores dataframe as csv file

        Parameters
        ----------
        data_dict : dict
            Results
        colums : list
            Names for the Columns (in df)
        image_name : str
            Filename of the images
        df_name : str
            Filename of the csv file
        set_y_axis_range : bool, optional
            Scale Y-Axis, by default False
        """
        df = pd.DataFrame(data_dict.items(), columns=colums)
        sns.set_theme(style="whitegrid")
        seaborn_plot = sns.relplot(data=df, x=colums[0], y=colums[1])

        # Remove Underscore at the end if it exists
        if image_name[-1] == "_":
            image_name = image_name[:-1]

        if set_y_axis_range:
            seaborn_plot.set(ylim=(0, 1))
        for file_type in self.FILETYPE_LIST:
            seaborn_plot.savefig(f"{Path.cwd().joinpath(image_name)}.{file_type}")
        if df_name is not None:
            df.to_csv(f"{Path.cwd().joinpath(df_name)}.csv")

    def get_test_dataloader(self, dataset: list, mean: list, std: list) -> DataLoader:
        """
        Create dataloader

        Parameters
        ----------
        dataset : str
            List with paths to datasets
        mean : list
            Mean values for all three channels in train dataset
        std : list
            Standard derivation values for all three channels in train dataset

        Returns
        -------
        DataLoader
        """
        test_set_normal = torchvision.datasets.ImageFolder(
            root=str(dataset),
            transform=transforms.Compose(
                [
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )

        return DataLoader(test_set_normal, batch_size=50, num_workers=8)

    def run_test(self, trainer: Trainer, model: LitModel, dataset: list) -> dict:
        """
        Performes test

        Parameters
        ----------
        trainer : Trainer
            Trainer to be used
        model : LitModel
            Trained model
        dataset : str
            Path to dataset

        Returns
        -------
        dict
            Results
        """
        result_dict = {}
        if isinstance(dataset, str):
            dataloader = self.get_test_dataloader(dataset, self.mean, self.std)
            result_dict[0] = trainer.test(model, test_dataloaders=dataloader)[0][
                "avg_test_acc"
            ]
        else:
            # Loop over range
            print("in else")
            for dataset in tqdm(dataset):
                # Create Dataloader
                dataloader = self.get_test_dataloader(dataset, self.mean, self.std)
                # run test and store the result
                result_dict[int(dataset.name.split("_")[-1])] = trainer.test(
                    model, test_dataloaders=dataloader
                )[0]["avg_test_acc"]

        return result_dict

    def main(self, checkpoint: str, method_names: list, datasets: list):
        """
        Parameters
        ----------
        checkpoint : str
            Trained model
        method_names : list
            Method that should be used
        datasets : list
            Datasets
        """
        # Set up model and trainer
        model = LitModel.load_from_checkpoint(
            checkpoint,
            mean=args.mean,
            std=args.std,
            train_dataset=None,
            test_dataset=None,
        )
        use_gpu = True if torch.cuda.is_available() else False
        if use_gpu:
            trainer = Trainer(gpus=1)
        else:
            trainer = Trainer(gpus=0)

        method_cnt = 0
        for dataset in datasets:
            # Check if there are multiple datasets with this prefix
            if dataset[-1] == "_":
                prefix_datasets = list(
                    (Path(dataset).parent).glob(f"{Path(dataset).name}[0-9]*")
                )
                result = self.run_test(trainer, model, prefix_datasets)
            else:
                result = self.run_test(trainer, model, dataset)

            self.store_diagrams_and_df(
                result,
                [method_names[method_cnt], "Test accuracy"],
                Path(dataset).name,
                f"{Path(dataset).name}_df",
            )

            method_cnt += 1


def parse_args():
    """
    Parses arguments

    Returns
    -------
    argsparse.Namespace
        Namespace with all args
    """
    # Handle args parsing
    parser = argparse.ArgumentParser(description="Mask images")

    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        metavar="Model Checkpoint",
        required=True,
        type=str,
        help="File with stored state of the trained model",
    )

    parser.add_argument(
        "--method_names",
        dest="method_names",
        metavar="List with names of Methods",
        nargs="+",
        required=True,
        help="List (seperated with Space) with names of methods, must be in the same order as --datasets",
    )
    parser.add_argument(
        "--datasets",
        dest="datasets",
        metavar="List with names of Datasets",
        nargs="+",
        required=True,
        help="List (seperated with Space) with name of datasets. Datasets must be in the same folder as this file. If a name ends with '_' the skript will use all folders with a prefix to that name",
    )

    parser.add_argument("--mean", dest="mean", nargs="+", required=True)

    parser.add_argument("--std", dest="std", default=None, nargs="+", required=True)

    args = parser.parse_args()
    args.mean = None if args.mean is None else [float(i) for i in args.mean]
    args.std = None if args.std is None else [float(i) for i in args.std]
    return args


if __name__ == "__main__":
    args = parse_args()
    runner = TestPerformance(mean=args.mean, std=args.std)
    runner.main(args.checkpoint, args.method_names, args.datasets)
