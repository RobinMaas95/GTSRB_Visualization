"""
Calculates the most important grid cells and masks them
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class CalculateGridCell:
    """ Calculates the most important grid cells """

    def __init__(self, dataset_folder, json_target_folder, json_name) -> None:
        self.dataset_folder = dataset_folder
        self.json_target_folder = json_target_folder
        self.json_name = json_name

    def get_grid_index(self, number: int) -> int:
        """
        Returns row or column id for value (grid with 8x8)

        Parameters
        ----------
        number : int
            
        Returns
        -------
        ind - Grid Cell Index
        """
        if 0 <= number <= 5:
            ind = 0
        elif 5 <= number <= 11:
            ind = 1
        elif 12 <= number <= 17:
            ind = 2
        elif 18 <= number <= 23:
            ind = 3
        elif 24 <= number <= 29:
            ind = 4
        elif 30 <= number <= 35:
            ind = 5
        elif 36 <= number <= 41:
            ind = 6
        else:
            ind = 7

        return ind

    def get_grid_number(self, row: int, column: int, number_columns=8) -> int:
        """
        Calculates the id of the grid cell id. Starting at 0 in the top left corner, ending with
        row_number*column_number in the bottom right corner

        Parameters
        ----------
        row : int
        column : int
        number_columns : int, optional
            number of columns in grid, by default 8

        Returns
        -------
        int
            Grid Cell ID
        """
        gid = row * number_columns + column
        return gid

    def calc_cells(self, dataset_folder: str) -> dict:
        """
        Calculates most important cells

        Parameters
        ----------
        dataset_folder : str
            Folder with Dataset (subdirectories contain images)

        Returns
        -------
        dict
            Dictionary with most relevant cells for each image
        """
        folder_list = [f for f in dataset_folder.iterdir() if f.is_dir()]
        overview_dict = {}

        for folder in folder_list:
            files = [x for x in folder.glob("**/*.ppm") if x.is_file()]

            folder_dict = {}

            preprocess = transforms.Compose([transforms.ToTensor()])
            for image in files:
                # Get flatten image tensor
                ex_image = Image.open(image)
                tensor = preprocess(ex_image)
                flatten_tensor = tensor.view(1, -1)

                # Get 576 pixels with the highest values (25% of all pixels)
                pixel_dict = {}
                for i in range(576):
                    pixel = flatten_tensor.argmax(1).item()
                    pixel_value = flatten_tensor[0][pixel].item()
                    if pixel_value == 0.0:
                        # The break is not really neccessary, because pixels
                        # with a value of 0.0 wouldnt change the end result, but
                        # we can save some processing time here
                        break
                    pixel_dict[pixel] = pixel_value
                    flatten_tensor[0][pixel] = 0

                # Get the most common square
                grid_square_dict = {}
                for pixel, pixel_value in pixel_dict.items():
                    row = int(pixel / 48)
                    gid = self.get_grid_number(
                        self.get_grid_index(row), self.get_grid_index(pixel - row * 48)
                    )
                    grid_val = grid_square_dict.get(gid, 0)
                    grid_square_dict[gid] = grid_val + pixel_value

                # Skip image, if there is no activation on it
                if len(grid_square_dict) == 0:
                    continue

                # Get second most activated square
                activated_cells = []
                grid_square_dict_copy = copy.deepcopy(grid_square_dict)
                for i in range(10):
                    try:
                        current_max = max(
                            grid_square_dict_copy,
                            key=lambda key: grid_square_dict_copy[key],
                        )
                    except ValueError:
                        # Stop looping, if there are no more squares in the list
                        break
                    activated_cells.append(current_max)
                    grid_square_dict_copy.pop(current_max)

                folder_dict[image] = activated_cells

            overview_dict[folder.name] = folder_dict

        return overview_dict

    def store_overview_dict(
        self,
        overview_dict: dict,
        json_target_folder: str,
        json_filename: str,
        vis_method: str,
    ) -> None:
        """
        Stores dictionary with relevant cells for all images to a json file

        Parameters
        ----------
        overview_dict : dict
            Dict with cells for all images
        json_target_folder : str
            Path to target folder
        json_filename : str
            Name of json file
        vis_method : str
        """
        ## create dict with strings as keys
        str_overview_dict = {}
        for subdict in overview_dict:
            str_subdict = {str(k): v for k, v in overview_dict[subdict].items()}
            str_overview_dict[subdict] = str_subdict

        ## Store dict
        target_folder = Path(json_target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)
        with open(
            str(target_folder.joinpath(f"{vis_method}_{json_filename}")), "w"
        ) as f:
            json.dump(str_overview_dict, f, indent=4, sort_keys=True)

    def main(self, vis_method: str) -> dict:
        """
        Parameters
        ----------
        vis_method : str

        Returns
        -------
        dict
            Overview dict with most important grid cells for each image
        """
        dataset_folder = Path(self.dataset_folder)
        overview_dict = self.calc_cells(dataset_folder)
        self.store_overview_dict(
            overview_dict, self.json_target_folder, self.json_name, vis_method
        )
        return overview_dict


class MaskOutCells:
    """
    Masks out Cells in images
    """

    def __init__(
        self,
        source_dict: str,
        org_image_folder: str,
        heatmap_folder: str,
        target_folder: str,
        img_size: int = 48,
        grid_size: int = 6,
        target_pixel_color: list = [255, 255, 255],
    ) -> None:
        self.source_dict = source_dict
        self.org_image_folder = Path(org_image_folder)
        self.heatmap_folder = Path(heatmap_folder)
        self.target_folder = Path(target_folder)
        self.img_size = img_size
        self.grid_size = grid_size
        self.target_pixel_color = target_pixel_color
        self.cells_in_row = self.img_size / self.grid_size

    def get_start_coordinates(self, grid_cell: int):
        """
        Calculates x-axis-start and y-axis-start of a grid cell

        Parameters
        ----------
        grid_cell : int
            Grid Cell

        Returns
        -------
        Tuple
            start_x_axis, start_y_axis
        """
        row_of_target = int(grid_cell / self.cells_in_row)
        col_of_target = int(grid_cell - row_of_target * self.cells_in_row)
        start_y_axis = int(row_of_target * self.grid_size)
        start_x_axis = int(col_of_target * self.grid_size)

        return start_x_axis, start_y_axis

    def maskout(self, number_of_marked_cells: int, vis_method: str) -> None:
        """
        Masks out the passed number of cells in each image

        Parameters
        ----------
        number_of_marked_cells : int
            Number of cells that should be masked
        vis_method : str
            Vis method
        """
        for folder in self.source_dict:
            for img_path, target_grid_cell_list in self.source_dict[folder].items():
                # Create paths
                target_image_heatmap = Path(img_path)
                target_image_original = self.org_image_folder.joinpath(
                    target_image_heatmap.parent.name, target_image_heatmap.name
                )

                ## Convert to numpy for higher performance
                try:
                    imnp = np.array(Image.open(target_image_original))
                except FileNotFoundError:
                    continue

                # Calculate start coords x-/y-axis
                for i in range(number_of_marked_cells):
                    try:
                        cel = target_grid_cell_list[i]
                    except IndexError:
                        # No more elements in list, skip to next image
                        break

                    start_x_axis, start_y_axis = self.get_start_coordinates(cel)

                    # Set Pixel Values
                    ## loop over cell to mask out the pixels
                    for y_axis in range(self.grid_size):
                        for x_axis in range(self.grid_size):
                            imnp[
                                start_y_axis + y_axis, start_x_axis + x_axis, :
                            ] = self.target_pixel_color

                # Save new image
                ## Make sure target folder (with class folder) exists
                gen_target_folder = self.target_folder.joinpath(
                    vis_method,
                    f"masked_{vis_method}_{str(number_of_marked_cells).zfill(2)}",
                    target_image_heatmap.parent.name,
                )
                gen_target_folder.mkdir(parents=True, exist_ok=True)
                ## Save
                Image.fromarray(imnp).save(
                    gen_target_folder.joinpath(target_image_heatmap.name)
                )


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
        "--heatmaps",
        dest="heatmaps",
        metavar="Heatmap Folder",
        required=True,
        type=str,
        help="Folder that contains the the heatmaps in subdirectories",
    )
    parser.add_argument(
        "--json_target",
        dest="json_target",
        metavar="JSON Target Folder",
        type=str,
        required=True,
        help="Folder where the result JSONs will be storted",
    )
    parser.add_argument(
        "--json_filename",
        metavar="JSON Filename",
        dest="json_filename",
        type=str,
        required=True,
        help="Base Name of the JSON Files",
    )
    parser.add_argument(
        "--org_images",
        metavar="Original Images",
        dest="org_images",
        type=str,
        required=True,
        help="Folder with the original dataset",
    )
    parser.add_argument(
        "--target",
        dest="target",
        metavar="Target Folder",
        type=str,
        required=True,
        help="Folder where the masked images will be stored",
    )
    parser.add_argument(
        "--max_number_masked",
        dest="num_masked",
        metavar="Max Number masked",
        type=int,
        default=10,
        help="Up to how many cells should be masked",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    num_dir = len([x for x in Path(args.heatmaps).iterdir() if x.is_dir()])
    print(num_dir)
    for idx, directory in tqdm(
        enumerate([x for x in Path(args.heatmaps).iterdir() if x.is_dir()]),
        total=num_dir,
    ):
        # Calc Dict
        calcGridCell = CalculateGridCell(
            directory, args.json_target, args.json_filename
        )
        overview_dict = calcGridCell.main(directory.name)

        # Baseline (without mask)
        gen_target_folder = Path(args.target).joinpath(
            directory.name, f"masked_{directory.name}_{str(0).zfill(2)}",
        )

        shutil.copytree(args.org_images, gen_target_folder)

        # Mask out cells
        maskCells = MaskOutCells(overview_dict, args.org_images, directory, args.target)
        for i in range(1, args.num_masked + 1):
            maskCells.maskout(number_of_marked_cells=i, vis_method=directory.name)

        # Create Zip File
        """
        zip_target = str(Path(args.target).joinpath(f"masked_{directory.name}"))
        shutil.make_archive(
            zip_target, "zip", Path(args.target).joinpath(directory.name)
        )
        """
