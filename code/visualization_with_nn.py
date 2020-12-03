"""
Implementation of multiple visualization algorithms.
Uses the following repos:

    - Repo: nn_interpretability
        - Autor: hans66hsu
        - URL: https://github.com/hans66hsu/nn_interpretability

    - Repo:
        - Author: Yuchi Ishikawa
        - URL: https://github.com/yiskw713/SmoothGradCAMplusplus
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from itertools import combinations
CURRENT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(CURRENT).joinpath("pytorch_cnn_visualizations", "src")))

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imsave
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import LitModel
from nn_interpretability.interpretation.am.general_am import ActivationMaximization
from nn_interpretability.interpretation.saliency_map.saliency_map import SaliencyMap
from SmoothGradCAMplusplus.cam import GradCAM, SmoothGradCAMpp
from SmoothGradCAMplusplus.utils.visualize import reverse_normalize, visualize


def parse_args(vis_list):
    """
    Parses args

    Parameters
    ----------
    vis_list : str
        Possible visualizations

    Returns
    -------
    argsparse.Namespace
        Namespace with all args
    """
    # Handle args parsing
    parser = argparse.ArgumentParser(description="Calls visualization algorithms")
    parser.add_argument(
        "--dest",
        dest="dest",
        metavar="Destination",
        required=True,
        type=str,
        help="Destination path, subfolder structure will be restored",
    )
    parser.add_argument(
        "--filetype",
        dest="filetype",
        metavar="Filetype",
        type=str,
        default="ppm",
        help="File type of the images inside the subfolders (without '.'). By default 'ppm'",
    )
    parser.add_argument(
        "--model",
        metavar="Model",
        dest="model",
        type=str,
        required=True,
        help="Path to model file (.pt)",
    )
    parser.add_argument(
        "--num",
        metavar="Number Images",
        dest="num_images",
        type=int,
        default=50,
        help="How many images per folder should be (randomly) selected (default: 50)",
    )
    parser.add_argument(
        "--src",
        dest="src",
        metavar="Source",
        type=str,
        help="Path to source directory, script runs over subfolders (default: current Directory)",
    )
    parser.add_argument(
        "--vis",
        metavar="Visualization",
        dest="vis",
        type=str,
        help="Visualization algorithems that should be used (default: all). Choose from: "
        + str(vis_list),
    )
    parser.add_argument(
        "--mean",
        metavar="Mean",
        dest="mean",
        type=list,
        default=[0.3121227400108073, 0.28787920805165235, 0.2983359377199073],
        help="Mean Values for normalization (List with three values)",
    )

    parser.add_argument(
        "--std",
        metavar="std",
        dest="std",
        type=list,
        default=[0.2788638916341836, 0.2672319741766035, 0.2756277781233763],
        help="Std Values for normalization (List with three values)",
    )

    args = parser.parse_args()

    # Set source
    if args.src is None:
        Path(os.getcwd())
    else:
        Path(args.src)

    # Set vis
    args.vis = args.vis.split("/")
    # dest.mkdir(parents=True, exist_ok=True)
    return args


def get_algo_combinations(algos: list) -> str:
    """
    Creates String with all possible combinations of the implemented visualization algorithms

    Parameters
    ----------
    algos : list
        List with all visualization algorithms

    Returns
    -------
    str
        All possible combinations
    """
    subsets = []
    for L in range(0, len(algos) + 1):
        for subset in combinations(algos, L):
            subsets.append(subset)

    subsets_str = []
    for combination in subsets:
        if len(combination) > 0:
            combination_str = ""
            for el in combination:
                combination_str += el if len(combination_str) == 0 else "/" + el
            subsets_str.append(combination_str)

    return subsets_str


def prep_image(org_image: str, mean: list, std: list) -> torch.Tensor:
    """
    Prepares image. This includes the normalization and the transformation into a tensor

    Parameters
    ----------
    org_image : str
        Path to original image
    mean : list
        List with mean values for the rgb channels
    std : list
        List with std values for rgb channels

    Returns
    -------
    tensor
        Prepared image tensor
    """
    ex_image = Image.open(org_image)
    normalize = transforms.Normalize(mean, std)
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])

    tensor = preprocess(ex_image)
    prep_img = tensor.unsqueeze(0)
    return prep_img


def activation_maximation(
    model: LitModel,
    target_class: str,
    org_image: torch.Tensor,
    dest: str,
    mean: list,
    std: list,
    device: str,
) -> None:
    """
    Performs activation maximation for the passed class

    Parameters
    ----------
    model : LitModel
        Model for activation maximation
    target_class : str
        Target class
    org_image : torch.Tensor
        Mean image of dataset
    dest : str
        Path to destination folder
    mean : list
        Mean values for all three channels of the train dataset
    std : list
        Mean values for all three channels of the train dataset
    device : str
        Device that should be used ('cpu'/'cuda')
    """
    # Params
    img_shape = (3, 48, 48)
    lr = 0.001
    reg_term = 1e-2
    epochs = 1000
    threshold = 0.995

    # Random image as start
    start_img = torch.rand((1, 3, 48, 48)).to(device)
    mean_img = org_image  # Mean was calculated in mean and passed as org_image

    # Preprocess
    normalize = transforms.Normalize(mean, std)
    transforms.Compose([transforms.ToTensor(), normalize])

    # Get Interpreter
    interpretor = ActivationMaximization(
        model=model,
        classes=[],
        preprocess=None,
        input_size=img_shape,
        start_img=start_img,
        class_num=int(target_class),
        lr=lr,
        reg_term=reg_term,
        class_mean=mean_img,
        epochs=epochs,
        threshold=threshold,
    )

    end_point = interpretor.interpret()

    # Calc score
    scores = model(end_point.to(device)).to(device)
    prob = torch.nn.functional.softmax(scores, 1)[0][int(target_class)] * 100
    print(f"Class {target_class}: {prob}")

    # Restore Image (unnormalize)
    x_restore = end_point.reshape(img_shape) * torch.tensor(std).view(
        3, 1, 1
    ) + torch.tensor(mean).view(3, 1, 1)
    image_restored = x_restore.permute(1, 2, 0)

    # Save Image
    image_dest = Path(dest).joinpath("activation_maximation", target_class)
    image_dest.parent.mkdir(exist_ok=True, parents=True)

    plt.imsave(f"{image_dest}.ppm", image_restored.numpy())


def salience_map(
    model: LitModel,
    _,
    org_image: torch.tensor,
    dest: str,
    mean: list,
    std: list,
    device: str,
) -> None:
    """
    Performs saliency map

    Parameters
    ----------
    model : LitModel
        Model for activation maximation
    target_class : _
        Not needed
    org_image : torch.Tensor
        Mean image of dataset
    dest : str
        Path to destination folder
    mean : list
        Mean values for all three channels of the train dataset
    std : list
        Mean values for all three channels of the train dataset
    device : str
        Device that should be used ('cpu'/'cuda')
    """
    # Prep image
    prep_img = prep_image(org_image, mean, std)
    prep_img = prep_img.to(device)

    # Create SaliencyMap
    interpretor = SaliencyMap(model, [], [48, 48], None)

    # Creat Map
    endpoint = interpretor.interpret(prep_img)

    # Save map
    image_dest = Path(dest).joinpath(
        "heatmap_saliency", org_image.parents[0].name, org_image.name
    )
    image_dest.parents[0].mkdir(parents=True, exist_ok=True)
    # heatmap = rgb2gray(heatmap)

    plt.imsave(str(image_dest), endpoint.cpu().squeeze(0), cmap="gray")


def grad_cam(
    model: LitModel,
    _,
    org_image: torch.tensor,
    dest: str,
    mean: list,
    std: list,
    device: str,
) -> None:
    """
    Performs GradCam

    Parameters
    ----------
    model : LitModel
        Model for activation maximation
    target_class : _
        Not needed
    org_image : torch.Tensor
        Mean image of dataset
    dest : str
        Path to destination folder
    mean : list
        Mean values for all three channels of the train dataset
    std : list
        Mean values for all three channels of the train dataset
    device : str
        Device that should be used ('cpu'/'cuda')
    """
    # Prep image
    prep_img = prep_image(org_image, mean, std)
    prep_img = prep_img.to(device)

    # Create SmoothGradCAM
    wrapped_model = GradCAM(model, model.features[14])

    cam, _ = wrapped_model(prep_img)
    img = reverse_normalize(prep_img)

    # Create heatmap
    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    heatmap = 255 * cam.squeeze()
    heatmap = heatmap.cpu()

    # Save heatmap
    image_dest = Path(dest).joinpath(
        "heatmap_grad_cam", org_image.parents[0].name, org_image.name
    )
    image_dest.parents[0].mkdir(parents=True, exist_ok=True)
    heatmap = rgb2gray(heatmap)
    grayscale_uint8 = heatmap.astype(np.uint8)
    imsave(image_dest, grayscale_uint8)

    # Save image (overlay)
    heatmap = visualize(img.cpu(), cam.cpu())
    image_dest = Path(dest).joinpath(
        "grad_cam", org_image.parents[0].name, org_image.name
    )
    image_dest.parents[0].mkdir(parents=True, exist_ok=True)
    save_image(heatmap, str(image_dest))


def grad_cam_plus_plus(
    model: LitModel,
    _,
    org_image: torch.tensor,
    dest: str,
    mean: list,
    std: list,
    device: str,
) -> None:
    """
    Performs GradCam++

    Parameters
    ----------
    model : LitModel
        Model for activation maximation
    target_class : _
        Not needed
    org_image : torch.Tensor
        Mean image of dataset
    dest : str
        Path to destination folder
    mean : list
        Mean values for all three channels of the train dataset
    std : list
        Mean values for all three channels of the train dataset
    device : str
        Device that should be used ('cpu'/'cuda')
    """
    # Prep image
    prep_img = prep_image(org_image, mean, std)
    prep_img = prep_img.to(device)

    # Create SmoothGradCAMpp
    wrapped_model = SmoothGradCAMpp(
        model, model.features[14], n_samples=25, stdev_spread=0.15
    )

    cam, _ = wrapped_model(prep_img)
    img = reverse_normalize(prep_img)

    # Create heatmap
    _, _, H, W = img.shape
    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    heatmap = 255 * cam.squeeze()
    heatmap = heatmap.cpu()

    # Save heatmap
    image_dest = Path(dest).joinpath(
        "heatmap_grad_cam_pp", org_image.parents[0].name, org_image.name
    )
    image_dest.parents[0].mkdir(parents=True, exist_ok=True)
    heatmap = rgb2gray(heatmap)
    grayscale_uint8 = heatmap.astype(np.uint8)
    imsave(image_dest, grayscale_uint8)

    # Save image (overlay)
    heatmap = visualize(img.cpu(), cam.cpu())
    image_dest = Path(dest).joinpath(
        "grad_cam_pp", org_image.parents[0].name, org_image.name
    )
    image_dest.parents[0].mkdir(parents=True, exist_ok=True)
    save_image(heatmap, str(image_dest))


def generate_mean_image(images: list, device: str) -> torch.tensor:
    """
    Generates mean images based on passed images

    Parameters
    ----------
    images : list
        List with image paths
    device : str
        'cpu' or 'cuda'

    Returns
    -------
    torch.tensor
        Mean image
    """
    image_sum = torch.zeros(images[0].size())

    for image in images:
        image_sum += image

    mean_image = (image_sum / len(images)).to(device)
    mean_image = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min())

    return mean_image.to(device)


def main(args, vis_to_function, device):
    """
    Calls visualization method

    Parameters
    ----------
    args : argparse.namespace
        Namespace with argumetns
    vis_to_function : str
        Target visualization method
    device : str
        'cpu' or 'cuda'
    """

    # Load model
    model = LitModel.load_from_checkpoint(
        args.model,
        mean=args.mean,
        std=args.std,
        train_dataset=None,
        test_dataset=args.src,
    )
    model.to(device)
    model.eval()

    # Loop over all subfolders (label)
    num_dir = len([x for x in Path(args.src).iterdir() if x.is_dir()])
    for idx, directory in tqdm(
        enumerate([x for x in Path(args.src).iterdir() if x.is_dir()]), total=num_dir
    ):

        # Get target class
        target_class = directory.name

        # Select random images
        files = [x for x in directory.glob(f"*.{args.filetype}") if x.is_file()]

        # Run passed algos over the selected images
        for algo in args.vis:
            if algo == "Activation Maximation":
                # Because AM acts on the complete class, we wont call the method on each image
                # Instead of that, we create the class mean and pass that into the method
                data_list = []
                for image in files:
                    # Generate Tensors
                    prep_img = prep_image(image, args.mean, args.std)
                    data_list.append(prep_img)

                class_mean_img = generate_mean_image(data_list, device).to(device)
                activation_maximation(
                    model,
                    directory.name,
                    class_mean_img,
                    args.dest,
                    args.mean,
                    args.std,
                    device,
                )

            else:
                for image in files:
                    globals()[vis_to_function.get(algo)](
                        model,
                        target_class,
                        image,
                        args.dest,
                        args.mean,
                        args.std,
                        device,
                    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    possible_algos = ["Activation Maximation", "Saliency", "GradCam", "GradCam++"]
    vis_to_function = {
        "GradCam": "grad_cam",
        "GradCam++": "grad_cam_plus_plus",
        "Saliency": "salience_map",
        "Activation Maximation": "activation_maximation",
    }
    combinations = get_algo_combinations(possible_algos)
    args = parse_args(combinations)
    main(args, vis_to_function, device)
