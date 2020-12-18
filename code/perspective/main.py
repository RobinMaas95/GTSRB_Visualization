import argparse
import os
import sys
from distutils.util import strtobool
from pathlib import Path

from tqdm import tqdm

from image_transformer import ImageTransformer
from util import save_image

# Usage:
#     Change main function with ideal arguments
#     then
#     python demo.py [name of the image] [degree to rotate] ([ideal width] [ideal height])
#     e.g.,
#     python demo.py images/000001.jpg 360
#     python demo.py images/000001.jpg 45 500 700
#
# Parameters:
#     img_path  : the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : the rotation around the x axis
#     phi       : the rotation around the y axis
#     gamma     : the rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image


def parse_args():
    # Handle args parsing
    parser = argparse.ArgumentParser(description="Rotate along X axis")
    parser.add_argument(
        "--dest",
        dest="dest",
        metavar="Destination",
        required=True,
        type=str,
        help="Destination path, subfolder structure will be restored",
    )
    parser.add_argument(
        "--src",
        dest="src",
        metavar="Source",
        type=str,
        help="Path to source directory, script runs over subfolders",
    )

    parser.add_argument(
        "--degree",
        metavar="Degree",
        dest="degree",
        type=int,
        help="Up to how many degrees should the image be rotated (x-axis)",
    )
    parser.add_argument(
        "--negative",
        metavar="Negative",
        dest="negative",
        type=bool,
        default=False,
        help="Also rotate the image in the negative direction?",
    )

    args = parser.parse_args()

    return args


def loop(start, end, it, directory, image):
    for ang in range(start, end, 1):
        Path(args.dest).joinpath(f"{str(ang).zfill(3)}_degree", directory.name).mkdir(
            parents=True, exist_ok=True
        )
        rotated_img = it.rotate_along_axis(phi=ang, dx=0)
        save_image(
            str(
                Path(args.dest).joinpath(
                    f"{str(ang).zfill(3)}_degree", directory.name, image.name
                )
            ),
            rotated_img,
        )


def main(args):
    # Iterate through rotation range
    start = 0
    end = args.degree
    neg_start = 360 - args.degree
    neg_end = 360

    num_dir = len([x for x in Path(args.src).iterdir() if x.is_dir()])
    for idx, directory in tqdm(
        enumerate([x for x in Path(args.src).iterdir() if x.is_dir()])
    ):
        files = [x for x in directory.glob(f"*.ppm") if x.is_file()]
        for image in files:
            # Make output dir
            it = ImageTransformer(str(image), None)
            # Rotate in positive degrees
            loop(start, end, it, directory, image)

            # If negative is true, rotate in negative direction too
            if args.negative:
                loop(neg_start, neg_end, it, directory, image)


if __name__ == "__main__":
    args = parse_args()
    main(args)
