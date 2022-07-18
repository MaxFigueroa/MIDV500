import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from utils import (
    list_annotation_paths_recursively,
    create_dir,
)


def tif2png(root_dir: str, export_dir: str, typename: str):
    """
    Walks inside root_dir (should oly contain original midv500 dataset folders),
    reads all annotations, and creates coco styled annotation file
    named as midv500_coco.json saved to export_dir.
    Sample inputs:
        root_dir: ~/data/midv500/
        export_dir: ~/data/
    """

    # raise error if export_dir is given as a json file path
    if "json" in export_dir:
        raise ValueError("export_dir should be a directory, not a file path!")

    # create export_dir if not present
    create_dir(export_dir)

    # create export_dir/images if not present
    absolute_img_path = os.path.join(export_dir, "images")
    create_dir(absolute_img_path)

    # init coco vars
    target_shape = (360, 640)

    annotation_paths = list_annotation_paths_recursively(root_dir, False)
    print("Converting to coco.")
    for ind, rel_annotation_path in enumerate(tqdm(annotation_paths)):
        # get image path
        rel_image_path = rel_annotation_path.replace("ground_truth", "images")
        rel_image_path = rel_image_path.replace("json", "tif")

        # just load image
        abs_image_path = os.path.join(root_dir, rel_image_path)
        image = cv2.imread(abs_image_path, cv2.IMREAD_UNCHANGED)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(absolute_img_path, f'{ind}.png'), image)

if __name__ == "__main__":
    # construct the argument parser
    ap = argparse.ArgumentParser()

    # add the arguments to the parser
    ap.add_argument(
        "root_dir",
        default="../data/",
        help="Directory of the downloaded MIDV-500 dataset.",
    )
    ap.add_argument(
        "export_dir", default="../coco/", help="Directory for coco file to be exported."
    )
    #args = vars(ap.parse_args())

    args = dict() 
    args["root_dir"] = "../data/46_ury_passport/"
    args["export_dir"] = "../pngexport/"

    # from tif to png
    tif2png(args["root_dir"], args["export_dir"], "simple")