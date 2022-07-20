import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from utils import (
    list_documents_path,
    get_bbox_inside_image,
    create_dir,
)

if __name__ == "__main__":
    export_dir = "../pngexport/"
    root_dir = "../data_test/"
    
    # create export_dir if not present
    create_dir(export_dir)

    # create export_dir/docs if not present
    absolute_img_path = os.path.join(export_dir, "docs")
    create_dir(absolute_img_path)

    # init coco vars
    images = []
    annotations = []
    target_w = 640
    target_h = 360

    annotation_paths = list_documents_path(root_dir)
    print("Converting to coco.")
    for ind, rel_annotation_path in enumerate(tqdm(annotation_paths)):
        # get image path
        rel_image_path = rel_annotation_path.replace("ground_truth", "images")
        rel_image_path = rel_image_path.replace("json", "tif")

        # load image
        abs_image_path = os.path.join(root_dir, rel_image_path)
        raw_image = cv2.imread(abs_image_path)
        
        # resize it and store it at images path
        image = cv2.resize(raw_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(absolute_img_path, f'{ind}.png'), image)
        
        # prepare image info
        image_dict = dict()
        image_dict["file_name"] = f'{ind}.png'
        image_dict["height"] = image.shape[0]
        image_dict["width"] = image.shape[1]
        image_dict["id"] = ind
        # add image info
        images.append(image_dict)

        # form image regions
        image_xmin = 0
        image_xmax = image.shape[1]
        image_ymin = 0
        image_ymax = image.shape[0]
        image_bbox = [image_xmin, image_ymin, image_xmax, image_ymax]

        # load mask coords
        abs_annotation_path = os.path.join(root_dir, rel_annotation_path)
        info = json.load(open(abs_annotation_path, "r", errors="ignore"))
        
        # fill mask with rectangles
        mask = np.zeros(image.shape, dtype=np.uint8)
        ratio_x = raw_image.shape[1]/image.shape[1]
        ratio_y = raw_image.shape[0]/image.shape[0]
        for label_tag in info:
            mask_coords = []
            for coord in info[label_tag]["quad"]:
                mask_coords.append([coord[0]/ratio_x, coord[1]/ratio_y])
            #mask_coords = [[round(h/3) for h,w in coord] for coord in info[label_tag]["quad"]]
            # create mask from poly coords
            mask_coords_np = np.array(mask_coords, dtype=np.int32)
            cv2.fillPoly(mask, mask_coords_np.reshape(-1, 4, 2), color=(255, 255, 255))

            # get voc style bounding box coordinates [minx, miny, maxx, maxy] of the mask
            label_xmin = min([pos[0] for pos in mask_coords])
            label_xmax = max([pos[0] for pos in mask_coords])
            label_ymin = min([pos[1] for pos in mask_coords])
            label_ymax = max([pos[1] for pos in mask_coords])
            label_bbox = [label_xmin, label_ymin, label_xmax, label_ymax]
            label_bbox = get_bbox_inside_image(label_bbox, image_bbox)

            # calculate coco style bbox coords [minx, miny, width, height] and area
            label_area = int(
                (label_bbox[2] - label_bbox[0]) * (label_bbox[3] - label_bbox[1])
            )
            label_bbox = [
                label_bbox[0],
                label_bbox[1],
                label_bbox[2] - label_bbox[0],
                label_bbox[3] - label_bbox[1],
            ]

            # prepare annotation info
            annotation_dict = dict()
            annotation_dict["iscrowd"] = 0
            annotation_dict["image_id"] = image_dict["id"]
            if(label_tag == "photo"):
                annotation_dict["category_id"] = 2
            elif(label_tag == "signature"):
                annotation_dict["category_id"] = 3
            else:
                annotation_dict["category_id"] = 1
                isfield = label_tag[:-2]
                if(isfield != "field"):
                    print(label_tag)
            annotation_dict["ignore"] = 0
            annotation_dict["id"] = ind

            annotation_dict["bbox"] = label_bbox
            annotation_dict["area"] = label_area
            annotation_dict["segmentation"] = [
                [single_coord for coord_pair in mask_coords for single_coord in coord_pair]
            ]
            # add annotation info
            annotations.append(annotation_dict)

        # store mask resized
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        # #mask_resized = cv2.resize(mask, target_shape, interpolation=cv2.INTER_AREA)
        # absolute_mask_path = os.path.join(export_dir, "mask")
        # create_dir(absolute_mask_path)
        # cv2.imwrite(os.path.join(absolute_mask_path, f'{ind}_mask.png'), mask)

    # combine lists and form coco dict
    coco_dict = dict()
    coco_dict["images"] = images
    coco_dict["annotations"] = annotations
    coco_dict["categories"] = [{"name": "field", "id": 1},{"name": "photo", "id": 2},{"name": "signature", "id": 3}]

    # export coco dict
    absolute_coco_path = os.path.join(export_dir, "coco")
    create_dir(absolute_coco_path)
    export_path = os.path.join(absolute_coco_path, "dataset.json")
    with open(export_path, "w") as f:
        json.dump(coco_dict, f)

