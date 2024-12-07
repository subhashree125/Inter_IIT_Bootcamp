import glob
import random
import json
import os
import six

import cv2
import numpy as np
from tqdm import tqdm
from time import time
from functions import get_image_array, get_pairs_from_paths, get_segmentation_array

weights_path = 'checkpoints/model.weights.h5'
n_classes = 27
input_height = 416
input_width = 608
epochs = 5

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 255]

class_colors = [
    (128, 64, 128),    # Road
    (250, 170, 160),   # Parking
    (81, 0, 81),       # Drivable fallback
    (244, 35, 232),    # Sidewalk
    (230, 150, 140),   # Rail track
    (152, 251, 152),   # Non-drivable fallback
    (220, 20, 60),     # Person
    (246, 198, 145),   # Animal
    (255, 0, 0),       # Rider
    (0, 0, 230),       # Motorcycle
    (119, 11, 32),     # Bicycle
    (255, 204, 54),    # Autorickshaw
    (0, 0, 142),       # Car
    (0, 0, 70),        # Truck
    (0, 60, 100),      # Bus
    (0, 0, 90),        # Caravan
    (0, 0, 110),       # Trailer
    (0, 80, 100),      # Train
    (136, 143, 153),   # Vehicle fallback
    (220, 190, 40),    # Curb
    (102, 102, 156),   # Wall
    (190, 153, 153),   # Fence
    (180, 165, 180),   # Guard rail
    (174, 64, 67),     # Billboard
    (220, 220, 0),     # Traffic sign
    (250, 170, 30),    # Traffic light
    (0, 0, 0)          # Unlabeled (255)
]


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height, output_width = seg_arr.shape
    seg_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for c in range(n_classes-1):
        seg_arr_c = seg_arr == c
        seg_img[:, :, 0] += (seg_arr_c * colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += (seg_arr_c * colors[c][1]).astype('uint8')
        seg_img[:, :, 2] += (seg_arr_c * colors[c][2]).astype('uint8')

    return seg_img

def get_legends(class_names, colors=class_colors):
    n_classes = len(class_names)
    legend = np.full(((n_classes * 25) + 25, 125, 3), 255, dtype="uint8")

    for i, (class_name, color) in enumerate(zip(class_names[:n_classes], colors[:n_classes])):
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, i * 25), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend
def overlay_seg_image(inp_img, seg_img):
    original_h, original_w = inp_img.shape[:2]
    seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    fused_img = (inp_img / 2 + seg_img / 2).astype('uint8')
    return fused_img

def concat_legends(seg_img, legend_img):
    new_h = max(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]
    out_img = np.full((new_h, new_w, 3), legend_img[0, 0, 0], dtype='uint8')
    out_img[:legend_img.shape[0], :legend_img.shape[1]] = legend_img
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = seg_img
    return out_img

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):
    if n_classes is None:
        n_classes = np.max(seg_arr) + 1  # Assuming classes start from 0

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        original_h, original_w = inp_img.shape[:2]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if prediction_height and prediction_width:
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height), interpolation=cv2.INTER_NEAREST)
        if inp_img is not None:
            inp_img = cv2.resize(inp_img, (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None, "Input image must be provided for overlay."
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None, "Class names must be provided to show legends."
        legend_img = get_legends(class_names, colors=colors)
        seg_img = concat_legends(seg_img, legend_img)

    return seg_img

def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None,
            read_image_type=1):
    
    assert inp is not None, "Input must be provided."
    assert isinstance(inp, (np.ndarray, six.string_types)), \
        "Input should be a NumPy array or a file path string."

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, read_image_type)
        assert inp is not None, f"Image at path {inp} could not be loaded."

    assert inp.ndim in [1, 3, 4], "Image should have 1, 3, or 4 dimensions."

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = visualize_segmentation(
        pr, inp, n_classes=n_classes, colors=colors, overlay_img=overlay_img,
        show_legends=show_legends, class_names=class_names,
        prediction_width=prediction_width, prediction_height=prediction_height
    )

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return seg_img

def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None, read_image_type=1):

    if inps is None and inp_dir is not None:
        inps = sorted(
            glob.glob(os.path.join(inp_dir, "*.jpg")) +
            glob.glob(os.path.join(inp_dir, "*.png")) +
            glob.glob(os.path.join(inp_dir, "*.jpeg"))
        )

    assert isinstance(inps, list), "Inputs should be provided as a list."

    all_prs = []

    if out_dir is not None and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, inp in enumerate(tqdm(inps, desc="Predicting multiple images")):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                # Extract the 'frameXXXX' part from 'frameXXXX_leftImg8bit'
                file_name = os.path.basename(inp)
                frame_id = file_name.split('_leftImg8bit')[0]  # Extract 'frameXXXX'
                out_fname = os.path.join(out_dir, f"{frame_id}.png")  # Save as 'frameXXXX.jpg'
            else:
                out_fname = os.path.join(out_dir, f"{i}.png")  # Fallback for non-string inputs

        pr = predict(
            model, inp, out_fname,
            overlay_img=overlay_img, class_names=class_names,
            show_legends=show_legends, colors=colors,
            prediction_width=prediction_width,
            prediction_height=prediction_height,
            read_image_type=read_image_type
        )
        all_prs.append(pr)

    return all_prs
