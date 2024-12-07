import itertools
import os
import random
import numpy as np
import cv2
from augmentation import augment_seg

DATA_LOADER_SEED = 0
random.seed(DATA_LOADER_SEED)

ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp", ".jpg"]

class DataLoaderError(Exception):
    pass

def get_image_list_from_path(images_path):
    return [
        os.path.join(images_path, dir_entry) for dir_entry in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, dir_entry)) and
        os.path.splitext(dir_entry)[1].lower() in ACCEPTABLE_IMAGE_FORMATS
    ]

def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Match images and segmentations from given paths. """
    image_files = [
        (os.path.splitext(dir_entry)[0], os.path.join(images_path, dir_entry))
        for dir_entry in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, dir_entry)) and
        os.path.splitext(dir_entry)[1].lower() in ACCEPTABLE_IMAGE_FORMATS
    ]

    segmentation_files = {
        os.path.splitext(dir_entry)[0]: os.path.join(segs_path, dir_entry)
        for dir_entry in os.listdir(segs_path)
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and
        os.path.splitext(dir_entry)[1].lower() in ACCEPTABLE_SEGMENTATION_FORMATS
    }

    return [
        (image_full_path, segmentation_files[file_name]) for file_name, image_full_path in image_files
        if file_name in segmentation_files
    ]

def get_image_array(image_input, width, height, imgNorm="sub_mean", read_image_type=1):
    """ Load image array from input. """
    img = cv2.imread(image_input, read_image_type) if isinstance(image_input, str) else image_input
    if img is None:
        raise DataLoaderError(f"Image not found: {image_input}")

    img = cv2.resize(img, (width, height)).astype(np.float32)
    if imgNorm == "sub_and_divide":
        return img / 127.5 - 1
    elif imgNorm == "sub_mean":
        means = [103.939, 116.779, 123.68]
        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]
        return img[:, :, ::-1]
    elif imgNorm == "divide":
        return img / 255.0
    return img

def get_segmentation_array(image_input, nClasses, width, height, no_reshape=False, read_image_type=1):
    """ Load segmentation array from input. """
    img = cv2.imread(image_input, read_image_type) if isinstance(image_input, str) else image_input
    if img is None:
        raise DataLoaderError(f"Segmentation not found: {image_input}")

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)[:, :, 0]
    seg_labels = np.zeros((height, width, nClasses))
    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    return np.reshape(seg_labels, (width * height, nClasses)) if not no_reshape else seg_labels

def image_segmentation_generator(images_path, segs_path=None, batch_size=16, n_classes=27,
                                 input_height=416, input_width=608, output_height=416,
                                 output_width=608, do_augment=False, augmentation_name="aug_all",
                                 preprocessing=None, read_image_type=cv2.IMREAD_COLOR, ignore_segs=False):
    """ Generates image and segmentation batches for training or validation. """
    
    if not ignore_segs:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        img_seg_pairs = iter(img_seg_pairs)
    else:
        img_list_gen = iter(get_image_list_from_path(images_path))

    while True:
        X_batch = []
        Y_batch = []
        
        for _ in range(batch_size):
            try:
                if ignore_segs:
                    im_path = next(img_list_gen)
                    im = cv2.imread(im_path, read_image_type)
                    seg = None
                else:
                    im_path, seg_path = next(img_seg_pairs)
                    im = cv2.imread(im_path, read_image_type)
                    seg = cv2.imread(seg_path, 1)

                if do_augment and seg is not None:
                    im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0], augmentation_name)

                if preprocessing:
                    im = preprocessing(im)

                X_batch.append(get_image_array(im, input_width, input_height))
                if seg is not None:
                    Y_batch.append(get_segmentation_array(seg, n_classes, output_width, output_height))

            except StopIteration:
                if not ignore_segs:
                    img_seg_pairs = iter(get_pairs_from_paths(images_path, segs_path))
                else:
                    img_list_gen = iter(get_image_list_from_path(images_path))
                break
            
        if len(X_batch) > 0:
            if not ignore_segs:
                yield (np.array(X_batch), np.array(Y_batch))
            else:
                yield np.array(X_batch)

