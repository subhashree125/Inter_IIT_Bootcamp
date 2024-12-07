import numpy as np

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except ImportError:
    print("Error in loading augmentation, can't import imgaug. Please make sure it is installed.")

IMAGE_AUGMENTATION_NUM_TRIES = 10
IMAGE_AUGMENTATION_SEQUENCE = None
loaded_augmentation_name = ""

def _load_augmentation_aug_geometric():
    return iaa.OneOf([
        iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.2)]),
        iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode='constant', pad_cval=(0, 255)),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            mode='constant',
            cval=(0, 255),
        )
    ])

def _load_augmentation_aug_non_geometric():
    return iaa.Sequential([
        iaa.Sometimes(0.3, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
        iaa.Sometimes(0.2, iaa.JpegCompression(compression=(70, 99))),
        iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Sometimes(0.2, iaa.MotionBlur(k=15, angle=[-45, 45])),
        iaa.Sometimes(0.2, iaa.MultiplyHue((0.5, 1.5))),
        iaa.Sometimes(0.2, iaa.MultiplySaturation((0.5, 1.5))),
        iaa.Sometimes(0.34, iaa.Grayscale(alpha=(0.0, 1.0))),
        iaa.Sometimes(0.1, iaa.GammaContrast((0.5, 2.0))),
        iaa.Sometimes(0.1, iaa.CLAHE()),
        iaa.Sometimes(0.1, iaa.HistogramEqualization()),
        iaa.Sometimes(0.2, iaa.LinearContrast((0.5, 2.0), per_channel=0.5)),
        iaa.Sometimes(0.1, iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)))
    ])

def _load_augmentation_aug_all():
    """Load image augmentation model"""
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode='constant', pad_cval=(0, 255))),
        iaa.SomeOf((0, 5), [
            iaa.Sometimes(0.3, iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11)),
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Invert(0.05, per_channel=True),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.AddToHueAndSaturation((-20, 20)),
            iaa.Grayscale(alpha=(0.0, 1.0)),
        ], random_order=True)
    ], random_order=True)

augmentation_functions = {
    "aug_all": _load_augmentation_aug_all,
    "aug_geometric": _load_augmentation_aug_geometric,
    "aug_non_geometric": _load_augmentation_aug_non_geometric
}

def _load_augmentation(augmentation_name="aug_all"):
    global IMAGE_AUGMENTATION_SEQUENCE

    if augmentation_name not in augmentation_functions:
        raise ValueError("Augmentation name not supported")

    IMAGE_AUGMENTATION_SEQUENCE = augmentation_functions[augmentation_name]()

def _augment_seg(img, seg, augmentation_name="aug_all", other_imgs=None):
    global loaded_augmentation_name

    if (not IMAGE_AUGMENTATION_SEQUENCE) or (augmentation_name != loaded_augmentation_name):
        _load_augmentation(augmentation_name)
        loaded_augmentation_name = augmentation_name

    aug_det = IMAGE_AUGMENTATION_SEQUENCE.to_deterministic()
    image_aug = aug_det.augment_image(img)

    if other_imgs is not None:
        image_aug = [image_aug]
        for other_img in other_imgs:
            image_aug.append(aug_det.augment_image(other_img))

    segmap = ia.SegmentationMapsOnImage(seg, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap).get_arr()

    return image_aug, segmap_aug

def augment_seg(img, seg, augmentation_name="aug_all", other_imgs=None):
    attempts = 0
    while attempts < IMAGE_AUGMENTATION_NUM_TRIES:
        try:
            return _augment_seg(img, seg, augmentation_name, other_imgs)
        except Exception:
            attempts += 1

    return _augment_seg(img, seg, augmentation_name, other_imgs)

