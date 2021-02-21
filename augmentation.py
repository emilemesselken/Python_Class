import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


ia.seed(4)

def start_aug(images, y):
    seq = iaa.Sequential([
    # iaa.Affine(rotate=(-25, 25)),
    iaa.Fliplr(p=1.0),
    # iaa.Crop(percent=(0, 0.4))
    ], random_order=True)

    y_aug_res = []
    images_aug_res = []
    for idx, image in enumerate(images):
        images_aug = [seq(image=image) for _ in range(8)]
        for img in images_aug:
            y_aug_res.append(y[idx])
            images_aug_res.append(img)

    images_aug_res = np.array(images_aug_res)
    y_aug_res = np.array(y_aug_res)

    a = np.append(images, images_aug_res, axis=0)
    b = np.append(y, y_aug_res)

    return a, b