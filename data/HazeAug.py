import os
from PIL import Image
import numpy as np
import cv2
from data.FDA import trans_image_by_ref

from PIL import Image, ImageFilter

depth_argu = False


def depth_change(depth):
    depth_strategy = np.random.uniform(0, 1)

    if 0.4 <= depth_strategy < 0.7:
        strategy = 'gamma'
    elif 0.7 <= depth_strategy < 1.0:
        strategy = 'normalize'
    else:
        strategy = 'identity'

    if strategy == "gamma":
        factor = np.random.uniform(0.2, 1.8)

        depth = np.array(depth ** factor)

    elif strategy == "normalize":
        # normalize float versions
        factor_alpha = np.random.uniform(0, 0.4)
        factor_beta = np.random.uniform(0, 2)
        depth = cv2.normalize(depth, None, alpha=factor_alpha, beta=factor_beta, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_32F)

    return depth


class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=1, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


def rt_haze_enhancement(pil_img, depth_path, ref_path):
    # add_haze
    A = np.random.rand() * 1.3 + 0.5
    beta = 2 * np.random.rand() + 0.8
    color_strategy = np.random.rand()
    if color_strategy <= 0.5:
        strategy = 'colour_cast'
    #     elif 0.3 < color_strategy <= 0.6:
    #         strategy = 'luminance'
    else:
        strategy = 'add_hazy'

    img = cv2.imread(pil_img)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) / 256.0  # + 1e-7

    if depth_argu == False:
        depth = depth_change(depth)

    img_f = img / 255.0  # 归一化

    td_bk = np.exp(- np.array(depth) * beta)
    td_bk = np.expand_dims(td_bk, axis=-1).repeat(3, axis=-1)
    img_bk = np.array(img_f) * td_bk + A * (1 - td_bk)

    img_bk = img_bk / np.max(img_bk) * 255
    img_bk = img_bk[:, :, ::-1]

    if strategy == 'colour_cast':
        img_bk = Image.fromarray(np.uint8(img_bk))  # .covert('RGB')
        img_bk = trans_image_by_ref(
            in_path=img_bk,
            ref_path=ref_path,
            value=np.random.rand() * 0.002 + 0.0001
        )

    if strategy == 'luminance':
        img_bk = np.power(img_bk, 0.95)  # 对像素值指数变换
        img_bk = Image.fromarray(np.uint8(img_bk))  # .covert('RGB')

    else:
        img_bk = Image.fromarray(np.uint8(img_bk))  # .covert('RGB')

    img_bk = img_bk.filter(ImageFilter.SMOOTH_MORE)

    return img_bk
