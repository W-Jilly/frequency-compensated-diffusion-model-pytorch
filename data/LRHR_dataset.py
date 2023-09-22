from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import h5py, os
import numpy as np
import torch
import copy

from data.HazeAug import rt_haze_enhancement


def neibor_16_mul(num, size=32):
    a = num // size
    b = num % size
    if b >= 0.5 * size:
        return size * (a + 1)
    else:
        return size * a


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False,
                 other_params=None):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.down_sample = other_params['down_sample'] if "down_sample" in other_params.keys() else None
        self.real_hr_path = other_params['hr_path'] if "hr_path" in other_params.keys() else None

        # rt daRESIDE_img_syntheic
        self.rt_da = other_params['HazeAug'] if "HazeAug" in other_params.keys() else None
        if self.rt_da:
            self.rt_da_ref = other_params['rt_da_ref']
            self.ref_imgs = []
            for dir in self.rt_da_ref:
                self.ref_imgs += [os.path.join(dir, i) for i in os.listdir(dir)]
            self.depth_path = other_params['depth_img_path']

        if datatype in ["haze_img"]:
            self.sr_path = Util.get_paths_from_images("{}/HR_hazy".format(dataroot))

            self.hr_path = Util.get_paths_from_images("{}/HR".format(dataroot))
            self.dataset_len = len(self.hr_path)

            self.dis_prefix = other_params['distanse_prefix'] if "distanse_prefix" in other_params.keys() else None
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        elif datatype in ["RESIDE_img_syntheic"]:

            self.sr_path = Util.get_paths_from_images(dataroot)
            self.hr_path = self.sr_path
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype in ["RESIDE_img_syntheic"]:

            if self.rt_da:

                img_SR = rt_haze_enhancement(
                    self.sr_path[index],
                    os.path.join(self.depth_path, "{}.png".format(self.sr_path[index].split("/")[-1].split("_")[0])),
                    ref_path=np.random.choice(self.ref_imgs)
                )
            else:
                img_SR = Image.open(self.sr_path[index]).convert("RGB")

            img_SR = img_SR.resize((self.r_res, self.r_res))

            # hr_path
            hr_path = "{}/{}.png".format(
                self.real_hr_path,
                self.sr_path[index].split("/")[-1].split("_")[0]
                )
            img_HR = Image.open(hr_path).convert("RGB")
            img_HR = img_HR.resize((self.r_res, self.r_res))

            if self.need_LR:
                img_LR = img_SR

        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.sr_path[index]).convert("RGB")

            if self.down_sample is not None:
                img_HR = self.resize(img_HR)
                img_SR = self.resize(img_SR)
                img_LR = self.resize(img_LR)

                if self.dis_prefix != None:
                    img_depth = self.resize(img_depth)

            else:
                img_HR = self.resize_to_resolution(img_HR)
                img_SR = self.resize_to_resolution(img_SR)
                img_LR = self.resize_to_resolution(img_LR)

        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))

            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))

            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

    def resize(self, input_image):
        H, W = np.shape(input_image)[:2]
        resize_H, resize_W = neibor_16_mul(int(H / self.down_sample)), neibor_16_mul(int(W / self.down_sample))
        out_image = input_image.resize((resize_W, resize_H))
        return out_image

    def resize_to_resolution(self, input_image):
        out_image = input_image.resize((self.r_res, self.r_res))
        return out_image
