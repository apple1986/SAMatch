import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from . import augs_TIBA as img_trsform
from dataloaders.transform import obtain_cutmix_box
class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_list.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val_list.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        case_1 = case.split(' ')[0]
        case_2 = case.split(' ')[1]
        if self.split == "train":
            image_train = cv2.imread(self._base_dir + "/{}".format(case_1), cv2.IMREAD_GRAYSCALE)
            image_label = cv2.imread(self._base_dir + "/{}".format(case_2), cv2.IMREAD_GRAYSCALE)
            # image_train = np.array(image_train)
            # image_label = np.array(image_label)
            #image_train = np.transpose(image_train, (2, 0, 1))
        else:
            image_train = cv2.imread(self._base_dir + "/{}".format(case_1), cv2.IMREAD_GRAYSCALE)
            image_label = cv2.imread(self._base_dir + "/{}".format(case_2), cv2.IMREAD_GRAYSCALE)
            #image_train = np.transpose(image_train, (2, 0, 1))
            # image_train = np.array(image_train)
            # image_label = np.array(image_label)
        image = image_train
        label = image_label

        
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class uni_BUSI(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_list.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val_list.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        case_1 = case.split(' ')[0]
        case_2 = case.split(' ')[1]
        if self.split == "train":
            image_train = cv2.imread(self._base_dir + "/{}".format(case_1), cv2.IMREAD_GRAYSCALE)
            image_label = cv2.imread(self._base_dir + "/{}".format(case_2), cv2.IMREAD_GRAYSCALE)
            # image_train = np.array(image_train)
            # image_label = np.array(image_label)
            #image_train = np.transpose(image_train, (2, 0, 1))
        else:
            image_train = cv2.imread(self._base_dir + "/{}".format(case_1), cv2.IMREAD_GRAYSCALE)
            image_label = cv2.imread(self._base_dir + "/{}".format(case_2), cv2.IMREAD_GRAYSCALE)
            #image_train = np.transpose(image_train, (2, 0, 1))
            # image_train = np.array(image_train)
            # image_label = np.array(image_label)
        image = image_train
        label = image_label
        
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample
    
    
    
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)




class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = (output_size,output_size)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = image, label
        if random.random() < 0.5:
            image_weak = random_rot_flip(image_weak)
        image_strong = image_weak
        if random.random() < 0.8:
            image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        else:
            image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        image_strong_1 = image_weak
        if random.random() < 0.8:
            image_strong_1 = color_jitter(image_weak).type("torch.FloatTensor")
        else:
            image_strong_1 = torch.from_numpy(image_strong_1.astype(np.float32)).unsqueeze(0)
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        #######################################
        cutmix_box1 = obtain_cutmix_box(self.output_size[0], p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.output_size[1], p=0.5)
        #######################################
        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong":  image_strong,#torch.from_numpy(image_strong).unsqueeze(0),
            "label": label,
            "image_strong_1": image_strong_1,
            "cutmix_w":cutmix_box1,
            "cutmix_s":cutmix_box2
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)




def butterworth_filter(image, D0=30, n=4, high_pass=True):
    # 获取图像的行数和列数
    rows, cols = image.shape

    # 创建一个中心在图像中心的频域滤波器
    u = np.arange(-cols // 2, cols // 2)
    v = np.arange(-rows // 2, rows // 2)
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U**2 + V**2)

    # 根据滤波器类型创建滤波函数
    if high_pass:
        H = 1 / (1 + (D0 / D)**(2 * n))
    else:
        H = 1 / (1 + (D / D0)**(2 * n))

    # 应用滤波器到傅里叶变换后的图像
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    filtered_transform = f_transform_shifted * H
    filtered_image = np.fft.ifftshift(filtered_transform)
    filtered_image = np.fft.ifft2(filtered_image)
    filtered_image = np.abs(filtered_image)
    filtered_image = np.float32(filtered_image)

    return filtered_image


def nearest_neighbor_downsample(input_array, scale_factor):
    if scale_factor <= 0:
        raise ValueError("Scale factor must be greater than 0.")
    
    input_height, input_width = input_array.shape
    output_height = input_height // scale_factor
    output_width = input_width // scale_factor

    output_array = np.zeros((output_height, output_width), dtype=input_array.dtype)

    for i in range(output_height):
        for j in range(output_width):
            y = i * scale_factor
            x = j * scale_factor
            output_array[i, j] = input_array[y, x]

    return output_array.astype(np.float32)