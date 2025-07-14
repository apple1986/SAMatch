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
            with open(self._base_dir + "/train.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = np.load(self._base_dir + "/data/train_npz/{}.npz".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/val/{}.npy.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
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


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        # image_strong = image_weak
        image_strong = image_weak
        if random.random() < 0.8:
            image_strong = augmentations.cta_apply(image_weak, ops_strong)
        # else:
        #     image_strong = torch.from_numpy(image_strong).astype(np.float32).unsqueeze(0)
        #if random.random() < 0.8:
        image_strong_1 = augmentations.cta_apply(image_weak, ops_strong)
        #else:
        #    image_strong_1 = torch.from_numpy(image_strong_1.astype(np.float32)).unsqueeze(0)
        #trs_form_strong = img_trsform.strong_img_aug(10,True)
        #image_strong_1 = enhance_image_with_fft(to_tensor(image_weak))
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
            "image_strong_1": to_tensor(image_strong_1),
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


def enhance_image_with_fft(image, d=10):
    # 进行傅里叶变换
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    if len(image.shape)==3:
        image = image.squeeze(0)
    # 设计高通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d] = 0

    # 进行逆傅里叶变换
    f_transform = np.fft.ifftshift(f_transform_shifted)
    image_enhanced = np.fft.ifft2(f_transform) 
    image_enhanced = np.abs(image_enhanced)
    if len(image_enhanced.shape)==2:
        image_enhanced = np.expand_dims(image_enhanced,axis=0)
    return image_enhanced

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
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = image, label
        if random.random() < 0.5:
            image_weak, label = random_rot_flip(image_weak, label)
        # strong augmentation is color jitter

        #image_strong_1 = structure_tensor_enhancement(image_weak)
        #trs_form_strong = img_trsform.strong_img_aug(10,True)
        #image_weak_t = transforms.ToPILImage()(image_weak.astype(np.float32)).convert("L")
        #pil_image = transforms.ToPILImage()(tensor_image)
        #image_strong = trs_form_strong(image_weak_t)
        #image_strong = transforms.ToTensor()(image_strong)
        #image_strong = nearest_neighbor_downsample(image_weak,4)
        #image_strong_1 = torch.from_numpy(enhance_image_with_fft(image_weak,30)).type("torch.FloatTensor")
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

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong":  image_strong,#torch.from_numpy(image_strong).unsqueeze(0),
            "label_aug": label,
            "image_strong_1": image_strong_1
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

def structure_tensor_enhancement(image_tensor, window_size=3, alpha=1.0, epsilon=1e-6):
    """
    使用结构张量分析来增强图像的纹理和边缘。

    参数：
    - image_tensor: 输入图像的Tensor表示，单通道灰度图像，形状为 (1, 1, H, W)
    - window_size: 结构张量计算中的窗口大小，默认为3
    - alpha: 增强强度参数，默认为1.0

    返回值：
    - enhanced_image_tensor: 增强后的图像的Tensor表示
    """
    # 计算图像的梯度
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    kernel_y = kernel_x.t()
    gradient_x = F.conv2d(image_tensor, kernel_x.view(1, 1, 3, 3), padding=1)
    gradient_y = F.conv2d(image_tensor, kernel_y.view(1, 1, 3, 3), padding=1)

    # 计算结构张量的分量
    Ixx = gradient_x ** 2
    Iyy = gradient_y ** 2
    Ixy = gradient_x * gradient_y

    # 计算结构张量的特征值
    window = torch.ones(1, 1, window_size, window_size).to(image_tensor.device)
    Ixx_sum = F.conv2d(Ixx, window, padding=window_size // 2)
    Iyy_sum = F.conv2d(Iyy, window, padding=window_size // 2)
    Ixy_sum = F.conv2d(Ixy, window, padding=window_size // 2)

    det = (Ixx_sum * Iyy_sum) - (Ixy_sum ** 2)
    trace = Ixx_sum + Iyy_sum

    # 添加容错处理，避免分母或分子为零的情况
    det = torch.clamp(det, min=epsilon)
    trace = torch.clamp(trace, min=epsilon)
   
    # 使用特征值信息增强图像
    enhanced_image_tensor = image_tensor + alpha * (trace / det) * image_tensor

    # 将像素值限制在0到1之间
    enhanced_image_tensor = torch.clamp(enhanced_image_tensor, 0, 1).squeeze(0)

    return enhanced_image_tensor



def enhance_image_with_fft(image, d=10):
        # 进行傅里叶变换
        f_transform = np.fft.fft2(image)
        f_transform_shifted = np.fft.fftshift(f_transform)
        if len(image.shape)==3:
            image = image.squeeze(0)
        # 设计高通滤波器
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        f_transform_shifted[crow - d:crow + d, ccol - d:ccol + d] = 0

        # 进行逆傅里叶变换
        f_transform = np.fft.ifftshift(f_transform_shifted)
        image_enhanced = np.fft.ifft2(f_transform) 
        image_enhanced = np.abs(image_enhanced)
        if len(image_enhanced.shape)==2:
            image_enhanced = np.expand_dims(image_enhanced,axis=0)
        return image_enhanced

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


def wiener_filter(image, kernel=np.ones((3, 3)) / 9, noise_var=25):
    # 使用傅里叶变换将图像和卷积核转换到频域
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)

    # 计算Wiener滤波器的频域表示
    wiener_filter_fft = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + noise_var)

    # 应用Wiener滤波器
    filtered_image_fft = image_fft * wiener_filter_fft

    # 进行逆傅里叶变换以获得滤波后的图像
    filtered_image = np.fft.ifft2(filtered_image_fft)
    filtered_image = np.abs(filtered_image)
    filtered_image = np.float32(filtered_image.real)

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