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
#from  segment_anything_main.segment_anything_main.segment_anything import sam_model_registry, SamPredictor
import pywt

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
        self.names = []
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"
        # load all training samples
        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
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
            h5f = h5py.File(self._base_dir + "/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/volumes/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                if  self.transform :
                    sample = self.transform(sample)
                else :
                    self.transform = normal((256,256))
                    sample = self.transform(sample)   
        else:
            sample['image'] =  resize_val(image)
            sample['label'] = resize_val(label)
        sample["idx"] = idx # image: 1x256x256, label: 
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

def guideFilter(I, p, winSize = (5,5), eps = 0.01):

    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑

    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差

    a = cov_Ip / (var_I + eps)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

    q = mean_a * I + mean_b
    return q

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
        if random.random() < 0.5:
            image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
            label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        else:
             image_weak = transforms.ToPILImage()(image)
             label_aug = transforms.ToPILImage()(label)
        image_strong = image_weak
        image_strong_1 = image_weak
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        # if random.random() < 0.8:
        #     image_strong_1 = augmentations.cta_apply(image_weak, ops_strong)
        #################################################################
        # image_strong_numpy = np.array(image_strong)
        # image_weak_numpy = np.array(image_weak)
        # image_strong_numpy_g = guideFilter(image_strong_numpy, image_strong_numpy)
        
        # coeffs_s = pywt.dwt2(image_strong_numpy, 'db2')
        # coeffs_w = pywt.dwt2(image_weak_numpy, 'db2')
        # new_coeffs1 = (coeffs_s[0], (coeffs_w[1][0]+coeffs_s[1][0], coeffs_w[1][1]+coeffs_s[1][1], coeffs_w[1][2]+coeffs_s[1][2]))
        # new_coeffs2 = (coeffs_w[0], (coeffs_s[1][0], coeffs_s[1][1], coeffs_s[1][2]))
        # reconstructed_image1 = pywt.idwt2(new_coeffs1, 'db2')  
        # reconstructed_image2 = pywt.idwt2(new_coeffs2, 'db2')  
        #image_strong = Image.fromarray(image_strong_numpy_g)
        # image_fft_w = Image.fromarray(reconstructed_image2.astype(np.float32))
        ##################################################################
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()
        #######################################
        # cutmix_box1 = obtain_cutmix_box(self.output_size[0], p=0.5)
        # cutmix_box2 = obtain_cutmix_box(self.output_size[1], p=0.5)
        #######################################
        sample = {
            "image": image,
            "image_weak": to_tensor(image_weak),
            "image_strong":  to_tensor(image_strong),
            "label_weak": label_aug,
            "image_strong_1": to_tensor(image_strong_1),
            # "cutmix_w":cutmix_box1,
            # "cutmix_s":cutmix_box2,
            # "image_fft_s": to_tensor(image_fft_s),
            # "image_fft_w": to_tensor(image_fft_w),
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

def resize_val(image):
        b,x, y = image.shape
        return zoom(image, (1,256 / x, 256 / y), order=0)

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label) # rotation 90 and flip
        elif random.random() > 0.5:
            image, label = random_rotate(image, label) # roataion [-20, 20]
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class normal(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = image, label
       
        
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        
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
            "label_aug": label,
            "image_strong_1": image_strong_1,
            "cutmix_w":cutmix_box1,
            "cutmix_s":cutmix_box2
        }
        return sample
    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    weakly aug: change a little of an image; strong aug: change much of an image
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
        if random.random() > 0.5:
            image_weak, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image_weak, label = random_rotate(image, label)
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
            "label_aug": label,
            "image_strong_1": image_strong_1,
            "cutmix_w":cutmix_box1,
            "cutmix_s":cutmix_box2
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

class aug(object):
    def __init__(self):
        self.output_size = 256
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image_resize = self.resize(image)
        label_resize = self.resize(label)
        image_pil = transforms.ToPILImage()(image.astype(np.float32))
        label_pil = transforms.ToPILImage()(label.astype(np.float32))
        trs_weak = build_basic_transfrom()
        image_weak, label_weak = trs_weak(image_pil, label_pil)
        trs_strong = build_additional_strong_transform()
        image_strong = trs_strong(image_weak.convert('L'))
        image_weak = transforms.ToTensor()(image_weak)
        label_weak = transforms.ToTensor()(label_weak)
        image_strong = transforms.ToTensor()(image_strong)
        
        sample = {
            "image": torch.from_numpy(image_resize).unsqueeze(0),
            "label": torch.from_numpy(label_resize),
            "image_weak": image_weak,
            "image_strong":  image_strong,
            "label_weak": label_weak.squeeze(0),
        }
        return sample
    def resize(self, image):
        x, y = image.shape
        return zoom(image, (256 / x, 256 / y), order=0)
    
def build_basic_transfrom(mean=[0.485, 0.456, 0.406]):
    ignore_label = None
    trs_form = []
    trs_form.append(img_trsform.Resize([256,256], [1.0,1.5]))
    trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))
# crop also sometime for validating
    crop_size, crop_type = [256,256], "rand"
    trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=ignore_label))
    return img_trsform.Compose(trs_form)        
 
def build_additional_strong_transform():
    strong_aug_nums = 3
    flag_use_rand_num = True
    strong_img_aug = img_trsform.strong_img_aug(strong_aug_nums,
            flag_using_random_num=flag_use_rand_num)
    return strong_img_aug
       
        
        
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    # TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)
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


