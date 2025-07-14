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
import collections
from  segment_anything_main.segment_anything_main.segment_anything import sam_model_registry, SamPredictor

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
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
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
        
        sample["idx"] = idx
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
       
        
        #image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        
        image_strong_1 = image_weak
        # if random.random() < 0.8:
        #     image_strong_1 = color_jitter(image_weak).type("torch.FloatTensor")
        # else:
        #     image_strong_1 = torch.from_numpy(image_strong_1.astype(np.float32)).unsqueeze(0)
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
            #"image_strong":  image_strong,#torch.from_numpy(image_strong).unsqueeze(0),
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

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = [256,256]

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        #w2 = RandomFlip()
        w1 = Crop()
        image_weak,label = w1(image,label)
       # image_weak,label = w2(image,label)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        img = transforms.ToPILImage()(image_weak)
        # if random.random() < 0.8:
        #     image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        str_aug = img_trsform.strong_img_aug(3,True)
       # image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        image_strong_1 = image_weak
        # if random.random() < 0.8:
        image_strong = str_aug(img)
        transform_to_tensor = transforms.ToTensor()
        image_strong = transform_to_tensor(image_strong)
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)     
        label = torch.from_numpy(label.astype(np.uint8))

        #######################################
        cutmix_box1 = obtain_cutmix_box(256, p=0.5)
        cutmix_box2 = obtain_cutmix_box(256, p=0.5)
        #######################################
        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong":  image_strong,#torch.from_numpy(image_strong).unsqueeze(0),
            "label_weak": label,
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



class ToTensorAndNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert len(mean) == len(std)
        assert len(mean) == 3
        self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, in_image, in_label):
        in_image = Image.fromarray(np.uint8(in_image))
        image = self.normalize(self.to_tensor(in_image))
        label = torch.from_numpy(np.array(in_label, dtype=np.int32)).long()

        return image, label



def build_basic_transfrom(split="val", mean=[0.485, 0.456, 0.406]):
    ignore_label = 255
    trs_form = []
    if split != "val":
            trs_form.append(img_trsform.Resize(256,[1.0,1.5]))
            trs_form.append(img_trsform.RandomFlip(prob=0.5, flag_hflip=True))
            crop_size, crop_type = [256,256], 'rand'
            trs_form.append(img_trsform.Crop(crop_size, crop_type=crop_type, mean=mean, ignore_value=ignore_label))

    return img_trsform.Compose(trs_form)


# class RandomFlip(object):
#     def __init__(self, prob=0.5, flag_hflip=True,):
#         self.prob = prob
#         if flag_hflip:
#             self.type_flip = Image.FLIP_LEFT_RIGHT
#         else:
#             self.type_flip = Image.FLIP_TOP_BOTTOM
            
#     def __call__(self, in_image, in_label):
#         if random.random() < self.prob:
#             in_image = in_image.transpose(self.type_flip)
#             in_label = in_label.transpose(self.type_flip)
#         return in_image, in_label

class Crop(object):
    def __init__(self, crop_size=[256,256], crop_type="rand", mean=[0.485, 0.456, 0.406], ignore_value=255):
        if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
            self.crop_h, self.crop_w = crop_size
        elif isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            raise ValueError
        
        self.crop_type = crop_type
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.ignore_value = ignore_value

    def __call__(self, in_image, in_label):
        # Padding to return the correct crop size
        w, h = in_image.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, 
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(np.asarray(in_image, dtype=np.float32), 
                                       value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(np.asarray(in_label, dtype=np.int32), 
                                       value=self.ignore_value, **pad_kwargs)
            image = Image.fromarray(np.uint8(image))
            label = Image.fromarray(np.uint8(label))
        else:
            image = in_image
            label = in_label
        
        # cropping
        w, h = image.shape
        if self.crop_type == "rand":
            x = random.randint(0, w - self.crop_w)
            y = random.randint(0, h - self.crop_h)
        else:
            x = (w - self.crop_w) // 2
            y = (h - self.crop_h) // 2
        image = image[y:y+self.crop_h, x:x+self.crop_w]
        label = label[y:y+self.crop_h, x:x+self.crop_w]
        return image, label