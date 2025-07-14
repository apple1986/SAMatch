# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:29:35 2020

@author: David
"""

import glob
import os
import json
import re
from tifffile import imread
import numpy as np
import h5py
from random import shuffle
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def tryint(s):                       
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):              
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):   
    return sorted(v_list, key=str2int)

# def getfilename(root_path):
#     for root, dirs, files in os.walk(root_path):
#         array = dirs
#         if array:
#             return array
        
def get_filename(data_path, pattern="*.png"):
    image_name = []
    data_image = os.path.join(data_path, pattern)
    for name in glob.glob(data_image, recursive=True):
        image_name.append(name)
    image_name = sort_humanly(image_name)
    return image_name


if __name__=='__main__':
    data_path = '/home/gxu/proj1/smatch/data/MRbrain/DICOM' # the whole brain path
    save_path = '/home/gpxu/vess_seg/vess_efficient/aping'
    # print(data_label)
    ## obtain file name
    image_name = get_filename(data_path, pattern="*.mat")
    image_name = sort_humanly(image_name)
    print(image_name[:10])

    # with open('whole_brain_name.txt','a') as f:
    #     for n in image_name:
    #         f.write(n + '\n')

    print('done')