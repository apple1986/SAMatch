## change .mat file to h5 format and save each volume as slices

import scipy.io as sio
import os
import utils.sortName as getname
import h5py

# set path
root_path = "/home/gxu/proj1/smatch/data/MRbrain"
img_path = os.path.join(root_path, "DICOM")
lab_path = os.path.join(root_path, "Labels")
img_lab_slice_path = os.path.join(root_path, "img_label_slice")
img_lab_volume_path = os.path.join(root_path, "img_label_volume")

# satatical all files
img_name = getname.get_filename(img_path, pattern="*.mat")
lab_name = getname.get_filename(lab_path, pattern="*.mat")
print(len(img_name))
# print(img_name[:10])

## convert .mat to h5
# load data
total_slice = 0
for one_img_path, one_lab_path in zip(img_name, lab_name):
    # basename
    img_basename = os.path.basename(one_img_path).split(".mat")[0]
    # lab_basename = os.path.basename(one_lab_path)
    # load data
    img_vol = sio.loadmat(one_img_path)["img"]
    lab_vol = sio.loadmat(one_lab_path)["label"]
    # save as volume
    vol_name = os.path.join(img_lab_volume_path, "subj"+img_basename +".h5")
    # save as h5
    hf = h5py.File(vol_name, 'w')
    hf.create_dataset('image', data=img_vol)
    hf.create_dataset('label', data=lab_vol)
    hf.close()

    # save as slice
    total_slice = total_slice + img_vol.shape[2]
    # read each slice 
    for n in range(0, img_vol.shape[2]): # how many slices in each case
        img = img_vol[:,:,n]
        lab = lab_vol[:,:,n]

        # save path
        data_save_name = os.path.join(img_lab_slice_path, "subj"+img_basename+"_s"+str(n)+".h5")
        # save as h5
        hf = h5py.File(data_save_name, 'w')
        hf.create_dataset('image', data=img)
        hf.create_dataset('label', data=lab)
        hf.close()

print("total slice number: {}".format(total_slice))

# ## test
# h5f = h5py.File(data_save_name, "r")
# image = h5f["image"][:]
# label = h5f["label"][:]
# print(image.shape)







